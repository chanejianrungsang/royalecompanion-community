"""
PaddleOCR-based match timer reader for Clash Royale replays.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from config.settings import TEMPLATES_DIR

TimerRegion = Dict[str, float]


class TimerOCR:
    """
    Locate the in-game timer via template matching and read values with PaddleOCR.

    The class keeps the last detected timer region so subsequent reads are fast,
    and it exposes helper methods for downstream elixir cross-checks.
    """

    def __init__(self, *, paddle_kwargs: Optional[Dict] = None) -> None:
        self.template_dir: Path = TEMPLATES_DIR / "match_timer"
        self.timer_templates: List[Dict[str, object]] = []
        self.timer_region: Optional[TimerRegion] = None
        self.last_timer_value: Optional[int] = None
        self.last_detection_time: float = 0.0

        ocr_kwargs = {"use_angle_cls": False, "lang": "en"}
        if paddle_kwargs:
            ocr_kwargs.update(paddle_kwargs)
        self.ocr = PaddleOCR(**ocr_kwargs)
        if hasattr(self.ocr, "use_doc_preprocessor"):
            self.ocr.use_doc_preprocessor = False
        self._load_timer_templates()

    # ------------------------------------------------------------------ Setup -
    def _load_timer_templates(self) -> None:
        """Load timer templates used to localise the clock region."""
        self.timer_templates = []

        if not self.template_dir.exists():
            print("TimerOCR: template directory not found:", self.template_dir)
            return

        for template_file in self.template_dir.glob("*.png"):
            template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
            if template is None:
                continue

            self.timer_templates.append(
                {
                    "image": template,
                    "name": template_file.stem,
                    "path": template_file,
                }
            )
            print(f"TimerOCR: loaded timer template {template_file.name}")

        print(f"TimerOCR: total templates loaded {len(self.timer_templates)}")

    # -------------------------------------------------------------- Detection -
    def locate_timer_region(self, frame: np.ndarray) -> Optional[TimerRegion]:
        """
        Find the timer region using template matching.

        Args:
            frame: RGB or BGR frame captured from the match.
        """
        if not self.timer_templates:
            return None

        frame_bgr = self._ensure_bgr(frame)
        frame_height, frame_width = frame_bgr.shape[:2]
        search_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        search_gray = search_gray[: frame_height // 2, :]  # upper half

        best_match: Optional[TimerRegion] = None
        best_confidence = 0.0
        scales: Sequence[float] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

        for template_info in self.timer_templates:
            template = template_info["image"]
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            for scale in scales:
                scaled = cv2.resize(
                    template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )

                if (
                    scaled.shape[0] < 5
                    or scaled.shape[1] < 5
                    or scaled.shape[0] >= search_gray.shape[0]
                    or scaled.shape[1] >= search_gray.shape[1]
                ):
                    continue

                result = cv2.matchTemplate(search_gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_confidence:
                    best_confidence = max_val
                    h, w = scaled.shape[:2]
                    best_match = {
                        "x": float(max_loc[0]),
                        "y": float(max_loc[1]),
                        "width": float(w),
                        "height": float(h),
                        "confidence": float(max_val),
                        "template": template_info["name"],
                        "scale": float(scale),
                    }

        if best_match and best_confidence >= 0.45:
            print(
                "TimerOCR: template match "
                f"{best_confidence:.3f} (template={best_match['template']}, "
                f"scale={best_match['scale']:.2f})"
            )
            self.timer_region = best_match
            return best_match

        return None

    # ------------------------------------------------------------------- OCR -
    def read_timer_value(self, frame: np.ndarray) -> Optional[int]:
        """
        Read the timer value from the frame using PaddleOCR.

        Returns an integer number of seconds remaining, or None if the value
        could not be determined.
        """
        if self.timer_region is None:
            if self.locate_timer_region(frame) is None:
                return None
        timer_region = self.timer_region
        if timer_region is None:
            return None

        frame_bgr = self._ensure_bgr(frame)
        x, y, w, h = self._expand_region(timer_region, frame_bgr.shape)
        timer_roi = frame_bgr[y : y + h, x : x + w]
        if timer_roi.size == 0:
            return None

        # Focus on the lower portion where the digits live; discard most of the label text.
        digit_start = int(timer_roi.shape[0] * 0.35)
        if digit_start > 0 and digit_start < timer_roi.shape[0] - 5:
            timer_roi = timer_roi[digit_start:, :]

        # Light preprocessing to help OCR: enlarge and sharpen contrast.
        roi_processed = self._prepare_roi(timer_roi)

        ocr_result = self.ocr.ocr(roi_processed)
        candidates = self._extract_text_candidates(ocr_result)
        if not candidates:
            return None

        for text, confidence in candidates:
            timer_seconds = self._parse_timer_text(text)
            if timer_seconds is not None:
                self.last_timer_value = timer_seconds
                self.last_detection_time = time.time()
                print(
                    f"TimerOCR: Paddle result '{text}' "
                    f"(confidence {confidence:.2f}) -> {timer_seconds}s"
                )
                return timer_seconds

        return None

    # --------------------------------------------------------------- Helpers -
    def _prepare_roi(self, roi: np.ndarray) -> np.ndarray:
        """Resize and enhance the ROI before OCR."""
        scaled = cv2.resize(roi, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 80, 255]))
        mask = cv2.bitwise_or(yellow_mask, white_mask)

        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )

        combined = cv2.bitwise_or(mask, adaptive)
        combined = cv2.medianBlur(combined, 3)
        combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    def _ensure_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2 or frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Heuristic: assume input from capture is BGR; if not, converting via RGB->BGR twice is a no-op.
        return frame

    def _expand_region(self, region: TimerRegion, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        height, width = shape[:2]
        pad_x = 15
        pad_top = 8
        pad_bottom = 47  # capture digits beneath the label

        x = max(0, int(region["x"]) - pad_x)
        y = max(0, int(region["y"]) - pad_top)
        region_width = int(region["width"])
        region_height = int(region["height"])

        w = min(width - x, region_width + pad_x * 2)
        bottom = min(height, y + region_height + pad_top + pad_bottom)
        h = bottom - y
        return x, y, w, h

    def _extract_text_candidates(self, ocr_result: Iterable) -> List[Tuple[str, float]]:
        candidates: List[Tuple[str, float]] = []
        if not ocr_result:
            return candidates

        def _add_candidate(entry: Sequence) -> None:
            if len(entry) < 2:
                return
            text_info = entry[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text = str(text_info[0])
                try:
                    conf = float(text_info[1])
                except (TypeError, ValueError):
                    conf = 0.0
                if text.strip():
                    candidates.append((text.strip(), conf))

        for item in ocr_result:
            if not item:
                continue
            if isinstance(item, dict):
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                for text, score in zip(texts, scores):
                    if text:
                        try:
                            conf = float(score)
                        except (TypeError, ValueError):
                            conf = 0.0
                        candidates.append((str(text).strip(), conf))
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(
                item[1], (list, tuple)
            ):
                _add_candidate(item)
                continue
            for sub in item:
                if isinstance(sub, (list, tuple)):
                    _add_candidate(sub)

        candidates.sort(key=lambda entry: entry[1], reverse=True)
        return candidates

    def _parse_timer_text(self, text: str) -> Optional[int]:
        """Convert OCR text into a number of seconds."""
        cleaned = re.sub(r"[^\d:]", "", text)
        if not cleaned:
            return None

        if ":" not in cleaned:
            return None

        match = re.match(r"(\d+):(\d{2})", cleaned)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds

        return None

    # ------------------------------------------------------------ Utilities -
    def get_match_elapsed_time(self, current_timer_seconds: Optional[int]) -> Optional[int]:
        if current_timer_seconds is None:
            return None
        elapsed = 180 - current_timer_seconds
        return max(0, elapsed)

    def cross_check_elixir_timing(
        self,
        elixir_start_time: Optional[float],
        current_elixir: float,
        current_timer_seconds: Optional[int],
    ) -> Optional[Dict[str, float]]:
        if current_timer_seconds is None or elixir_start_time is None:
            return None

        match_elapsed = self.get_match_elapsed_time(current_timer_seconds)
        if match_elapsed is None:
            return None

        elixir_elapsed = time.time() - elixir_start_time
        expected_elixir = min(10.0, 5.0 + match_elapsed * (1 / 2.8))
        elixir_difference = current_elixir - expected_elixir
        time_difference = elixir_elapsed - match_elapsed

        return {
            "match_elapsed": match_elapsed,
            "elixir_elapsed": elixir_elapsed,
            "expected_elixir": expected_elixir,
            "actual_elixir": current_elixir,
            "elixir_difference": elixir_difference,
            "time_difference": time_difference,
            "timer_seconds": current_timer_seconds,
            "needs_adjustment": abs(elixir_difference) > 0.5
            or abs(time_difference) > 1.0,
        }
