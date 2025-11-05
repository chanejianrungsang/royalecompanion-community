"""Card slot locator that aligns with OCR-based anchors.

This module normalizes frames to 9:16 portrait, runs PaddleOCR to locate the
"Cards" and "Played" labels, and predicts the eight card slot rectangles using
scale-invariant ratios derived from reference measurements.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

# Ratios derived from canonical 900x1600 reference frame
SLOT_H_RATIO = 117 / 1600


# Anchor alignment parameters (ratios based on canonical 900x1600 frame)
TOP_OFFSET_ABOVE_CARDS_PX = 1  # Slot top sits one pixel above the cards label in the reference frame
BOTTOM_PADDING_RATIO = 69 / 117  # Slot bottom extends this fraction of slot height below the played baseline
# Cached anchor shift (in normalized pixels). Default is zero (no shift).
ANCHOR_BASE_SHIFT_RATIO = 0.0  # Use explicit per-anchor shift override
ANCHOR_SCALE_GAIN_PX = 0.0  # Disable scale-based shift as well
CACHED_ANCHOR_SHIFT_PX = 0  # Override shift when non-zero


LOGGER = logging.getLogger(__name__)

@dataclass
class OcrAnchor:
    """Stores OCR anchor information for a detected label."""

    text: str
    box: np.ndarray  # shape (4, 2)
    confidence: float

    @property
    def rightmost_x(self) -> int:
        return int(round(float(np.max(self.box[:, 0]))))

    @property
    def leftmost_x(self) -> int:
        return int(round(float(np.min(self.box[:, 0]))))

    @property
    def top_y(self) -> int:
        return int(round(float(np.min(self.box[:, 1]))))

    @property
    def bottom_y(self) -> int:
        return int(round(float(np.max(self.box[:, 1]))))

@dataclass
class SlotLayout:
    """Holds computed slot geometry."""

    slots: List[Tuple[int, int, int, int]]
    slot_size: Tuple[int, int]
    spacing: int
    cards_anchor_x: int
    played_anchor_x: int
    normalization: Dict[str, Tuple[int, int]]

@dataclass
class NormalizationInfo:
    """Tracks how an image was normalized before OCR."""

    original_size: Tuple[int, int]
    portrait_size: Tuple[int, int]
    resize_target: Tuple[int, int]
    x_offset: float
    y_offset: float

@dataclass(frozen=True)
class SlotCalibration:
    """Raw-space calibration describing slot layout."""

    name: str
    slot_width_main_raw: float
    slot_width_tail_raw: float
    gap_common_raw: float
    gap_extra_raw: float
    cards_slot_offset_raw: float
    played_slot_offset_raw: float
    cards_anchor_shift_raw: float
    played_anchor_shift_raw: float


CANONICAL_CALIBRATION = SlotCalibration(
    name="canonical_900x1600",
    slot_width_main_raw=96.0,
    slot_width_tail_raw=96.0,
    gap_common_raw=1.0,
    gap_extra_raw=1.0,
    cards_slot_offset_raw=9.0,
    played_slot_offset_raw=4.0,
    cards_anchor_shift_raw=4.0,
    played_anchor_shift_raw=4.0,
)

LIVE_CAPTURE_CALIBRATION = SlotCalibration(
    name="live_capture",
    slot_width_main_raw=60.0,
    slot_width_tail_raw=60.0,
    gap_common_raw=1.0,
    gap_extra_raw=1.0,
    cards_slot_offset_raw=6.0,
    played_slot_offset_raw=0.0,
    cards_anchor_shift_raw=4.0,
    played_anchor_shift_raw=4.0,
)

class CardSlotLocator:
    """Detects enemy card slots using PaddleOCR anchors."""

    def __init__(
        self,
        *,
        paddle_kwargs: Optional[Dict] = None,
        target_size: Tuple[int, int] = (900, 1600),
        background_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self._ocr = PaddleOCR(use_angle_cls=False, lang="en", **(paddle_kwargs or {}))
        self._target_size = target_size
        self._background_color = background_color

    def locate_slots(
        self,
        frame: np.ndarray,
        *,
        assume_normalized: bool = False,
        return_debug: bool = False,
    ) -> Tuple[Optional[SlotLayout], Optional[np.ndarray]]:
        """Locate eight card slots using OCR anchors.

        Args:
            frame: Input BGR image.
            assume_normalized: Skip ratio normalization when True.
            return_debug: When True, return visualization image.
        """
        debug_img = None
        if assume_normalized:
            normalized = cv2.resize(frame, self._target_size, interpolation=cv2.INTER_AREA)
            norm_info = NormalizationInfo(
                original_size=(frame.shape[1], frame.shape[0]),
                portrait_size=(frame.shape[1], frame.shape[0]),
                resize_target=self._target_size,
                x_offset=0.0,
                y_offset=0.0,
            )
        else:
            normalized, norm_info = self._normalize_to_portrait(frame)

        ocr_entries = self._run_paddle_ocr(normalized)
        if not ocr_entries:
            LOGGER.warning("No OCR entries detected; cannot compute slots")
            return None, debug_img

        anchors = self._extract_anchors(ocr_entries)
        if "cards" not in anchors or "played" not in anchors:
            LOGGER.warning("Missing required anchors: found=%s", list(anchors))
            return None, debug_img



        slot_layout = self._compute_slots_from_anchors(
            normalized,
            anchors,
            norm_info=norm_info,
            calibration=self._select_calibration(norm_info),
        )

        if return_debug:
            debug_img = self._draw_debug(
                normalized,
                anchors,
                slot_layout=slot_layout,
            )

        return slot_layout, debug_img

    def _select_calibration(self, norm_info: NormalizationInfo) -> SlotCalibration:
        """Choose between canonical and live-capture calibrations."""
        orig_width = norm_info.original_size[0]
        # Frames close to native 900x1600 keep the canonical calibration.
        if orig_width >= 850:
            return CANONICAL_CALIBRATION
        return LIVE_CAPTURE_CALIBRATION


    def _normalize_to_portrait(self, frame: np.ndarray) -> Tuple[np.ndarray, NormalizationInfo]:
        """Letterbox/crop the frame to enforce a 9:16 portrait aspect."""
        target_ratio = self._target_size[0] / self._target_size[1]
        orig_h, orig_w = frame.shape[:2]
        h, w = orig_h, orig_w
        target_width = max(1, int(round(h * target_ratio)))
        offset_x = 0.0
        offset_y = 0.0

        if w > target_width:
            delta = w - target_width
            left = delta // 2
            right = delta - left
            frame = frame[:, left : w - right]
            LOGGER.debug("Cropped width by (%d, %d)", left, right)
            offset_x += left
        elif w < target_width:
            delta = target_width - w
            left = delta // 2
            right = delta - left
            frame = cv2.copyMakeBorder(
                frame,
                0,
                0,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=self._background_color,
            )
            LOGGER.debug("Padded width by (%d, %d)", left, right)
            offset_x -= left

        h, w = frame.shape[:2]
        target_height = max(1, int(round(w / target_ratio)))

        if h > target_height:
            delta = h - target_height
            top = delta // 2
            bottom = delta - top
            frame = frame[top : h - bottom, :]
            LOGGER.debug("Cropped height by (%d, %d)", top, bottom)
            offset_y += top
        elif h < target_height:
            delta = target_height - h
            top = delta // 2
            bottom = delta - top
            frame = cv2.copyMakeBorder(
                frame,
                top,
                bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=self._background_color,
            )
            LOGGER.debug("Padded height by (%d, %d)", top, bottom)
            offset_y -= top

        portrait_w, portrait_h = frame.shape[1], frame.shape[0]

        portrait = cv2.resize(
            frame,
            self._target_size,
            interpolation=cv2.INTER_AREA if max(frame.shape[:2]) >= max(self._target_size) else cv2.INTER_CUBIC,
        )
        info = NormalizationInfo(
            original_size=(orig_w, orig_h),
            portrait_size=(portrait_w, portrait_h),
            resize_target=self._target_size,
            x_offset=offset_x,
            y_offset=offset_y,
        )
        return portrait, info

    def _run_paddle_ocr(self, image: np.ndarray) -> List[OcrAnchor]:
        """Run PaddleOCR on the top-left anchor region and return entries with bounding boxes."""

        if image.size == 0:
            return []

        height, width = image.shape[:2]
        x_max = max(1, int(round(width * 0.25)))
        y_max = max(1, int(round(height * 0.25)))
        # Restrict OCR to the scoreboard quadrant where the anchor text lives.
        ocr_region = np.ascontiguousarray(image[:y_max, :x_max])

        # Try new predict() method first (PaddleOCR v3+), fall back to old ocr() method
        result = None
        try:
            if hasattr(self._ocr, 'predict'):
                result = self._ocr.predict(ocr_region)
            else:
                result = self._ocr.ocr(ocr_region, cls=False)
        except (TypeError, AttributeError) as exc:
            if "cls" in str(exc):
                LOGGER.debug("PaddleOCR backend does not support cls kwarg; retrying without it")
                result = self._ocr.ocr(ocr_region)
            else:
                raise

        entries: List[OcrAnchor] = []
        if not result:
            return entries

        def _as_polygon(vertices: object) -> Optional[np.ndarray]:
            try:
                array = np.asarray(vertices, dtype=np.float32)
            except (TypeError, ValueError):
                return None
            if array.size == 0:
                return None
            if array.ndim == 1:
                if array.size % 2 != 0:
                    return None
                array = array.reshape(-1, 2)
            else:
                try:
                    array = array.reshape(-1, 2)
                except ValueError:
                    return None
            if array.shape[1] != 2:
                return None
            return array

        for line in result:
            texts = scores = polys = None
            if hasattr(line, "rec_texts"):
                texts = getattr(line, "rec_texts", None)
                scores = getattr(line, "rec_scores", None)
                polys = getattr(line, "rec_polys", None) or getattr(line, "dt_polys", None)
            elif isinstance(line, dict):
                texts = line.get("rec_texts")
                scores = line.get("rec_scores")
                polys = line.get("rec_polys") or line.get("dt_polys")

            if texts and scores and polys:
                for text, score, poly in zip(texts, scores, polys):
                    if not text:
                        continue
                    polygon = _as_polygon(poly)
                    if polygon is None:
                        continue
                    entries.append(
                        OcrAnchor(
                            text=str(text).strip(),
                            box=polygon,
                            confidence=float(score),
                        )
                    )
                continue

            if not isinstance(line, (list, tuple)):
                continue
            for item in line:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                box, value = item
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                text, confidence = value
                if not text:
                    continue
                polygon = _as_polygon(box)
                if polygon is None:
                    continue
                entries.append(
                    OcrAnchor(
                        text=str(text).strip(),
                        box=polygon,
                        confidence=float(confidence),
                    )
                )
        return entries

    def _anchor_line_x(
        self,
        anchor: OcrAnchor,
        *,
        norm_width: int,
        scale_norm_to_portrait_x: float,
        scale_portrait_to_raw_x: float,
        offset_x: float,
        scale_x: float,
        requested_shift_raw: float | None,
        base_shift_norm: float,
        scale_bonus_norm: float,
    ) -> Tuple[int, float, float, float, float, float]:
        """Convert anchor box into a normalized line position with calibrated raw shift."""
        if norm_width <= 0:
            norm_width = self._target_size[0]

        anchor_right_norm = float(np.max(anchor.box[:, 0]))
        anchor_right_portrait = anchor_right_norm * scale_norm_to_portrait_x
        anchor_right_raw = (anchor_right_portrait + offset_x) * scale_portrait_to_raw_x

        if requested_shift_raw is not None:
            shift_raw = float(requested_shift_raw)
        elif CACHED_ANCHOR_SHIFT_PX:
            shift_raw = float(CACHED_ANCHOR_SHIFT_PX) / max(scale_norm_to_portrait_x * scale_portrait_to_raw_x, 1e-6)
        else:
            base_shift_raw = base_shift_norm / max(scale_norm_to_portrait_x * scale_portrait_to_raw_x, 1e-6)
            scale_bonus_raw = scale_bonus_norm / max(scale_norm_to_portrait_x * scale_portrait_to_raw_x, 1e-6)
            shift_raw = max(0.0, base_shift_raw + scale_bonus_raw)

        line_raw = anchor_right_raw + shift_raw
        line_portrait = line_raw / scale_portrait_to_raw_x - offset_x
        line_norm = line_portrait / scale_norm_to_portrait_x

        line_norm = max(0.0, min(norm_width - 1.0, line_norm))
        line_int = int(round(line_norm))

        shift_norm = line_norm - anchor_right_norm

        return line_int, line_norm, line_raw, shift_norm, shift_raw, anchor_right_norm

    def _extract_anchors(self, entries: Sequence[OcrAnchor]) -> Dict[str, OcrAnchor]:
        """Pick the best matching OCR entry for each anchor word."""
        anchors: Dict[str, OcrAnchor] = {}
        for target_word in ("cards", "played"):
            best_entry: Optional[OcrAnchor] = None
            best_score: float = 0.0
            target_clean = "".join(ch for ch in target_word if ch.isalpha())

            for entry in entries:
                text_normalized = entry.text.lower().strip()
                if not text_normalized:
                    continue

                clean = "".join(ch for ch in text_normalized if ch.isalpha())
                if not clean:
                    continue

                ratio = 0.0

                if target_word in text_normalized:
                    ratio = 1.0
                else:
                    if target_word == "played":
                        if text_normalized in ["layed", "plaved", "playea", "piayed", "played"]:
                            ratio = 0.95
                        elif "layed" in text_normalized or "ayed" in text_normalized:
                            ratio = 0.9
                    elif target_word == "cards":
                        if text_normalized in ["cara", "card", "carda", "caras", "ards", "ads"]:
                            ratio = 0.95
                        elif "ards" in text_normalized or "ads" in text_normalized:
                            ratio = 0.9

                    if ratio == 0.0:
                        ratio = SequenceMatcher(None, clean, target_clean).ratio()

                if ratio < 0.55:
                    continue

                confidence = entry.confidence if entry.confidence is not None else 1.0
                score = ratio * confidence
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry:
                anchors[target_word] = best_entry
                LOGGER.debug("Anchor '%s' -> text='%s' conf=%.3f ratio_score=%.3f", target_word, best_entry.text, best_entry.confidence, best_score)
        return anchors

    def _compute_slots_from_anchors(
        self,
        image: np.ndarray,
        anchors: Dict[str, OcrAnchor],
        *,
        norm_info: NormalizationInfo,
        calibration: SlotCalibration,
    ) -> SlotLayout:
        """Compute slot rectangles from anchor boxes."""
        height, width = image.shape[:2]

        orig_w, orig_h = norm_info.original_size
        portrait_w, portrait_h = norm_info.portrait_size
        target_w, target_h = norm_info.resize_target
        offset_x = norm_info.x_offset
        offset_y = norm_info.y_offset

        scale_norm_to_portrait_x = portrait_w / target_w if target_w else 1.0
        scale_norm_to_portrait_y = portrait_h / target_h if target_h else 1.0
        scale_portrait_to_raw_x = orig_w / portrait_w if portrait_w else 1.0
        scale_portrait_to_raw_y = orig_h / portrait_h if portrait_h else 1.0
        scale_norm_to_raw_x = scale_norm_to_portrait_x * scale_portrait_to_raw_x
        scale_norm_to_raw_y = scale_norm_to_portrait_y * scale_portrait_to_raw_y
        scale_raw_to_norm_x = 1.0 / scale_norm_to_raw_x if scale_norm_to_raw_x else 0.0
        scale_raw_to_norm_y = 1.0 / scale_norm_to_raw_y if scale_norm_to_raw_y else 0.0

        reference_width = max(1, orig_w)
        scale_x = width / reference_width if reference_width else 1.0
        base_shift_norm = width * ANCHOR_BASE_SHIFT_RATIO
        scale_bonus_norm = max(scale_x - 1.0, 0.0) * ANCHOR_SCALE_GAIN_PX

        cards_anchor = anchors["cards"]
        played_anchor = anchors["played"]

        cards_line_x, cards_line_norm, cards_line_raw, cards_shift_norm, cards_shift_raw, cards_right_norm = self._anchor_line_x(
            cards_anchor,
            norm_width=width,
            scale_norm_to_portrait_x=scale_norm_to_portrait_x,
            scale_portrait_to_raw_x=scale_portrait_to_raw_x,
            offset_x=offset_x,
            scale_x=scale_x,
            requested_shift_raw=calibration.cards_anchor_shift_raw,
            base_shift_norm=base_shift_norm,
            scale_bonus_norm=scale_bonus_norm,
        )
        played_line_x, played_line_norm, played_line_raw, played_shift_norm, played_shift_raw, played_right_norm = self._anchor_line_x(
            played_anchor,
            norm_width=width,
            scale_norm_to_portrait_x=scale_norm_to_portrait_x,
            scale_portrait_to_raw_x=scale_portrait_to_raw_x,
            offset_x=offset_x,
            scale_x=scale_x,
            requested_shift_raw=calibration.played_anchor_shift_raw,
            base_shift_norm=base_shift_norm,
            scale_bonus_norm=scale_bonus_norm,
        )

        slot_width_main_norm = max(1, int(round(calibration.slot_width_main_raw / scale_norm_to_raw_x)))
        slot_width_tail_norm = max(1, int(round(calibration.slot_width_tail_raw / scale_norm_to_raw_x)))
        slot_h = max(1, int(round(SLOT_H_RATIO * height)))
        gap_common_norm = max(1, int(round(calibration.gap_common_raw / scale_norm_to_raw_x)))
        gap_extra_norm = max(1, int(round(calibration.gap_extra_raw / scale_norm_to_raw_x)))

        cards_raw = cards_line_raw
        played_raw = played_line_raw

        first_slot_raw = max(
            cards_raw + calibration.cards_slot_offset_raw,
            played_raw + calibration.played_slot_offset_raw,
        )
        first_slot_portrait = first_slot_raw / scale_portrait_to_raw_x - offset_x
        first_slot_x = int(round(first_slot_portrait / scale_norm_to_portrait_x))

        top_from_cards = max(0, cards_anchor.top_y - TOP_OFFSET_ABOVE_CARDS_PX)
        baseline = max(cards_anchor.bottom_y, played_anchor.bottom_y)
        bottom_padding = int(round(slot_h * BOTTOM_PADDING_RATIO))
        top_from_played = max(0, baseline + bottom_padding - slot_h)
        first_slot_y = max(top_from_cards, top_from_played)
        if first_slot_y + slot_h > height:
            first_slot_y = max(0, height - slot_h)

        slots: List[Tuple[int, int, int, int]] = []
        raw_offset = 0.0
        first_slot_x = None
        norm_slot_width_main = slot_width_main_norm
        norm_slot_width_tail = slot_width_tail_norm

        for idx in range(8):
            is_tail = idx >= 6
            slot_width_raw = calibration.slot_width_tail_raw if is_tail else calibration.slot_width_main_raw
            left_raw = first_slot_raw + raw_offset
            right_raw = left_raw + slot_width_raw

            left_portrait = left_raw / scale_portrait_to_raw_x - offset_x
            right_portrait = right_raw / scale_portrait_to_raw_x - offset_x

            x1 = int(round(left_portrait / scale_norm_to_portrait_x))
            x2 = int(round(right_portrait / scale_norm_to_portrait_x))

            if x2 <= x1:
                x2 = x1 + 1

            if x1 < 0:
                x1 = 0
            if x2 > width:
                x2 = width

            slots.append((x1, first_slot_y, x2, first_slot_y + slot_h))
            if first_slot_x is None:
                first_slot_x = x1
                norm_slot_width_main = x2 - x1

            if is_tail:
                norm_slot_width_tail = x2 - x1

            raw_offset += slot_width_raw
            if idx < 7:
                extra_raw = calibration.gap_extra_raw if idx == 5 else calibration.gap_common_raw
                raw_offset += extra_raw

        if not slots:
            return SlotLayout(
                slots=[],
                slot_size=(slot_width_main_norm, slot_h),
                spacing=slot_width_main_norm + gap_common_norm,
                cards_anchor_x=cards_line_x,
                played_anchor_x=played_line_x,
                normalization={},
            )

        first_slot_x = first_slot_x if first_slot_x is not None else slots[0][0]

        cards_offset = max(0, first_slot_x - cards_line_x)
        played_offset = max(0, first_slot_x - played_line_x)

        gap_common_px = gap_common_norm
        gap_extra_px = gap_extra_norm
        if len(slots) > 1:
            gap_common_px = max(0, slots[1][0] - slots[0][2])
        if len(slots) > 6:
            gap_extra_px = max(0, slots[6][0] - slots[5][2])

        cards_offset_raw = first_slot_raw - cards_raw
        played_offset_raw = first_slot_raw - played_raw

        normalization = {
            "calibration": calibration.name,
            "original": norm_info.original_size,
            "normalized": norm_info.resize_target,
            "portrait_size": norm_info.portrait_size,
            "target_size": norm_info.resize_target,
            "offset_x": norm_info.x_offset,
            "offset_y": norm_info.y_offset,
            "scale_x": scale_x,
            "scale_norm_to_raw_x": scale_norm_to_raw_x,
            "scale_norm_to_raw_y": scale_norm_to_raw_y,
            "scale_raw_to_norm_x": scale_raw_to_norm_x,
            "scale_raw_to_norm_y": scale_raw_to_norm_y,
            "scale_norm_to_portrait_x": scale_norm_to_portrait_x,
            "scale_norm_to_portrait_y": scale_norm_to_portrait_y,
            "scale_portrait_to_raw_x": scale_portrait_to_raw_x,
            "scale_portrait_to_raw_y": scale_portrait_to_raw_y,
            "anchor_shift_reference": reference_width,
            "anchor_base_shift_px": base_shift_norm,
            "anchor_scale_bonus_px": scale_bonus_norm,
            "cards_anchor_line_x": cards_line_x,
            "cards_anchor_line_norm": cards_line_norm,
            "cards_anchor_line_raw": cards_line_raw,
            "cards_anchor_shift_px": cards_shift_norm,
            "cards_anchor_shift_raw_px": cards_shift_raw,
            "cards_anchor_right_norm": cards_right_norm,
            "played_anchor_line_x": played_line_x,
            "played_anchor_line_norm": played_line_norm,
            "played_anchor_line_raw": played_line_raw,
            "played_anchor_shift_px": played_shift_norm,
            "played_anchor_shift_raw_px": played_shift_raw,
            "played_anchor_right_norm": played_right_norm,
            "cards_slot_offset_px": cards_offset,
            "played_slot_offset_px": played_offset,
            "cards_slot_offset_raw_px": cards_offset_raw,
            "played_slot_offset_raw_px": played_offset_raw,
            "top_from_cards_px": top_from_cards,
            "top_from_played_px": top_from_played,
            "slot_width_main_px": norm_slot_width_main,
            "slot_width_tail_px": norm_slot_width_tail,
            "slot_height_px": slot_h,
            "slot_width_main_raw_px": calibration.slot_width_main_raw,
            "slot_width_tail_raw_px": calibration.slot_width_tail_raw,
            "gap_common_px": gap_common_px,
            "gap_extra_px": gap_extra_px,
            "gap_common_raw_px": calibration.gap_common_raw,
            "gap_extra_raw_px": calibration.gap_extra_raw,
            "first_slot_x_px": first_slot_x,
            "first_slot_raw_px": first_slot_raw,
        }

        return SlotLayout(
            slots=slots,
            slot_size=(norm_slot_width_main, slot_h),
            spacing=norm_slot_width_main + gap_common_px,
            cards_anchor_x=cards_line_x,
            played_anchor_x=played_line_x,
            normalization=normalization,
        )
    def _draw_debug(
        self,
        image: np.ndarray,
        anchors: Dict[str, OcrAnchor],
        *,
        slot_layout: SlotLayout,
    ) -> np.ndarray:
        """Render anchor guide lines and optional slot rectangles."""
        debug = image.copy()
        cards_anchor = anchors.get("cards")
        played_anchor = anchors.get("played")

        height, width = debug.shape[:2]

        anchor_lines = {
            "Cards": slot_layout.cards_anchor_x,
            "Played": slot_layout.played_anchor_x,
        }

        for label, color, anchor in (
            ("Cards", (0, 255, 0), cards_anchor),
            ("Played", (0, 128, 255), played_anchor),
        ):
            x = anchor_lines.get(label)
            if x is None:
                continue
            cv2.line(debug, (x, 0), (x, height - 1), color, 1)
            cv2.putText(
                debug,
                f"{label} x={x}",
                (max(0, min(width - 60, x - 40)), 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            if anchor is not None:
                polygon = anchor.box.astype(int)
                cv2.polylines(debug, [polygon], isClosed=True, color=color, thickness=1)

        for idx, (x1, y1, x2, y2) in enumerate(slot_layout.slots, start=1):
            cv2.rectangle(debug, (x1, y1), (max(x1, x2 - 1), y2), (255, 0, 0), 1)
            cv2.putText(
                debug,
                str(idx),
                (x1 + 5, min(height - 5, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        return debug


