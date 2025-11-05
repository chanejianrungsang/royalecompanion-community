"""Elixir bubble detector for Clash Royale card replays.

This module detects purple elixir bubbles that appear above card slots when a card
is replayed. It reuses the CardSlotLocator for slot geometry only (no card detection),
defines a unified ROI spanning all slots, runs PaddleOCR once, and maps numeric values
back to their corresponding slot indices.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from cv.card_slot_locator import CardSlotLocator

LOGGER = logging.getLogger(__name__)


class ElixirBubbleDetector:
    """Detects elixir bubbles above card slots using OCR."""

    def __init__(
        self,
        *,
        paddle_kwargs: Optional[Dict] = None,
        confidence_threshold: float = 0.5,
        vertical_offset_above_slot: int = 80,
    ) -> None:
        """Initialize the detector.
        
        Args:
            paddle_kwargs: Optional kwargs for PaddleOCR initialization.
            confidence_threshold: Minimum OCR confidence to accept (0-1).
            vertical_offset_above_slot: Pixels above slot top to include in ROI.
        """
        self._ocr = PaddleOCR(use_angle_cls=False, lang="en", **(paddle_kwargs or {}))
        self._slot_locator = CardSlotLocator(paddle_kwargs=paddle_kwargs)
        self._confidence_threshold = confidence_threshold
        self._vertical_offset = vertical_offset_above_slot

    def detect_elixir_bubbles(
        self,
        frame: np.ndarray,
        *,
        return_debug: bool = False,
        identified_slots: Optional[set] = None,
        slot_boxes: Optional[list] = None,
    ) -> Tuple[List[Dict[str, int]], Optional[np.ndarray]]:
        """Detect elixir bubbles above card slots.
        
        This is a SECONDARY detection method - it only checks slots where cards have
        already been identified via primary detection (ResNet). Bubbles indicate when
        a previously-identified card is replayed, allowing us to track future plays
        beyond the initial identification.
        
        Args:
            frame: Input BGR image (ideally 900x1600 normalized).
            return_debug: If True, return visualization image.
            identified_slots: Set of slot indices (1-8) where cards have been identified.
                            If None or empty, no bubble detection is performed.
            
        Returns:
            Tuple of:
                - List of detections: [{"slot": slot_idx, "value": -elixir_cost}, ...]
                - Debug image (if return_debug=True), otherwise None
        """
        # Only run on identified slots (secondary detection)
        if identified_slots is None or not identified_slots:
            LOGGER.debug("No identified slots; skipping bubble detection (secondary detection only)")
            return [], None
        
        # Get slot layout from CardSlotLocator (reuse geometry only) ONLY if we
        # weren't provided boxes by the caller. Avoid running OCR here when the
        # tracker already has cached slot positions — locating slots is heavy
        # (PaddleOCR) and can block the main loop.
        slot_layout = None
        if slot_boxes and len(slot_boxes) >= 8:
            # Convert provided slot_boxes to expected SlotLayout-like structure
            try:
                class _SimpleLayout:
                    pass
                slot_layout = _SimpleLayout()
                slot_layout.slots = slot_boxes
                slot_layout.normalization = {}
            except Exception:
                slot_layout = None
        else:
            slot_layout, _ = self._slot_locator.locate_slots(frame, assume_normalized=True)
        
        if slot_layout is None or not getattr(slot_layout, 'slots', None):
            LOGGER.warning("No slots detected; cannot locate elixir bubbles")
            return [], None
        
        # Process each slot individually (only if already identified)
        all_detections = []
        for slot_idx, slot_box in enumerate(slot_layout.slots, start=1):
            # CRITICAL: Only check slots that have been identified via primary detection
            if slot_idx not in identified_slots:
                LOGGER.debug(f"Skipping slot {slot_idx} - not yet identified via primary detection")
                continue
            
            # Define ROI for this specific slot
            roi_coords, roi_offset = self._define_slot_roi(slot_box, frame.shape)
            if roi_coords is None:
                continue
            
            x_left, y_top, x_right, y_bottom = roi_coords
            roi = frame[y_top:y_bottom, x_left:x_right]
            
            if roi.size == 0:
                continue
            
            # Run OCR on this slot's ROI
            ocr_detections = self._run_ocr_on_roi(roi)
            
            # Parse detections for this slot
            for text, conf, box in ocr_detections:
                if conf < self._confidence_threshold:
                    continue
                
                elixir_value = self._parse_elixir_text(text)
                if elixir_value is not None:
                    all_detections.append({
                        "slot": slot_idx,
                        "value": -elixir_value,
                        "text": text,
                        "confidence": conf,
                    })
                    LOGGER.info(f"Detected elixir bubble: slot={slot_idx}, value=-{elixir_value}, text='{text}', conf={conf:.3f}")
                    break  # Only take first valid detection per slot
        
        # Optional debug visualization
        debug_img = None
        if return_debug:
            debug_img = self._draw_debug(
                frame,
                slot_layout.slots,
                all_detections,
            )
        
        return all_detections, debug_img
    
    def _define_slot_roi(
        self,
        slot_box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[int, int]]:
        """Define ROI for a single slot.
        
        Args:
            slot_box: (x1, y1, x2, y2) for the slot.
            frame_shape: Shape of the frame.
            
        Returns:
            Tuple of (roi_coords, offset) where roi_coords is (x_left, y_top, x_right, y_bottom)
            and offset is (x_offset, y_offset)
        """
        x1, y1, x2, y2 = slot_box
        
        # Bubble appears above the slot
        # Extend upward from slot top
        y_top = max(0, y1 - self._vertical_offset)
        y_bottom = y1 + 10  # Include a bit of slot top
        
        # Extend horizontally slightly beyond slot bounds
        slot_width = x2 - x1
        x_left = max(0, x1 - int(slot_width * 0.1))
        x_right = min(frame_shape[1], x2 + int(slot_width * 0.1))
        
        # Validate
        if y_top >= y_bottom or x_left >= x_right:
            return None, (0, 0)
        
        return (x_left, y_top, x_right, y_bottom), (x_left, y_top)

    def _define_roi(
        self,
        slot_boxes: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Define ROI covering all slots from slot tops upward.
        
        Args:
            slot_boxes: List of (x1, y1, x2, y2) for each slot.
            frame_shape: Shape of the frame (height, width, ...).
            
        Returns:
            (x_min, y_top, x_max, y_bottom) or None if invalid.
        """
        if not slot_boxes:
            return None
        
        height = frame_shape[0]
        
        # Horizontal bounds: from slot 1 left to slot 8 right
        x_min = slot_boxes[0][0]
        x_max = slot_boxes[-1][2]
        
        # Vertical bounds: from frame top (or offset) down to slot tops
        slot_tops = [box[1] for box in slot_boxes]
        min_slot_top = int(min(slot_tops))
        
        # Start from top of frame (bubbles appear very high)
        y_top = 0
        # Go down to just below slot tops to ensure we capture bubbles
        y_bottom = min_slot_top + 20  # Include a bit of the slot for context
        
        # Validate bounds
        if y_top >= y_bottom or x_min >= x_max:
            return None
        if y_bottom > height or x_max > frame_shape[1]:
            return None
        
        return (x_min, y_top, x_max, y_bottom)

    def _run_ocr_on_roi(self, roi: np.ndarray) -> List[Tuple[str, float, np.ndarray]]:
        """Run PaddleOCR on the ROI and extract text detections.
        
        Args:
            roi: Region of interest image.
            
        Returns:
            List of (text, confidence, bounding_box) tuples.
        """
        detections: List[Tuple[str, float, np.ndarray]] = []
        
        try:
            # Try new predict() method first (PaddleOCR v3+)
            if hasattr(self._ocr, 'predict'):
                result = self._ocr.predict(roi)
            else:
                result = self._ocr.ocr(roi, cls=False)
        except (TypeError, AttributeError) as exc:
            if "cls" in str(exc):
                LOGGER.debug("PaddleOCR backend does not support cls kwarg; retrying")
                result = self._ocr.ocr(roi)
            else:
                raise
        except ValueError as exc:
            # Handle "axes don't match array" error from document preprocessor
            if "axes don't match array" in str(exc):
                LOGGER.debug(f"PaddleOCR preprocessing failed (shape issue): {exc}")
                return detections  # Return empty list, skip this frame
            else:
                raise
        
        if not result:
            return detections
        
        # Parse result based on PaddleOCR version
        for line in result:
            # Handle new format (PaddleOCR v3+)
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
                    box = self._parse_polygon(poly)
                    if box is not None:
                        detections.append((str(text).strip(), float(score), box))
                continue
            
            # Handle old format (list of [[box], (text, conf)])
            if not isinstance(line, (list, tuple)):
                continue
            for item in line:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                box_data, value = item
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                text, confidence = value
                if not text:
                    continue
                box = self._parse_polygon(box_data)
                if box is not None:
                    detections.append((str(text).strip(), float(confidence), box))
        
        return detections

    def _parse_polygon(self, vertices: object) -> Optional[np.ndarray]:
        """Convert polygon vertices to numpy array.
        
        Args:
            vertices: Polygon vertices in various formats.
            
        Returns:
            Numpy array of shape (4, 2) or None if invalid.
        """
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

    def _filter_and_map_to_slots(
        self,
        ocr_detections: List[Tuple[str, float, np.ndarray]],
        slot_boxes: List[Tuple[int, int, int, int]],
        roi_offset: Tuple[int, int],
    ) -> List[Dict[str, int]]:
        """Filter OCR results and map to slot indices.
        
        Args:
            ocr_detections: List of (text, confidence, box) from OCR.
            slot_boxes: List of slot bounding boxes.
            roi_offset: (x_offset, y_offset) to convert ROI coords to frame coords.
            
        Returns:
            List of bubble detections: [{"slot": idx, "value": -cost}, ...]
        """
        from collections import defaultdict
        
        x_offset, y_offset = roi_offset
        
        # Filter valid numeric detections
        valid_detections = []
        for text, conf, box in ocr_detections:
            # Check confidence
            if conf < self._confidence_threshold:
                continue
            
            # Parse the text to extract elixir value
            # OCR may read "-3" as "3", "43", "-3", "3-", etc.
            elixir_value = self._parse_elixir_text(text)
            if elixir_value is None:
                continue
            
            # Convert box coordinates from ROI to frame coordinates
            box_frame = box + np.array([x_offset, y_offset])
            center_x = float(np.mean(box_frame[:, 0]))
            center_y = float(np.mean(box_frame[:, 1]))
            
            valid_detections.append({
                "text": text,
                "value": elixir_value,
                "center_x": center_x,
                "center_y": center_y,
                "box": box_frame,
            })
        
        # Map detections to slots based on horizontal position
        slot_detections = []
        for det in valid_detections:
            slot_idx = self._find_slot_for_detection(det["center_x"], slot_boxes)
            if slot_idx is not None:
                det["slot"] = slot_idx
                slot_detections.append(det)
        
        # Handle duplicates: keep the detection closest (lowest y) to slot top
        best_per_slot = defaultdict(lambda: {"value": None, "y": float("inf")})
        
        for det in slot_detections:
            slot_idx = det["slot"]
            if det["center_y"] < best_per_slot[slot_idx]["y"]:
                best_per_slot[slot_idx] = {
                    "value": det["value"],
                    "y": det["center_y"],
                }
        
        # Build final results with negative values
        results = []
        for slot_idx in sorted(best_per_slot.keys()):
            value = best_per_slot[slot_idx]["value"]
            if value is not None:
                results.append({"slot": slot_idx, "value": -value})
        
        return results
    
    def _parse_elixir_text(self, text: str) -> Optional[int]:
        """Parse OCR text to extract elixir value.
        
        Handles common OCR errors:
        - "3" → 3
        - "-3" → 3
        - "43" → 3 (dash merged with digit)
        - "3-" → 3
        - "S" → 5 (S looks like 5)
        - "O" or "o" → 0 (but 0 is invalid, so rejected)
        - etc.
        
        Args:
            text: OCR text string.
            
        Returns:
            Elixir value (1-10) or None if invalid.
        """
        text = text.strip()
        
        # Handle specific OCR character confusions
        # S is often misread as 5
        if text.upper() == "S":
            return 5
        
        # Remove any minus signs
        text_clean = text.replace('-', '').replace('_', '').strip()
        
        # Check if it's a valid digit
        if not text_clean:
            return None
        
        # Handle cases like "43" (dash merged with 3)
        # In game, elixir costs are 1-10, most commonly 1-6
        # If we see "43", it's likely "-3"
        # If we see "42", it's likely "-2"
        # If we see "45", it's likely "-5"
        if text_clean == "43":
            return 3
        elif text_clean == "42":
            return 2
        elif text_clean == "44":
            return 4
        elif text_clean == "45":
            return 5
        elif text_clean == "46":
            return 6
        elif text_clean == "47":
            return 7
        elif text_clean == "48":
            return 8
        elif text_clean == "49":
            return 9
        elif text_clean == "410":
            return 10
        
        # Try standard parsing
        if text_clean.isdigit():
            value = int(text_clean)
            if 1 <= value <= 10:
                return value
        
        # Try parsing with length constraint
        if len(text_clean) <= 2 and text_clean.isdigit():
            value = int(text_clean)
            if 1 <= value <= 10:
                return value
        
        return None

    def _find_slot_for_detection(
        self,
        center_x: float,
        slot_boxes: List[Tuple[int, int, int, int]],
    ) -> Optional[int]:
        """Find which slot a detection belongs to based on horizontal position.
        
        Args:
            center_x: Horizontal center of detection.
            slot_boxes: List of slot bounding boxes.
            
        Returns:
            Slot index (1-8) or None if not within any slot.
        """
        for idx, (x1, y1, x2, y2) in enumerate(slot_boxes, start=1):
            if x1 <= center_x <= x2:
                return idx
        
        return None

    def _draw_debug(
        self,
        frame: np.ndarray,
        slot_boxes: List[Tuple[int, int, int, int]],
        detections: List[Dict],
    ) -> np.ndarray:
        """Draw debug visualization.
        
        Args:
            frame: Original frame.
            slot_boxes: Slot bounding boxes.
            detections: List of bubble detections.
            
        Returns:
            Annotated debug image.
        """
        debug = frame.copy()
        
        # Draw slot boxes
        for idx, (x1, y1, x2, y2) in enumerate(slot_boxes, start=1):
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                debug,
                f"Slot {idx}",
                (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            
            # Draw ROI for this slot
            y_top = max(0, y1 - self._vertical_offset)
            y_bottom = y1 + 10
            x_left = max(0, x1 - int((x2-x1) * 0.1))
            x_right = min(debug.shape[1], x2 + int((x2-x1) * 0.1))
            cv2.rectangle(debug, (x_left, y_top), (x_right, y_bottom), (255, 255, 0), 1)
        
        # Draw detections
        for det in detections:
            slot_idx = det["slot"]
            value = det["value"]
            x1, y1, x2, y2 = slot_boxes[slot_idx - 1]
            
            # Draw marker above slot
            marker_x = (x1 + x2) // 2
            marker_y = y1 - 10
            cv2.circle(debug, (marker_x, marker_y), 5, (0, 0, 255), -1)
            cv2.putText(
                debug,
                str(value),
                (marker_x - 10, marker_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        
        return debug


def detect_elixir_bubbles(
    frame: np.ndarray,
    paddle_slot_locator: CardSlotLocator,
    ocr: PaddleOCR,
) -> List[Dict[str, int]]:
    """Standalone function to detect elixir bubbles.
    
    This is a convenience wrapper that matches the signature in the requirements.
    
    Args:
        frame: Input BGR image.
        paddle_slot_locator: CardSlotLocator instance (for slot geometry only).
        ocr: PaddleOCR instance.
        
    Returns:
        List of detections: [{"slot": slot_idx, "value": -elixir_cost}, ...]
    """
    detector = ElixirBubbleDetector()
    detector._slot_locator = paddle_slot_locator
    detector._ocr = ocr
    detections, _ = detector.detect_elixir_bubbles(frame)
    return detections
