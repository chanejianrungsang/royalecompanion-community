#!/usr/bin/env python3
"""
Clash Royale Elixir Tracker - Main Application
Tracks opponent elixir by spectating another Clash Royale instance.

Features:
- Normal Mode: Window selection + elixir tracking via card detection
- Stealth Mode: Uses existing stopwatch-based detection system
"""

import sys
import cv2
import numpy as np
import re
import time
import json
from pathlib import Path
from typing import Optional, Tuple
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QComboBox, QTextEdit,
                            QGroupBox, QRadioButton, QButtonGroup, QProgressBar,
                            QFrame, QGridLayout, QCheckBox)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPixmap, QImage
import win32gui
from config.constants import INITIAL_ELIXIR, ELIXIR_PHASES
from config.settings import CARD_DATABASE_PATH, CARD_IMAGES_DIR, TEMPLATES_DIR

def find_template_in_image(image, template, threshold=0.7):
    """
    Find template in image using OpenCV multi-scale template matching
    Returns: (x, y, width, height, confidence) or None
    """
    # Convert to grayscale for matching
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if len(template.shape) == 3:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_template = template

    # Multi-scale template matching
    best_match = None
    best_confidence = 0.0

    # Try different scales
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        if scale != 1.0:
            template_scaled = cv2.resize(gray_template, None, fx=scale, fy=scale)
        else:
            template_scaled = gray_template

        if template_scaled.shape[0] > gray_image.shape[0] or template_scaled.shape[1] > gray_image.shape[1]:
            continue

        # Template matching
        result = cv2.matchTemplate(gray_image, template_scaled, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Debug logging for each scale
        print(f"[Template Debug] Scale {scale}: max_val={max_val:.3f}, threshold={threshold}")

        if max_val > best_confidence and max_val >= threshold:
            best_confidence = max_val
            h, w = template_scaled.shape[:2]
            best_match = (max_loc[0], max_loc[1], w, h, max_val)

    return best_match
import win32con
import win32ui
import ctypes
from ctypes import wintypes
from PIL import Image, ImageGrab
import time
import json
from pathlib import Path

# Import neural detection components
try:
    from cv.resnet_embedder import ResNetEmbedder
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("Neural detection not available - falling back to template matching")

# Import Paddle OCR for timer detection
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Paddle OCR not available - timer detection disabled")

# Import sophisticated elixir system
from cv.elixir_counter import ElixirCounter

# Import elixir bubble detector for spectator slot detection
try:
    from cv.elixir_bubble_detector import ElixirBubbleDetector
    BUBBLE_DETECTOR_AVAILABLE = True
except ImportError:
    BUBBLE_DETECTOR_AVAILABLE = False
    print("Elixir Bubble Detector not available")

# Card slot locator for aligning the snapshot crops
try:
    from cv.card_slot_locator import CardSlotLocator
    SLOT_LOCATOR_AVAILABLE = True
except Exception as exc:
    SLOT_LOCATOR_AVAILABLE = False
    print(f"Card slot locator unavailable - {exc}")

import logging

LOGGER = logging.getLogger(__name__)
# Removed print statements to avoid interfering with JSON stdout in headless mode

CARD_SLOT_HORIZONTAL_SHIFT = 0
BUBBLE_PURPLE_LOWER = np.array([120, 50, 100], dtype=np.uint8)
BUBBLE_PURPLE_UPPER = np.array([160, 255, 255], dtype=np.uint8)

try:
    from capture.windows_capture import WindowsBackgroundCapture
except ImportError:
    WindowsBackgroundCapture = None

class WindowCapture:
    """Minimal window capture helper for the UI demo."""

    def __init__(self) -> None:
        self.hwnd = None
        self.window_title = ""
        self.width = 0
        self.height = 0
        self.client_left = self.client_top = self.client_right = self.client_bottom = None
        self.client_width = self.client_height = None
        self._backend = WindowsBackgroundCapture() if WindowsBackgroundCapture else None

    def get_window_list(self):
        windows = []
        def _enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((hwnd, title))
            return True
        win32gui.EnumWindows(_enum_cb, None)
        windows.sort(key=lambda item: item[1].lower())
        return windows

    def set_target_window(self, hwnd) -> bool:
        try:
            if isinstance(hwnd, str):
                match = [(h, t) for h, t in self.get_window_list() if hwnd.lower() in t.lower()]
                if not match:
                    return False
                hwnd = match[0][0]
            self.hwnd = hwnd
            self.window_title = win32gui.GetWindowText(hwnd)
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            self.width = max(0, right - left)
            self.height = max(0, bottom - top)
            try:
                client_rect = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
                client_right, client_bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
                self.client_left, self.client_top = client_left, client_top
                self.client_right, self.client_bottom = client_right, client_bottom
                self.client_width = max(0, client_right - client_left)
                self.client_height = max(0, client_bottom - client_top)
            except win32gui.error:
                self.client_left = self.client_top = self.client_right = self.client_bottom = None
                self.client_width = self.client_height = None
            return True
        except Exception as exc:
            LOGGER.warning("Window capture setup failed: %s", exc)
            return False

    def capture_window(self, full_window: bool = False):
        frame = None
        if self._backend and self.hwnd:
            try:
                start = time.time()
                self._backend.hwnd = self.hwnd
                self._backend.width = self.width
                self._backend.height = self.height
                frame = self._backend.capture_background_window()
                dur = (time.time() - start) * 1000
                # Log only if capture is slow (>100ms) or empty
                if dur > 100:
                    LOGGER.warning(f"Window capture slow: {dur:.0f}ms")
                if frame is None or getattr(frame, 'size', 0) == 0:
                    LOGGER.warning("Window capture returned empty frame")
            except Exception:
                frame = None
        if (
            frame is None
            or getattr(frame, "size", 0) == 0
            or (hasattr(frame, "mean") and float(frame.mean()) < 1.0)
        ) and self.hwnd:
            try:
                left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                grab = ImageGrab.grab(bbox=(left, top, right, bottom))
                frame = cv2.cvtColor(np.array(grab), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                LOGGER.warning(f"ImageGrab fallback failed: {exc}")
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return frame

class ElixirTracker(QThread):
    """Stub tracker thread that feeds placeholder data to the UI."""

    frame_update = pyqtSignal(np.ndarray)
    elixir_update = pyqtSignal(float)
    card_detected = pyqtSignal(str, float)
    card_cycle_update = pyqtSignal(list, list)
    timer_update = pyqtSignal(str)
    status_update = pyqtSignal(str)
    # Signal to request the main UI to run a full-window snapshot detection
    request_snapshot = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.window_capture = WindowCapture()
        self.mode = "normal"
        self.running = False
        self.waiting_for_vs_clear = False
        self.vs_badge_seen = False
        self.game_started = False
        self.enemy_elixir = INITIAL_ELIXIR
        self.current_phase = "single"
        self.match_start_time = None
        self.last_elixir_update = time.time()
        self.card_slots = []
        self.opponent_cards_in_hand = ["unknown_png"] * 4
        self.opponent_upcoming_cards = ["unknown_png"] * 4
        self.opponent_card_cycle = ["unknown_png"] * 8
        self.card_cycle_costed = set()
        self.champion_played = None
        self.champion_alive = False
        self.locked_slots = {}
        self.slot_match_history = {}
        self.last_captured_frame = None
        self._card_debug_frame_saved = False
        self._card_debug_capture_time = None
        self.timer_merged_bbox = None
        self.timer_roi_bounds = None
        self.timer_detection_stage = "initial"
        self.timer_initialized = False
        self.timer_ocr_started = False
        self.vs_badge_seen_time = None
        self.timer_countdown_active = False
        self._last_timer_display = None
        self.ocr = None
        self.timer_debug_request = False
        self.last_timer_check = 0.0
        self.timer_check_interval = 10.0
        self.card_detection_interval = 1.0
        self.last_card_detection_time = 0.0
        self.vs_badge_detection_time = None
        
        # Motion detection for efficient ResNet gating
        self.slot_previous_frames = {}  # slot_idx -> 3x3 numpy array
        self.slot_change_threshold = 30  # Pixel difference threshold (lowered from 80 for spectator mode)
        self.motion_sensitivity = 1.5  # >1.25 = use 5x5 patch instead of 3x3 for better detection
        self.motion_check_interval = 0.2  # Check for motion every 200ms
        self.slot_last_card = {}  # slot_idx -> last detected card name
        self.slot_motion_detected_time = {}  # slot_idx -> time.time() when motion detected
        self.resnet_delay_after_motion = 1.0  # Wait 1.0s after motion before ResNet (100% accuracy in testing)
        
        # Replay detection settings (NEW)
        self.slot_identification_time = {}  # slot_idx -> time.time() when card first identified
        self.slot_last_replay_time = {}  # slot_idx -> time.time() of last replay detection
        self.initial_motion_grace_period = 2.5  # Ignore motion for 2.5s after ID (filters card settling animation)
        self.replay_cooldown = 2.0  # Min 2.0s between replay detections (prevents duplicates)
        
        # Motion debugging flags
        self.motion_debug = False  # Enable comprehensive motion detection logging
        self.disable_motion_gate = False  # Bypass motion detection (always run ResNet)
        
        # Log motion detection configuration on startup
        LOGGER.info("=" * 60)
        LOGGER.info("MOTION DETECTION CONFIGURATION")
        LOGGER.info(f"   Threshold: {self.slot_change_threshold} (lower = more sensitive)")
        LOGGER.info(f"   Sensitivity: {self.motion_sensitivity} (>1.25 = 5x5 patch)")
        LOGGER.info(f"   Check Interval: {self.motion_check_interval}s")
        LOGGER.info(f"   Animation Delay: {self.resnet_delay_after_motion}s (100% accuracy)")
        LOGGER.info(f"   Grace Period: {self.initial_motion_grace_period}s (filters settling animation)")
        LOGGER.info(f"   Replay Cooldown: {self.replay_cooldown}s (prevents duplicates)")
        LOGGER.info(f"   Motion Debug: {self.motion_debug}")
        LOGGER.info(f"   Motion Gate Enabled: {not self.disable_motion_gate}")
        LOGGER.info("=" * 60)
        
        # Smooth elixir display system with detection delay compensation
        self.target_enemy_elixir = INITIAL_ELIXIR  # Actual game state (calculations)
        self.displayed_enemy_elixir = INITIAL_ELIXIR  # Smoothly animated display value
        self.last_smooth_update_time = time.time()
        
        # Detection delay constants (will be calibrated dynamically)
        self.MOTION_DETECTION_OVERHEAD = 0.1  # 100ms for motion detection
        self.RESNET_INFERENCE_TIME = 0.5  # 500ms baseline (will be measured)
        self.RESNET_DETECTION_DELAY = self.MOTION_DETECTION_OVERHEAD + self.RESNET_INFERENCE_TIME  # Total delay
        self.BUBBLE_DETECTION_DELAY = 0.3  # 300ms: faster OCR-only detection
        
        # Smoothing configuration
        self.smooth_timer = None  # Will be initialized in run()
        self.SMOOTH_UPDATE_INTERVAL = 100  # 100ms = 10 FPS for smooth animations
        self.SMOOTH_ALPHA_REGEN = 0.9  # High alpha for smooth regeneration (faster convergence, less skipping)
        self.SMOOTH_ALPHA_DROP = 1.0  # Instant drop for card plays (no smoothing)
        
        self.elixir_counter = ElixirCounter()
        self.elixir_regeneration_started = False
        self.double_bonus_applied = False
        self._previous_elixir_phase = None
        
        # 9-elixir checkpoint alignment
        self.nine_elixir_checkpoint_done = False
        self.nine_elixir_tolerance = 0.5
        
        self.templates = {}
        self._load_vs_timer_templates()
        self.slot_locator = None
        if SLOT_LOCATOR_AVAILABLE:
            try:
                self.slot_locator = CardSlotLocator()
            except Exception as exc:
                print(f"Slot locator initialization failed: {exc}")
                self.slot_locator = None
        self.card_templates = {}
        self.card_templates_gray = {}
        self.card_aliases = {}
        self.dynamic_slot_vertical_offset = 0
        self.slot_deferral_streak = 0
        self._card_debug_frame_saved = False
        self._card_debug_capture_time = None
        self.unknown_template = None
        self.resnet_embedder = None
        self.UNKNOWN_EMBED_THRESHOLD = 0.8
        self.UNKNOWN_TEMPLATE_THRESHOLD = 0.5
        self.card_elixir_costs = {}
        self.bubble_detection_interval = 0.2
        self.last_bubble_detection_time = 0.0
        self.bubble_mask_threshold = 50
        self.bubble_roi_height_ratio = 0.45
        
        # Initialize new bubble detector system
        self.bubble_detector = None
        if BUBBLE_DETECTOR_AVAILABLE:
            try:
                self.bubble_detector = ElixirBubbleDetector()
                LOGGER.info("ElixirBubbleDetector initialized successfully")
            except Exception as exc:
                LOGGER.warning(f"ElixirBubbleDetector initialization failed: {exc}")
                self.bubble_detector = None
        
        # Bubble detection state tracking
        self.discovered_slots = set()  # Track which slots (1-8) we've discovered
        self.identified_slots = set()  # Track which slots have been identified via primary detection (ResNet)
        self.last_bubble_time = {}  # Cooldown tracking per slot (slot_num -> timestamp)
        self.bubble_cooldown_seconds = 2.0  # Prevent re-detecting same bubble
        
        # ThreadPoolExecutor for offloading expensive, blocking operations
        # (e.g., PaddleOCR-based CardSlotLocator). This prevents the
        # tracker main loop and UI from freezing while slot finding runs.
        try:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=1)
        except Exception:
            self._executor = None
        self._slot_find_future = None
        # One-time initial identification state: run a single ResNet pass across
        # all slots once they are discovered, before resuming motion-gated checks.
        self._initial_identification_started = False
        self._initial_identification_future = None
        self._initial_identification_done = False
        # Motion gate enabled by default (revert to event-driven ResNet).
        self.disable_motion_gate = False
        # Disable ResNet polling when motion gate is active
        self.resnet_poll_interval = 0.0
        self._last_resnet_poll = 0.0
        # Per-slot trailing detection history for sliding-window, trailing-weighted
        # consensus. Entries are (timestamp, name, score). We keep a short
        # history (e.g., ~6 samples → ~900ms at 150ms interval) and weight
        # recent samples higher when making consensus decisions.
        self.slot_detection_history = {}  # slot_idx -> list[(ts, name, score)]
        self.SLOT_HISTORY_MAX = 6
        self.SLOT_HISTORY_TAU = 0.35  # seconds: exponential decay half-life tuning
        self.SLOT_CONSENSUS_MIN_SAMPLES = 3
        self.SLOT_CONSENSUS_WEIGHTED_THRESHOLD = 0.75
        # Frame ingress diagnostics
        self._frame_ingress_log_count = 0
        self._frame_ingress_wait_logged = False
        
        # Problematic cards that require consensus before acceptance (prevent misidentification)
        # These cards are often confused with others, so we wait for multiple detections
        # Add card names EXACTLY as they appear in card_database.json (case-insensitive match)
        self.PROBLEMATIC_CARDS_REQUIRE_CONSENSUS = {
            "rg",  # Royal Giant - often confused with Knight
            # Add more cards here as you discover issues:
            # "pekka",        # Example: confused with Mini PEKKA
            # "megaknight",   # Example: confused with regular Knight
            # "minipekka",    # Example: confused with regular PEKKA
        }

        self._init_card_detection()
        self._load_card_costs()
        if OCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
                LOGGER.info("Timer OCR initialized")
            except Exception as exc:
                LOGGER.warning("Timer OCR failed to initialize: %s", exc)
                self.ocr = None
        else:
            LOGGER.info("Timer OCR not available - install paddleocr")

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_window(self, hwnd) -> bool:
        success = self.window_capture.set_target_window(hwnd)
        if success:
            # Reset frame-ingress diagnostics so the next capture cycle reports fresh data
            self._frame_ingress_log_count = 0
            self._frame_ingress_wait_logged = False
        return success

    def set_target_window(self, hwnd) -> bool:
        """Alias for compatibility with the headless backend IPC."""
        return self.set_window(hwnd)

    def _smooth_elixir_update(self) -> None:
        """
        Smooth elixir display update (called at 100ms intervals = 10 FPS).
        
        Strategy:
        - NO smoothing for regeneration (shows every 0.1 increment)
        - INSTANT updates for card plays (drops)
        
        This ensures the display shows every 0.1 elixir change during regeneration
        while maintaining instant response for card plays.
        
        Note: target_enemy_elixir is set by _update_elixir_regeneration()
        """
        current_time = time.time()
        
        # Use the target value set by _update_elixir_regeneration()
        # Don't read from counter again to avoid double-updating
        target_value = self.target_enemy_elixir
        
        # Calculate delta since last update
        delta = target_value - self.displayed_enemy_elixir
        
        # Debug: Log first few updates after regeneration starts
        if not hasattr(self, '_smooth_debug_count'):
            self._smooth_debug_count = 0
        if self.elixir_regeneration_started and self._smooth_debug_count < 30:
            self._smooth_debug_count += 1
            if self._smooth_debug_count % 5 == 0:
                LOGGER.info(f"[MOTION] Smooth update #{self._smooth_debug_count}: target={target_value:.3f}, displayed={self.displayed_enemy_elixir:.3f}, delta={delta:+.3f}")
        
        # For drops (card plays), snap instantly
        if delta < -0.5:
            # Card play detected - instant update
            self.displayed_enemy_elixir = target_value
        else:
            # For regeneration or small changes, update directly (no smoothing)
            # This ensures we display every 0.1 increment
            self.displayed_enemy_elixir = target_value
        
        # Round to 0.1 precision for display
        display_value = round(self.displayed_enemy_elixir, 1)
        
        # Initialize tracking variable if needed
        if not hasattr(self, '_last_rounded_display'):
            self._last_rounded_display = None  # Use None to ensure first update always emits
        
        # Debug: Log when rounded value changes
        if display_value != self._last_rounded_display and self._smooth_debug_count < 30:
            LOGGER.info(f"📊 Display changed: {self._last_rounded_display} → {display_value:.1f}")
        
        # Only emit to GUI if value actually changed (reduce GUI overhead by ~90%)
        if display_value != self._last_rounded_display:
            self.elixir_update.emit(display_value)
            self._last_rounded_display = display_value
        
        self.enemy_elixir = display_value  # Keep legacy variable synced
        
        self.last_smooth_update_time = current_time

    def run(self) -> None:
        self.running = True
        self.status_update.emit(f"Tracker thread running in {self.mode} mode (demo)")
        
        # Track previous state to avoid redundant GUI updates
        prev_cards_in_hand = None
        prev_upcoming_cards = None
        
        # Performance monitoring
        loop_count = 0
        slow_loops = 0
        
        while self.running:
            loop_start = time.time()

            if not self.window_capture.hwnd:
                time.sleep(0.1)
                continue

            if self._frame_ingress_wait_logged:
                self._frame_ingress_wait_logged = False
            
            frame = self.window_capture.capture_window()
            self.last_captured_frame = frame
            trigger_timer = False
            if frame is not None:
                self.frame_update.emit(frame)
                try:
                    trigger_timer = self._update_vs_detection_state(frame)
                except Exception as exc:
                    self.status_update.emit(f"VS detection error: {exc}")
                    trigger_timer = False
                try:
                    if self.timer_ocr_started or trigger_timer:
                        # Aggressive OCR during 2:53 → 2:52 transition for precise checkpoint
                        aggressive = self._should_use_aggressive_timer_ocr()
                        self.detect_match_timer(frame, aggressive=aggressive)
                except Exception as exc:
                    self.status_update.emit(f"Timer detection error: {exc}")
                self._update_timer_countdown()
                current_time = time.time()
                if current_time - getattr(self, "last_card_detection_time", 0.0) >= self.motion_check_interval:
                    card_start = time.time()
                    self._process_card_cycle(frame)
                    card_time = (time.time() - card_start) * 1000
                    if card_time > 50:  # Log if card processing takes >50ms
                        LOGGER.warning(f"Card detection took {card_time:.0f}ms (slow!)")
                    self.last_card_detection_time = current_time
                if current_time - getattr(self, "last_bubble_detection_time", 0.0) >= getattr(self, "bubble_detection_interval", 0.2):
                    try:
                        self._detect_bubble_plays(frame)
                    except Exception as exc:
                        self.status_update.emit(f"Bubble detection error: {exc}")
                    self.last_bubble_detection_time = current_time
            # Update pre-match elixir ramp (5.0 → 6.2) or normal regeneration
            self._update_prematch_elixir_ramp()
            self._update_elixir_alignment()  # Smooth alignment to timer-based targets
            self._update_elixir_regeneration()
            
            # Smooth elixir display update (interpolates displayed value toward target)
            self._smooth_elixir_update()
            
            # Only emit card cycle updates if cards actually changed (reduce GUI overhead)
            if (self.opponent_cards_in_hand != prev_cards_in_hand or 
                self.opponent_upcoming_cards != prev_upcoming_cards):
                self.card_cycle_update.emit(self.opponent_cards_in_hand, self.opponent_upcoming_cards)
                prev_cards_in_hand = list(self.opponent_cards_in_hand)
                prev_upcoming_cards = list(self.opponent_upcoming_cards)
            
            # Performance monitoring
            loop_time = (time.time() - loop_start) * 1000
            loop_count += 1
            if loop_time > 500:  # Log if loop takes >500ms
                slow_loops += 1
                LOGGER.warning(f"Main loop #{loop_count} took {loop_time:.0f}ms (target: 100ms)")
            
            self.msleep(100)  # 100ms = 10 FPS for smoother elixir counter
        self.status_update.emit("Tracker thread stopped")

    def stop(self) -> None:
        self.running = False
        self.elixir_regeneration_started = False
        # Shutdown background executor if present
        try:
            if getattr(self, '_executor', None):
                self._executor.shutdown(wait=False)
        except Exception:
            pass

    def reset_slot_locks(self):
        self.locked_slots.clear()
    
    def reset_motion_detection(self):
        """Reset motion detection state for new match."""
        self.slot_previous_frames.clear()
        self.slot_last_card.clear()
        self.slot_motion_detected_time.clear()
        
        # Clear replay detection tracking (NEW)
        self.slot_identification_time.clear()
        self.slot_last_replay_time.clear()
    
    def _should_use_aggressive_timer_ocr(self) -> bool:
        """Enable aggressive OCR from timer init until 9-elixir checkpoint completes."""
        # Aggressive OCR until checkpoint is done
        if self.nine_elixir_checkpoint_done:
            return False
        
        # Only if regeneration has started
        if not self.elixir_regeneration_started:
            return False
        
        # Only if timer has been initialized (so we have a starting point)
        if not self.timer_initialized:
            return False
        
        # Run aggressive OCR for all timer values until checkpoint
        return True

    def _has_slot_changed(self, slot_idx: int, slot_frame) -> bool:
        """
        Fast motion detection: Check if 3x3 center pixels changed.
        Returns True if motion detected (should run ResNet), False if unchanged.
        Records motion detection time for delayed ResNet scanning.
        """
        # ENHANCED DEBUG: Log crop frame info
        motion_debug = getattr(self, 'motion_debug', False)
        
        if motion_debug:
            crop_info = f"shape={slot_frame.shape if slot_frame is not None else 'None'}, size={slot_frame.size if slot_frame is not None else 0}"
            LOGGER.info(f"[MOTION] [MOTION DEBUG] Slot {slot_idx} _has_slot_changed() called - crop {crop_info}")
        
        if slot_frame is None or slot_frame.size == 0:
            LOGGER.warning(f"[WARN] [MOTION DEBUG] Slot {slot_idx} - EMPTY/NULL crop frame, returning True (will run ResNet)")
            return True  # Uncertain = run full detection
        
        # Extract a small center patch; default 3x3, optionally 5x5 when sensitivity is higher
        h, w = slot_frame.shape[:2]
        center_y, center_x = h // 2, w // 2
        patch_size = 3
        if getattr(self, 'motion_sensitivity', 1.0) > 1.25:
            patch_size = 5
        half = patch_size // 2

        # Safety check
        if center_y < half or center_x < half or center_y >= h - half or center_x >= w - half:
            LOGGER.warning(f"[WARN] [MOTION DEBUG] Slot {slot_idx} - crop too small for {patch_size}x{patch_size} patch (h={h}, w={w}), returning True")
            return True  # Uncertain = run full detection

        # Extract center region
        region = slot_frame[center_y-half:center_y+half+1, center_x-half:center_x+half+1].copy()
        
        if motion_debug:
            LOGGER.info(f"[MOTION] [MOTION DEBUG] Slot {slot_idx} - extracted {patch_size}x{patch_size} center patch from ({w}x{h}) crop")
        
        # First frame for this slot?
        if slot_idx not in self.slot_previous_frames:
            self.slot_previous_frames[slot_idx] = region.copy()  # CRITICAL: Must copy to avoid reference bug
            LOGGER.info(f"[NEW] [COPY-FIX-v2] Motion: Slot {slot_idx} - first frame cached with .copy(), will run ResNet")
            return True  # First time = run full detection
        
        # Compare with cached reference
        prev_region = self.slot_previous_frames[slot_idx]
        pixel_diff = np.abs(region.astype(np.float32) - prev_region.astype(np.float32))
        max_diff = np.max(pixel_diff)
        # Store the last max diff for diagnostics
        try:
            self._last_slot_max_diff = getattr(self, '_last_slot_max_diff', {})
            self._last_slot_max_diff[slot_idx] = float(max_diff)
        except Exception:
            pass

        # Enhanced logging: Always log when motion_debug enabled OR occasionally
        log_this_check = motion_debug
        if not hasattr(self, '_motion_log_counter'):
            self._motion_log_counter = 0
        self._motion_log_counter += 1
        if self._motion_log_counter % 50 == 0:
            log_this_check = True
            
        if log_this_check:
            LOGGER.info(f"[MOTION] [MOTION] Slot {slot_idx} max_diff={max_diff:.1f} (threshold={self.slot_change_threshold})")
        
        # Motion detected?
        if max_diff > self.slot_change_threshold:
            self.slot_previous_frames[slot_idx] = region.copy()  # CRITICAL: Must copy to avoid reference bug
            
            # Only set timestamp if this is FIRST motion (not a retry/continuation)
            # Once set, keep original timestamp until card is identified to preserve elixir backdate accuracy
            if slot_idx not in self.slot_motion_detected_time:
                self.slot_motion_detected_time[slot_idx] = time.time()  # Record FIRST motion time
                LOGGER.info(f"[MOTION] [v2-COPY-FIX] Motion: Slot {slot_idx} - FIRST DETECTION (diff={max_diff:.1f} > {self.slot_change_threshold})")
            else:
                # Subsequent motion detected (animation continuation or retry) - keep original timestamp
                original_time = self.slot_motion_detected_time[slot_idx]
                time_since_first = time.time() - original_time
                LOGGER.info(f"[MOTION] [v2-COPY-FIX] Motion: Slot {slot_idx} - CONTINUED (diff={max_diff:.1f}, first detected {time_since_first:.2f}s ago)")
            
            # UI message removed - motion detection working, no need to spam
            return True
        
        # Log when motion NOT detected (only in debug mode to avoid spam)
        if motion_debug:
            LOGGER.info(f"[MOTION] [MOTION DEBUG] Slot {slot_idx} max_diff={max_diff:.1f} < threshold={self.slot_change_threshold} — NO MOTION")
            # UI message removed - too spammy, logs are sufficient

        return False  # No motion = skip ResNet

    def detect_vs_badge(self, frame):
        """Detect VS badge to know when the match starts."""
        if frame is None or frame.size == 0:
            return False
        templates = self.templates.get('vs_badges')
        if not templates:
            return False
        try:
            search_frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception:
            search_frame = frame
        for template, name in templates:
            try:
                result = cv2.matchTemplate(search_frame, template, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.7:
                self.status_update.emit(f"VS badge detected ({name})")
                return True
        return False

    def _update_vs_detection_state(self, frame):
        """Handle VS badge detection and timer OCR gating."""
        if frame is None or frame.size == 0:
            return False
        now = time.time()
        trigger_timer = False
        if not self.vs_badge_seen:
            LOGGER.info("VS badge not seen yet - running detect_vs_badge() check")
            detected = False
            try:
                detected = self.detect_vs_badge(frame)
            except Exception as exc:
                LOGGER.warning("detect_vs_badge() raised: %s", exc)
            if detected:
                self.vs_badge_seen = True
                self.vs_badge_seen_time = now
                self.vs_badge_detection_time = now
                self.timer_ocr_started = False
                self.timer_initialized = False
                self.timer_countdown_active = False
                self.match_start_time = None
                self._last_timer_display = None
                self.timer_detection_stage = "initial"
                self.timer_merged_bbox = None
                self.timer_roi_bounds = None
                self.last_timer_check = 0.0
                self._prepare_for_new_match()
                # REMOVED: Pre-match ramp (now using reverse-calculated initial elixir)
                # self._start_prematch_elixir_ramp()
                self.status_update.emit("VS badge detected - waiting 2s before timer OCR")
            else:
                LOGGER.info("VS badge check: not detected")
        if self.vs_badge_seen and not self.timer_ocr_started and self.vs_badge_seen_time is not None:
            if now - self.vs_badge_seen_time >= 2.0:
                self.timer_ocr_started = True
                self.last_timer_check = 0.0
                self.last_card_detection_time = 0.0
                trigger_timer = True
                self.status_update.emit("Starting timer OCR & card detection sequence")
                self._start_elixir_regeneration()
        return trigger_timer

    def _prepare_for_new_match(self):
        """Reset elixir model when a new match is detected."""
        self.elixir_counter.reset()
        self.elixir_counter.enemy_elixir = INITIAL_ELIXIR
        self.elixir_regeneration_started = False
        self.double_bonus_applied = False
        self._previous_elixir_phase = None
        self.prematch_ramp_active = False
        self._aligned_at_255 = False
        self._aligned_at_252 = False
        if hasattr(self, '_elixir_alignment_target'):
            delattr(self, '_elixir_alignment_target')
        
        # Reset 9-elixir checkpoint state
        self.nine_elixir_checkpoint_done = False
        self._last_timer_for_checkpoint = None
        
        # Reset motion detection state (CRITICAL for new match)
        self.reset_motion_detection()
        
        # Reset card slots (will be found once at ~2:55 timer)
        self.card_slots = []
        
        # Reset bubble detection state
        self.discovered_slots.clear()
        self.identified_slots.clear()  # Clear primary detection tracking
        self.last_bubble_time.clear()
        self.status_update.emit("[RESET] Bubble detection reset - ready for new match")
        
        # Reset smooth elixir display system
        self.target_enemy_elixir = INITIAL_ELIXIR
        self.displayed_enemy_elixir = INITIAL_ELIXIR
        self.last_smooth_update_time = time.time()
        
        self.enemy_elixir = INITIAL_ELIXIR
        self.opponent_card_cycle = ["unknown_png"] * 8
        self.opponent_cards_in_hand = ["unknown_png"] * 4
        self.opponent_upcoming_cards = ["unknown_png"] * 4
        self.dynamic_slot_vertical_offset = 0
        self.slot_deferral_streak = 0
        self.card_cycle_costed.clear()
        self.last_bubble_detection_time = 0.0

    def _start_prematch_elixir_ramp(self):
        """Gradually increase elixir from 5.0 to 6.2 during pre-match phase."""
        self.prematch_ramp_active = True
        self.prematch_ramp_target = 6.2
        self.prematch_ramp_start_time = time.time()
        self.enemy_elixir = 5.0
        self._displayed_elixir = 5.0
        self.target_enemy_elixir = 5.0  # Set target for smooth update
        # Removed emit - _smooth_elixir_update() will handle display
        self.status_update.emit("Pre-match elixir ramp started (5.0 → 6.2)")
    
    def _update_prematch_elixir_ramp(self):
        """Smoothly ramp elixir from 5.0 to 6.2 in 0.15 increments (faster to complete in ~800ms)."""
        if not getattr(self, 'prematch_ramp_active', False):
            return
        
        if self.enemy_elixir < self.prematch_ramp_target:
            # Increment by 0.15 per 100ms (instead of 0.1) to reach 6.2 faster
            # 1.2 elixir / 0.15 = 8 updates = 800ms total
            self.enemy_elixir = min(self.enemy_elixir + 0.15, self.prematch_ramp_target)
            self._displayed_elixir = self.enemy_elixir
            self.target_enemy_elixir = self.enemy_elixir  # Update target for smooth display
            # Removed emit - _smooth_elixir_update() will handle display
        else:
            self.prematch_ramp_active = False
            self.status_update.emit(f"Pre-match ramp complete at {self.enemy_elixir:.1f}")

    def _start_elixir_regeneration(self):
        """Begin elixir regeneration with reverse-calculated initial elixir."""
        if self.elixir_regeneration_started:
            return
        
        # Stop pre-match ramp if still active (legacy system)
        if getattr(self, 'prematch_ramp_active', False):
            self.prematch_ramp_active = False
        
        try:
            # Start with reverse-calculated initial elixir (7.875 to hit 9.0 at checkpoint)
            self.elixir_counter.start_regeneration()
            
            # Sync display values with the counter's initial elixir (7.875)
            initial_elixir = self.elixir_counter.enemy_elixir
            self.target_enemy_elixir = initial_elixir
            self.displayed_enemy_elixir = initial_elixir
            self.enemy_elixir = initial_elixir
            
            # Reset last display to force next update to emit
            if hasattr(self, '_last_rounded_display'):
                self._last_rounded_display = None
            
        except Exception as exc:
            self.status_update.emit(f"Elixir regeneration failed to start: {exc}")
            return
        
        # Get the reverse-calculated initial elixir
        initial_elixir = self.elixir_counter.enemy_elixir
        
        self.elixir_regeneration_started = True
        self.double_bonus_applied = False
        self._previous_elixir_phase = self.elixir_counter.get_current_phase()
        self.enemy_elixir = initial_elixir
        self._displayed_elixir = initial_elixir
        self.target_enemy_elixir = initial_elixir  # Set target for smooth display
        # Removed emit - _smooth_elixir_update() will handle display
        self.status_update.emit(f"Elixir regeneration started at {initial_elixir:.2f} (reverse-calculated for 9.0 checkpoint)")
        # Immediately notify UI of the starting elixir so GUI doesn't stay at 5.0
        try:
            # Use unrounded value for internal sync, GUI will format to 0.1
            self.elixir_update.emit(initial_elixir)
        except Exception:
            pass

    def _update_elixir_regeneration(self):
        """Advance the elixir counter and apply phase-based bonuses."""
        # Debug: Count how many times this is called
        if not hasattr(self, '_regen_call_count'):
            self._regen_call_count = 0
        self._regen_call_count += 1
        
        if not self.elixir_regeneration_started:
            if self._regen_call_count % 50 == 0:  # Log every 5 seconds
                self.status_update.emit(f"[WARN] Elixir regen NOT started (called {self._regen_call_count} times)")
            return
        
        # Check if match_start_time is set
        if self.elixir_counter.match_start_time is None:
            self.status_update.emit("[WARN] Elixir update called but match_start_time is None!")
            return
        
        previous_phase = self._previous_elixir_phase
        previous_elixir = self.elixir_counter.get_enemy_elixir()
        
        # Debug: Log BEFORE update
        if hasattr(self, '_debug_card_detection_time'):
            time_since = time.time() - self._debug_card_detection_time
            if time_since < 3.0:
                print(f"[BEFORE UPDATE] enemy_elixir={previous_elixir:.3f}, match_start_time exists={self.elixir_counter.match_start_time is not None}")
        
        self.elixir_counter.update()
        current_phase = self.elixir_counter.get_current_phase()
        current_elixir = self.elixir_counter.get_enemy_elixir()
        
        # Debug: Log AFTER update
        if hasattr(self, '_debug_card_detection_time'):
            time_since = time.time() - self._debug_card_detection_time
            if time_since < 3.0:
                expected_total = self.elixir_counter._expected_total()
                print(f"[AFTER UPDATE] enemy_elixir={current_elixir:.3f}, expected={expected_total:.3f}, spent={self.elixir_counter.spent_elixir:.3f}")
                msg = f"[MOTION] Regen tick ({time_since:.1f}s): {previous_elixir:.2f} → {current_elixir:.2f} | Spent: {self.elixir_counter.spent_elixir:.1f} | Expected: {expected_total:.2f}"
                print(msg)  # Print directly to console
                self.status_update.emit(msg)
        
        # Log elixir changes every 5 seconds (50 updates at 100ms interval)
        if not hasattr(self, '_elixir_log_counter'):
            self._elixir_log_counter = 0
        self._elixir_log_counter += 1
        if self._elixir_log_counter >= 50:  # Every 5 seconds
            elapsed = time.time() - self.elixir_counter.match_start_time
            expected_total = self.elixir_counter._expected_total()
            self.status_update.emit(
                f"⚡ Elixir: {current_elixir:.1f} | Phase: {current_phase} | "
                f"Spent: {self.elixir_counter.spent_elixir:.1f} | Expected: {expected_total:.1f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            self._elixir_log_counter = 0
        
        if previous_phase != 'double' and current_phase == 'double' and not self.double_bonus_applied:
            self.elixir_counter.add_bonus(1.0)
            self.double_bonus_applied = True
            self.status_update.emit("Double elixir bonus applied (+1)")
        self._previous_elixir_phase = current_phase
        
        # Update target elixir for smooth display system
        # The _smooth_elixir_update() method will handle the actual display updates
        self.target_enemy_elixir = current_elixir

    def _smooth_align_elixir_to_timer(self, minutes: int, seconds: int):
        """
        Smoothly align elixir to match expected game state based on timer.
        
        Key timing points:
        - 2:55.00 → 8.0 elixir (5s into match)
        - 2:52.20 → 9.0 elixir (7.8s into match)
        """
        # Calculate expected elixir based on timer
        elapsed_seconds = 180 - (minutes * 60 + seconds)
        
        # Game starts with 6.22 effective elixir at 3:00 timer
        # (accounts for pre-match regeneration)
        initial_game_elixir = 6.22
        expected_elixir_raw = initial_game_elixir + (elapsed_seconds * (1.0 / 2.8))
        expected_elixir = min(expected_elixir_raw - self.elixir_counter.spent_elixir, 10.0)
        
        current_elixir = self.enemy_elixir
        diff = expected_elixir - current_elixir
        
        # Only align if difference is significant (> 0.3)
        if abs(diff) > 0.3:
            # Store alignment target for smooth transition
            if not hasattr(self, '_elixir_alignment_target'):
                self._elixir_alignment_target = expected_elixir
                self._elixir_alignment_start = current_elixir
                self._elixir_alignment_steps = 0
                self.status_update.emit(
                    f"🎯 Timer-based alignment: {current_elixir:.1f} → {expected_elixir:.1f} "
                    f"(at {minutes}:{seconds:02d}, spent: {self.elixir_counter.spent_elixir:.1f})"
                )
    
    def _update_elixir_alignment(self):
        """Smoothly transition elixir to alignment target in 0.1 increments."""
        if not hasattr(self, '_elixir_alignment_target'):
            return
        
        target = self._elixir_alignment_target
        current = self.enemy_elixir
        
        if abs(target - current) < 0.05:
            # Alignment complete
            self.enemy_elixir = target
            self._displayed_elixir = target
            self.target_enemy_elixir = target  # Set target for smooth display
            # Removed emit - _smooth_elixir_update() will handle display
            delattr(self, '_elixir_alignment_target')
            self.status_update.emit(f"[OK] Alignment complete at {self.enemy_elixir:.1f}")
        else:
            # Move 0.1 elixir per update toward target
            step = 0.1 if target > current else -0.1
            self.enemy_elixir = round(current + step, 1)
            self._displayed_elixir = self.enemy_elixir
            self.target_enemy_elixir = self.enemy_elixir  # Update target for smooth display
            # Removed emit - _smooth_elixir_update() will handle display

    def _check_nine_elixir_alignment(self, minutes: int, seconds: int):
        """
        One-time alignment check at 2:52 to sync elixir counter with game state.
        
        Game mechanics:
        - At 2:52.20 (exactly 7.8 seconds into match), opponent reaches 9.0 elixir
        - This is a fixed checkpoint in the game, regardless of when we started tracking
        - We must account for any elixir they spent between when we started and 2:52
        
        Strategy:
        - At 2:52, set the baseline to 9.0 elixir (game truth)
        - Subtract any cards they played before this checkpoint
        - Adjust spent_elixir to align tracked elixir to this corrected baseline
        
        Example:
        - At 2:52, baseline is 9.0 elixir
        - Player spent 3 elixir (Knight) at 2:54 (before checkpoint)
        - Expected current elixir: 9.0 - 3.0 = 6.0
        - Our tracker shows: 5.8 (slight drift)
        - Checkpoint corrects drift: adjusts spent_elixir to make display = 6.0
        
        """
        if self.nine_elixir_checkpoint_done:
            return
        
        # Get current state
        tracked_elixir = self.elixir_counter.get_enemy_elixir()
        total_spent = self.elixir_counter.spent_elixir  # How much they've spent since we started tracking
        
        # Fixed checkpoint: At 2:52, opponent has 9.0 elixir (before any spending)
        checkpoint_baseline = 9.0
        
        # Calculate expected elixir after accounting for spending
        expected_elixir_at_checkpoint = checkpoint_baseline - total_spent
        
        # Clamp to 0-10 range
        expected_elixir_at_checkpoint = max(0.0, min(expected_elixir_at_checkpoint, 10.0))
        
        # Safety check: If opponent has spent a LOT (>12 elixir), something is wrong
        if total_spent > 12.0:
            self.status_update.emit(
                f"[WARN] 9-Elixir checkpoint: Opponent spent {total_spent:.1f} elixir "
                f"before 2:52 (suspicious amount). Skipping alignment - possible detection errors."
            )
            return
        
        # Skip if opponent just played a card (tracked elixir is very low and spent is low)
        # This suggests they just spent their elixir and are in temporary low state
        if tracked_elixir < 1.0 and total_spent < 5.0:
            self.status_update.emit(
                f"⏭️ 9-Elixir checkpoint skipped: Tracked={tracked_elixir:.1f} too low "
                f"(opponent likely just played card). Will retry next timer read."
            )
            return
        
        # Calculate drift between expected (9.0 - spent) and what we're tracking
        drift = expected_elixir_at_checkpoint - tracked_elixir
        
        self.status_update.emit(
            f"🎯 9-Elixir Checkpoint (2:{seconds:02d}): "
            f"Baseline={checkpoint_baseline:.1f}, Spent={total_spent:.1f}, Expected={expected_elixir_at_checkpoint:.1f}, "
            f"Tracked={tracked_elixir:.1f}, Drift={drift:+.1f}"
        )
        
        # ALWAYS apply checkpoint correction - this is the authoritative game state
        # Override: Set expected_total to 9.0 (game truth at this checkpoint)
        # Backdate match_start_time to make expected_total = 9.0
        # expected_total = INITIAL_ELIXIR (5.0) + (elapsed × regen_rate)
        # 9.0 = 5.0 + (elapsed × 0.357)
        # elapsed = (9.0 - 5.0) / 0.357 ≈ 11.2 seconds
        import time
        target_elapsed_for_9_elixir = (checkpoint_baseline - INITIAL_ELIXIR) / 0.357
        new_match_start_time = time.time() - target_elapsed_for_9_elixir
        self.elixir_counter.match_start_time = new_match_start_time
        
        # Keep spent_elixir as-is - it already correctly tracks any spending before checkpoint
        # The formula enemy_elixir = expected_total - spent_elixir will now work correctly
        # Example: If they spent 3 elixir, spent_elixir=3.0, expected_total=9.0 → enemy_elixir=6.0
        
        # Recalculate enemy_elixir with new expected_total
        self.elixir_counter.update()
        
        corrected_elixir = self.elixir_counter.get_enemy_elixir()
        
        # Sync displayed elixir immediately to the corrected value
        self._displayed_elixir = round(corrected_elixir, 1)
        self.enemy_elixir = self._displayed_elixir
        self.target_enemy_elixir = self._displayed_elixir  # Update target for smooth display
        
        # Removed emit - _smooth_elixir_update() will handle display
        
        # Log the correction
        if abs(drift) > self.nine_elixir_tolerance:
            self.status_update.emit(
                f"[OK] Elixir aligned: {tracked_elixir:.1f} → {corrected_elixir:.1f} "
                f"(corrected {drift:+.1f} elixir, spent={self.elixir_counter.spent_elixir:.1f})"
            )
        else:
            self.status_update.emit(
                f"[OK] Elixir checkpoint applied: set to {corrected_elixir:.1f} "
                f"(drift {drift:+.1f} was within tolerance, but checkpoint is authoritative)"
            )
        
        # Mark as done - only do this once per match
        self.nine_elixir_checkpoint_done = True



    def _eligible_hand_cards(self):
        cards = []
        cycle = getattr(self, 'opponent_card_cycle', [])
        if not cycle:
            return cards
        state_elixir = self.elixir_counter.get_enemy_elixir()
        hand = cycle[:4]
        for idx, name in enumerate(hand):
            if name in {"unknown", "unknown_png"}:
                continue
            cost = self._get_card_cost(name) if hasattr(self, '_get_card_cost') else None
            if cost is None:
                cost = self.card_elixir_costs.get(name.lower())
            if cost is None:
                continue
            if cost <= state_elixir + 2:
                cards.append({"slot": idx, "name": name, "cost": cost})
        return cards

    def _detect_bubble_plays(self, frame):
        """
        Detect elixir bubbles using the new ElixirBubbleDetector.
        
        IMPORTANT: This is SECONDARY detection only. Bubbles are only checked for slots
        where cards have already been identified via primary detection (ResNet).
        This allows tracking of future replays without interfering with initial card discovery.
        
        Features:
        - Only runs on slots with identified cards (secondary detection)
        - Detects bubbles above deck slots (spectator view)
        - Prevents duplicate subtractions with per-slot cooldown
        - Tracks discovered slots for deck discovery progress
        """
        # Require match to be started
        if not self.elixir_counter.match_start_time:
            return
        
        # Require bubble detector to be available
        if self.bubble_detector is None:
            return
        
        # Skip if no slots have been identified yet via primary detection
        if not self.identified_slots:
            return
        
        try:
            # Run bubble detection ONLY on identified slots (secondary detection)
            detections, debug_frame = self.bubble_detector.detect_elixir_bubbles(
                frame,
                identified_slots=self.identified_slots,
                slot_boxes=self.card_slots if self.card_slots else None,
            )
            
            if not detections:
                return
            
            current_time = time.time()
            
            # Process each detection
            for detection in detections:
                slot = detection['slot']  # 1-8
                value = detection['value']  # Already negative (e.g., -3, -4, -5)
                confidence = detection['confidence']
                
                # Validate detected cost against known card cost
                detected_cost = abs(value)
                card_name = self.opponent_card_cycle[slot - 1] if slot <= len(self.opponent_card_cycle) else "unknown_png"
                
                # Get expected cost for this card
                expected_cost = self._get_card_cost(card_name)
                
                # If we know the card cost, validate the bubble cost matches
                if expected_cost is not None and detected_cost != expected_cost:
                    self.status_update.emit(
                        f"[WARN] Bubble validation failed: Slot {slot} ({card_name}) expects {expected_cost} elixir, "
                        f"but detected {detected_cost}. Skipping (likely OCR error or wrong slot)."
                    )
                    continue  # Skip this invalid detection
                
                # Cooldown check: prevent re-detecting same bubble within 2 seconds
                if slot in self.last_bubble_time:
                    time_since_last = current_time - self.last_bubble_time[slot]
                    if time_since_last < self.bubble_cooldown_seconds:
                        # Too soon - skip this detection
                        continue
                
                # Apply elixir subtraction with SECONDARY detection delay compensation (300ms)
                cost = abs(value)  # Get positive cost for logging
                
                # Calculate actual play time (accounting for bubble OCR delay)
                actual_play_time = current_time - self.BUBBLE_DETECTION_DELAY
                
                # Update to current state first
                self.elixir_counter.update()
                current_elixir = self.elixir_counter.get_enemy_elixir()
                
                # Calculate elixir at time of actual card play (300ms ago)
                if self.elixir_counter.match_start_time:
                    current_phase = self.elixir_counter.get_current_phase()
                    regen_rate = ELIXIR_PHASES[current_phase]
                    regen_during_detection = self.BUBBLE_DETECTION_DELAY * regen_rate
                    
                    # Elixir at play time = current - regeneration during detection
                    elixir_at_play = current_elixir - regen_during_detection
                    
                    # Calculate what elixir should be now
                    corrected_elixir = max(0.0, elixir_at_play - cost)
                    
                    # Update the elixir counter to the corrected value
                    expected_total = self.elixir_counter._expected_total()
                    self.elixir_counter.spent_elixir = max(0.0, expected_total - corrected_elixir)
                    self.elixir_counter.enemy_elixir = corrected_elixir
                    
                    elixir_after = corrected_elixir
                else:
                    # No regeneration active, just subtract cost normally
                    self.elixir_counter.register_spend(cost)
                    elixir_after = self.elixir_counter.get_enemy_elixir()
                
                # Update target for smooth display system
                self.target_enemy_elixir = elixir_after
                
                # For card plays, snap displayed value immediately (no smoothing delay)
                self.displayed_enemy_elixir = elixir_after
                
                # Update tracking state
                self.discovered_slots.add(slot)
                self.last_bubble_time[slot] = current_time
                
                # Status message with discovery progress
                identified_count = len(self.identified_slots)
                discovered_count = len(self.discovered_slots)
                self.status_update.emit(
                    f"💜 SECONDARY (Bubble): {card_name} replay (-{cost} elixir, compensated {self.BUBBLE_DETECTION_DELAY*1000:.0f}ms delay) "
                    f"| Slot {slot} | Elixir: {current_elixir:.1f} → {elixir_after:.1f} | Identified: {identified_count}/8"
                )
                
        except Exception as exc:
            self.status_update.emit(f"Bubble detection error: {exc}")
            import traceback
            traceback.print_exc()

    def _extract_bubble_digit(self, ocr_result):
        if not ocr_result:
            return None
        digits = []
        for line in ocr_result:
            if not line:
                continue
            if isinstance(line, list):
                for item in line:
                    if not item:
                        continue
                    text = str(item[1][0]) if isinstance(item, (list, tuple)) and len(item) > 1 else str(item)
                    match = re.search(r'\d', text)
                    if match:
                        digits.append(int(match.group(0)))
            else:
                text = str(line)
                match = re.search(r'\d', text)
                if match:
                    digits.append(int(match.group(0)))
        return digits[0] if digits else None

    def _merge_card_cycle(self, previous_cycle, detected, log_deferrals=False):
        UNKNOWN = {"unknown", "unknown_png"}
        if not previous_cycle:
            previous_cycle = ["unknown_png"] * 8
        final = list(previous_cycle[:8])
        if len(final) < 8:
            final += ["unknown_png"] * (8 - len(final))
        deferrals = 0

        for idx in range(8):
            if idx < len(detected):
                candidate, score = detected[idx]
            else:
                candidate, score = ("unknown_png", 0.0)
            candidate = candidate or "unknown_png"
            if candidate.lower() == "unknown":
                candidate = "unknown_png"

            prev = final[idx]
            prior_known = all(name not in UNKNOWN for name in final[:idx])

            if candidate not in UNKNOWN and not prior_known:
                deferrals += 1
                if log_deferrals:
                    self.status_update.emit(
                        f"Slot {idx + 1}: deferring {candidate} until earlier slots are known"
                    )
                continue

            if prev not in UNKNOWN and (candidate in UNKNOWN or score < 0.45):
                continue

            final[idx] = candidate

        return final, deferrals

    def integrate_detected_cards(self, detected, log_source=None, log_deferrals=False):
        previous_cycle = list(self.opponent_card_cycle)
        merged_cycle, deferrals = self._merge_card_cycle(previous_cycle, detected, log_deferrals=log_deferrals)
        if deferrals > 0:
            self.slot_deferral_streak += 1
        else:
            self.slot_deferral_streak = 0
        if self.slot_deferral_streak >= 2 and getattr(self, 'dynamic_slot_vertical_offset', 0) == 0:
            self.dynamic_slot_vertical_offset = -15
            self.card_slots = []
            self.slot_deferral_streak = 0
            self.status_update.emit("Slot locator deferring repeatedly - shifting slots up by 15px")
        if merged_cycle == previous_cycle:
            return False

        UNKNOWN = {"unknown", "unknown_png"}
        newly_identified = []
        for prev, new_name in zip(previous_cycle, merged_cycle):
            if new_name in UNKNOWN:
                continue
            if prev in UNKNOWN or prev != new_name:
                newly_identified.append(new_name)

        # Map newly identified cards to their slot indices for motion timestamp lookup
        for slot_idx, (prev, new_name) in enumerate(zip(previous_cycle, merged_cycle), start=1):
            if new_name in UNKNOWN:
                continue
            if prev in UNKNOWN or prev != new_name:
                # This slot was just identified - pass slot index for motion backdating
                self._apply_initial_card_cost(self.card_aliases.get(new_name, new_name), log_source or "auto", slot_idx=slot_idx)

        self.opponent_card_cycle = merged_cycle
        
        # Update identified_slots for bubble detection (secondary detection)
        # Mark slots as identified when they have a known card name
        for slot_idx, card_name in enumerate(merged_cycle, start=1):
            if card_name not in UNKNOWN:
                if slot_idx not in self.identified_slots:
                    self.identified_slots.add(slot_idx)
                    # Clear motion timestamp now that card is identified (for next card in this slot)
                    if slot_idx in self.slot_motion_detected_time:
                        del self.slot_motion_detected_time[slot_idx]
                    identified_count = len(self.identified_slots)
                    self.status_update.emit(
                        f"[CARD] Primary: Slot {slot_idx} identified as '{card_name}' | "
                        f"Identified: {identified_count}/8 (bubble detection now enabled for this slot)"
                    )
        
        self.opponent_cards_in_hand = merged_cycle[:4]
        self.opponent_upcoming_cards = merged_cycle[4:8]
        self.card_cycle_update.emit(self.opponent_cards_in_hand, self.opponent_upcoming_cards)
        if log_source:
            self.status_update.emit(f"{log_source}: updated card cycle -> {merged_cycle}")
        return True

    def _process_card_cycle(self, frame):
        # Debug logging throttled to avoid spam (only log occasionally)
        if not hasattr(self, '_process_debug_counter'):
            self._process_debug_counter = 0
        should_log = (self._process_debug_counter % 25 == 0)  # Log every 25th call (~5 seconds)
        self._process_debug_counter += 1
        
        if should_log:
            timer_dbg = getattr(self, '_last_timer_for_checkpoint', None)
            if not timer_dbg:
                timer_dbg = getattr(self, '_last_timer_display', None) or "?"
            debug_msg = (
                f"[Card Debug] slots={len(self.card_slots)} vs_seen={self.vs_badge_seen} "
                f"timer={timer_dbg} init_done={self._initial_identification_done} "
                f"regen_started={self.elixir_regeneration_started}"
            )
            try:
                self.status_update.emit(debug_msg)
            except Exception:
                pass
        if frame is None or frame.size == 0:
            if should_log:
                LOGGER.info("[Card Debug] Skipped: frame is None or empty")
            return
        if not (self.vs_badge_seen or self.elixir_regeneration_started):
            if should_log:
                LOGGER.info("[Card Debug] Skipped: VS badge not seen and elixir not started")
            return

        # Debug frame saving disabled
        # if (
        #     not self._card_debug_frame_saved
        #     and self._card_debug_capture_time is not None
        #     and time.time() >= self._card_debug_capture_time
        #     and getattr(frame, "size", 0) > 0
        # ):
        #     try:
        #         debug_path = Path("debug_live_frame.png")
        #         cv2.imwrite(str(debug_path), frame)
        #         self.status_update.emit(f"[Card Debug] Saved frame snapshot to {debug_path}")
        #     except Exception as exc:
        #         LOGGER.warning("Failed to save card debug frame: %s", exc)
        #     self._card_debug_frame_saved = True
        #     self._card_debug_capture_time = None

        # Find card slots ONCE at the beginning (anchors appear at ~2:55 timer)
        # Once found, slots never change position, so we cache them permanently
        if not self.card_slots:
            # OPTIMIZATION: Only try to find slots when timer <= 2:55 (anchors become visible)
            # This avoids expensive PaddleOCR calls (900-1300ms) when slots aren't visible yet
            timer_str = getattr(self, '_last_timer_for_checkpoint', None)
            if timer_str:
                try:
                    minutes, seconds = map(int, timer_str.split(':'))
                    total_seconds = minutes * 60 + seconds
                    # Only try at 2:55 or below (175 seconds or less)
                    if total_seconds > 175:
                        if should_log:
                            LOGGER.info(f"  ↳ Timer at {timer_str} - waiting for 2:55 to find card slots")
                        return
                except ValueError:
                    pass  # Invalid timer format, continue anyway
            
            # Throttle slot finding attempts to avoid performance hit (900-1300ms each)
            current_time = time.time()
            if not hasattr(self, '_last_slot_find_attempt'):
                self._last_slot_find_attempt = 0
            
            # Only try once every 2 seconds until we succeed
            if current_time - self._last_slot_find_attempt >= 2.0:
                if should_log:
                    LOGGER.info(f"  ↳ Timer at {timer_str or 'unknown'} - attempting to find card slots...")
                # If an executor is available, run slot finding in background so
                # the main loop and UI are not blocked. Use a fresh full-window
                # capture (same as snapshot) to increase OCR reliability.
                if getattr(self, '_executor', None):
                    # If a previous job is running, don't submit another
                    if getattr(self, '_slot_find_future', None) and not self._slot_find_future.done():
                        if should_log:
                            LOGGER.info("  ↳ Slot finding already in progress, skipping submit")
                    else:
                        try:
                            # Prefer a fresh full-window capture for OCR
                            try:
                                fresh = None
                                if getattr(self, 'window_capture', None) and getattr(self.window_capture, 'hwnd', None):
                                    fresh = self.window_capture.capture_window(full_window=True)
                            except Exception:
                                fresh = None
                            submit_frame = fresh if (fresh is not None and getattr(fresh, 'size', 0) > 0) else frame
                            self._slot_find_future = self._executor.submit(self.find_card_slots, submit_frame.copy())
                            # Attach a callback to report when finished
                            def _slot_done(fut):
                                try:
                                    slots_res = fut.result()
                                    if slots_res:
                                        LOGGER.info(f"Card slots found and cached (background): {len(slots_res)} slots detected.")
                                        try:
                                            self.status_update.emit(f"Card slots found ({len(slots_res)} slots) - will use cached positions")
                                        except Exception:
                                            pass
                                    else:
                                        LOGGER.warning("Background slot finding completed but no anchors found.")
                                except Exception as e:
                                    LOGGER.warning("Background slot finding failed: %s", e)

                            try:
                                self._slot_find_future.add_done_callback(_slot_done)
                            except Exception:
                                # Older Python versions or certain Future implementations may not support callback
                                pass
                        except Exception as exc:
                            LOGGER.warning("Failed to submit slot finding job: %s", exc)
                else:
                    # No executor available - run synchronously (worst-case blocking)
                    try:
                        fresh = None
                        if getattr(self, 'window_capture', None) and getattr(self.window_capture, 'hwnd', None):
                            fresh = self.window_capture.capture_window(full_window=True)
                    except Exception:
                        fresh = None
                    sync_frame = fresh if (fresh is not None and getattr(fresh, 'size', 0) > 0) else frame
                    self.find_card_slots(sync_frame)

                self._last_slot_find_attempt = current_time

                # If slots were discovered synchronously above, log immediately with positions.
                if self.card_slots:
                    LOGGER.info(f"Card slots found and cached! {len(self.card_slots)} slots detected.")
                    # Log each slot's position for debugging
                    for idx, (x, y, w, h) in enumerate(self.card_slots[:8], start=1):
                        LOGGER.info(f"  Slot {idx}: x={x}, y={y}, w={w}, h={h}")
                    self.status_update.emit(f"✅ Card slots found ({len(self.card_slots)} slots) - positions logged")
            
        if not self.card_slots:
            if should_log:
                LOGGER.info("  ↳ Skipped: Could not find card slots")
            return

        # If we have slots but haven't run the one-time initial identification,
        # schedule a background job to run ResNet across all unknown slots and
        # populate initial identities. Motion-gated per-slot ResNet checks will
        # be deferred until this initial pass completes.
        if self.card_slots and not self._initial_identification_done:
            if not self._initial_identification_started:
                self._initial_identification_started = True
                LOGGER.info("Scheduling one-time initial identification sweep (background)")
                try:
                    if getattr(self, '_executor', None):
                        self._initial_identification_future = self._executor.submit(self._run_initial_identification, frame.copy())
                        # Attach callback to mark done
                        def _initial_done(fut):
                            try:
                                res = fut.result()
                                LOGGER.info("Initial identification sweep completed")
                                try:
                                    self.status_update.emit("Initial identification completed")
                                except Exception:
                                    pass
                            except Exception as e:
                                LOGGER.warning("Initial identification failed: %s", e)
                            finally:
                                self._initial_identification_done = True

                        try:
                            self._initial_identification_future.add_done_callback(_initial_done)
                        except Exception:
                            # If futures don't support add_done_callback, we still mark it started
                            pass
                    else:
                        # No executor: run synchronously but mark done immediately
                        self._run_initial_identification(frame)
                        self._initial_identification_done = True
                except Exception as exc:
                    LOGGER.warning("Failed to start initial identification: %s", exc)

            # While the initial identification is pending, skip motion-gated checks
            if not self._initial_identification_done:
                if should_log:
                    LOGGER.info("  ↳ Waiting for initial identification to complete - deferring motion-gated checks")
                return

        previous_cycle = list(self.opponent_card_cycle)
        UNKNOWN = {"unknown", "unknown_png"}
        capture_frame = frame
        
        # FRAME DIMENSION TRACKING: Log original frame vs capture frame
        original_frame_shape = frame.shape if frame is not None and hasattr(frame, 'shape') else None
        
        if self.window_capture.hwnd:
            try:
                captured = self.window_capture.capture_window(full_window=True)
                if captured is not None and captured.size > 0:
                    capture_frame = captured
            except Exception as e:
                capture_frame = frame
                LOGGER.warning(f"[WARN] Failed to capture window: {e}")
        
        # ENHANCED DIAGNOSTICS: Always log frame dimensions when motion_debug enabled
        motion_debug = getattr(self, 'motion_debug', False)
        capture_frame_shape = capture_frame.shape if capture_frame is not None and hasattr(capture_frame, 'shape') else None
        
        if motion_debug:
            shape_info = f"original={original_frame_shape}, capture={capture_frame_shape}"
            if self.card_slots:
                boxes = []
                for i, (x, y, w, h) in enumerate(self.card_slots[:8], start=1):
                    boxes.append(f"{i}:(x={int(x)},y={int(y)},w={int(w)},h={int(h)})")
                slot_info = ', '.join(boxes)
            else:
                slot_info = 'slots=EMPTY'
            # Log to file only, not UI (too verbose)
            LOGGER.info(f"[MOTION DEBUG] Frame shapes: {shape_info} | Slots: [{slot_info}]")
        
        # CRITICAL CHECK: Validate capture frame exists
        if capture_frame is None or capture_frame.size == 0:
            if should_log:
                LOGGER.warning(f"[WARN] capture_frame is None/empty - skipping card cycle processing")
            return
        
        # ONE-TIME DEBUG: Save frame with slot overlays to verify coordinates
        if not hasattr(self, '_debug_slots_frame_saved'):
            self._debug_slots_frame_saved = False
        if not self._debug_slots_frame_saved and self.card_slots:
            try:
                debug_overlay = capture_frame.copy()
                for idx, (x, y, w, h) in enumerate(self.card_slots[:8], start=1):
                    # Draw slot rectangle
                    cv2.rectangle(debug_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Draw slot number
                    cv2.putText(
                        debug_overlay,
                        f"Slot {idx}",
                        (x + 2, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    # Draw coordinates
                    cv2.putText(
                        debug_overlay,
                        f"({x},{y})",
                        (x + 2, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                # Debug slots overlay saving disabled
                # debug_path = Path("debug_slots_live.png")
                # cv2.imwrite(str(debug_path), debug_overlay)
                # self.status_update.emit(f"✅ Saved slot overlay to {debug_path} (frame: {capture_frame.shape[1]}x{capture_frame.shape[0]})")
                # LOGGER.info(f"✅ [DEBUG] Saved slot overlay: {debug_path} | Frame: {capture_frame.shape[1]}x{capture_frame.shape[0]} | Slots: {len(self.card_slots)}")
                self._debug_slots_frame_saved = True
            except Exception as exc:
                LOGGER.warning(f"Failed to save debug slot overlay: {exc}")
        
        detected = []
        # Use 1-based indexing for display (slots 1-8 instead of 0-7)
        current_time = time.time()
        do_resnet_poll = (self.resnet_poll_interval and (current_time - getattr(self, '_last_resnet_poll', 0.0) >= self.resnet_poll_interval))
        if do_resnet_poll:
            self._last_resnet_poll = current_time
            if should_log:
                LOGGER.info(f"[TIME] ResNet poll triggered (interval={self.resnet_poll_interval}s)")

        # DEBUG: Log identified_slots state once per cycle
        if should_log:
            LOGGER.info(f"[MOTION] identified_slots = {sorted(self.identified_slots)}")
        
        for display_idx, slot in enumerate(self.card_slots[:8], start=1):
            array_idx = display_idx - 1  # Convert to 0-based for array access
            prev = previous_cycle[array_idx] if array_idx < len(previous_cycle) else "unknown_png"
            if should_log:
                LOGGER.info(f"[RESET] Slot {display_idx}: Previous={prev}, in UNKNOWN={prev in UNKNOWN}")
            if prev not in UNKNOWN:
                detected.append((prev, 1.0))
                if should_log:
                    LOGGER.info(f"  ↳ Using cached: {prev}")
                continue
            
            # Skip running ResNet on slots that have already been identified via motion detection
            # Once a card is identified (not unknown), we never need to check it again
            if display_idx in self.identified_slots:
                # BUG FIX: If card is unknown but slot is marked as identified, remove it from identified_slots
                if prev in UNKNOWN:
                    self.identified_slots.discard(display_idx)
                    if should_log:
                        LOGGER.warning(f"[FIX] Slot {display_idx} was in identified_slots but card is '{prev}' - removed from identified_slots")
                else:
                    if should_log:
                        LOGGER.info(f"  ↳ Slot {display_idx} already identified; skipping motion/ResNet")
                    detected.append((previous_cycle[array_idx], 1.0))
                    continue

            # If motion gate disabled, or a resnet poll is scheduled, we run detection.
            run_detection = self.disable_motion_gate or do_resnet_poll

            # Respect animation settle time: only run ResNet if enough time has
            # passed since motion was observed (resnet_delay_after_motion).
            last_motion = self.slot_motion_detected_time.get(display_idx)
            if last_motion is not None:
                time_since_motion = current_time - last_motion
            else:
                time_since_motion = float('inf')

            if not run_detection:
                # Default behavior: use motion gate
                # CROP VALIDATION: Extract crop and validate before passing to motion detector
                x, y, w, h = slot[0], slot[1], slot[2], slot[3]
                
                # Bounds checking
                frame_h, frame_w = capture_frame.shape[:2]
                if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                    LOGGER.error(f"[ERROR] Slot {display_idx} CROP OUT OF BOUNDS! slot=({x},{y},{w},{h}) frame=({frame_w},{frame_h})")
                    try:
                        self.status_update.emit(f"[ERROR] Slot {display_idx} crop out of bounds - skipping")
                    except Exception:
                        pass
                    detected.append(("unknown_png", 0.0))
                    continue
                
                # Extract crop
                slot_crop = capture_frame[y:y+h, x:x+w]
                
                # Validate crop
                if slot_crop is None or slot_crop.size == 0:
                    LOGGER.error(f"[ERROR] Slot {display_idx} EMPTY CROP! slot=({x},{y},{w},{h}) crop_size={slot_crop.size if slot_crop is not None else 'None'}")
                    try:
                        self.status_update.emit(f"[ERROR] Slot {display_idx} empty crop - skipping")
                    except Exception:
                        pass
                    detected.append(("unknown_png", 0.0))
                    continue
                
                # Log crop info in debug mode
                if motion_debug and should_log:
                    LOGGER.info(f"[MOTION] [MOTION DEBUG] Slot {display_idx} crop extracted: shape={slot_crop.shape}, from ({x},{y}) size ({w}x{h})")
                
                # Now call motion detector with validated crop
                has_motion = self._has_slot_changed(display_idx, slot_crop)
                if not has_motion:
                    if should_log:
                        LOGGER.info(f"  ↳ Slot {display_idx}: no motion, skipping ResNet")
                    detected.append(("unknown_png", 0.0))
                    continue
                # If motion detected, schedule detection after animation delay
                if time_since_motion < self.resnet_delay_after_motion:
                    LOGGER.info(f"⏳ Slot {display_idx}: Motion detected but animation delay not passed yet ({time_since_motion:.2f}s / {self.resnet_delay_after_motion}s)")
                    detected.append(("unknown_png", 0.0))
                    continue
                else:
                    LOGGER.info(f"[OK] Slot {display_idx}: Motion detected AND animation delay passed ({time_since_motion:.2f}s) → Running ResNet now!")

            # Otherwise run detection now
            if should_log:
                LOGGER.info(f"  ↳ Running detection for slot {display_idx} (run_detection={run_detection})")
            
            # Log ResNet attempt details when triggered by motion
            if last_motion is not None:
                time_since_motion = time.time() - last_motion
                LOGGER.info(f"🔍 ResNet triggered by motion: Slot {display_idx} - delay={time_since_motion:.3f}s since motion, running detection...")
            
            # Skip motion gate since we already checked motion above
            # Pass motion_detected=True so detect_card_in_slot knows to retry on unknown
            name, score = self.detect_card_in_slot(capture_frame, slot, display_idx, skip_motion_gate=True, motion_detected=True)
            if not name:
                name = "unknown_png"
                score = 0.0

            # Record detection into per-slot history for trailing-weighted consensus
            now_ts = time.time()
            hist = self.slot_detection_history.setdefault(display_idx, [])
            hist.append((now_ts, name, float(score)))
            # Trim history
            if len(hist) > self.SLOT_HISTORY_MAX:
                hist.pop(0)

            # Compute exponential trailing weights and weighted confidences
            weight_acc = {}
            score_acc = {}
            count_acc = {}
            for ts, nm, sc in hist:
                w = float(np.exp(-(now_ts - ts) / self.SLOT_HISTORY_TAU))
                weight_acc[nm] = weight_acc.get(nm, 0.0) + w
                score_acc[nm] = score_acc.get(nm, 0.0) + w * sc
                count_acc[nm] = count_acc.get(nm, 0) + 1

            best_name = None
            best_weighted_conf = 0.0
            best_count = 0
            for nm in score_acc:
                weighted_conf = score_acc[nm] / weight_acc[nm] if weight_acc[nm] > 0 else 0.0
                if weighted_conf > best_weighted_conf:
                    best_weighted_conf = weighted_conf
                    best_name = nm
                    best_count = count_acc.get(nm, 0)

            # Check if this card is problematic and requires consensus
            is_problematic = best_name and best_name.lower().replace(" ", "").replace("_", "") in self.PROBLEMATIC_CARDS_REQUIRE_CONSENSUS
            
            # Accept consensus only when we have enough samples and high weighted confidence
            # OR if it's not a problematic card and we have a strong single detection
            has_strong_consensus = (best_name and best_name != "unknown_png" and 
                                   best_count >= self.SLOT_CONSENSUS_MIN_SAMPLES and 
                                   best_weighted_conf >= self.SLOT_CONSENSUS_WEIGHTED_THRESHOLD)
            
            if has_strong_consensus:
                consensus_name = best_name
                consensus_conf = best_weighted_conf
                if display_idx not in self.identified_slots:
                    self.identified_slots.add(display_idx)
                    # DON'T clear motion timestamp here - it's needed for elixir backdating
                    # Will be cleared in integrate_detected_cards AFTER elixir cost is applied
                    consensus_type = "[WARN] PROBLEMATIC-VERIFIED" if is_problematic else "consensus"
                    LOGGER.info(f"[OK] Slot {display_idx} {consensus_type}: {consensus_name} (conf={consensus_conf:.2f}, samples={best_count}) - marked identified")
                detected.append((consensus_name, consensus_conf))
            elif is_problematic and best_count < self.SLOT_CONSENSUS_MIN_SAMPLES:
                # Problematic card detected but not enough samples yet - return unknown to keep collecting
                LOGGER.info(f"[WARN] Slot {display_idx}: {best_name} is problematic, waiting for consensus ({best_count}/{self.SLOT_CONSENSUS_MIN_SAMPLES} samples)")
                detected.append(("unknown_png", 0.0))  # Force another detection cycle
            else:
                # No strong consensus yet — return latest detection as tentative
                detected.append((name, score))
            # Debug: Log detection results (temporary - remove after fixing)
            if name != "unknown_png" or should_log:
                LOGGER.info(f"  ↳ Detected: {name} (score={score:.2f})")

        changed = self.integrate_detected_cards(detected, log_source="Auto detection", log_deferrals=True)
        # Removed spam: "Auto detection: cycle unchanged"

    def prepare_for_new_match(self):
        """Expose elixir reset for UI controls."""
        self._prepare_for_new_match()

    def detect_match_timer(self, frame, aggressive=False):
        """Detect and OCR the in-game match timer using a fixed ROI.
        
        Args:
            frame: The frame to detect timer in
            aggressive: If True, run OCR on every frame (for 2:53→2:52 transition)
        """
        if frame is None or frame.size == 0:
            return None
        if self.ocr is None:
            return None
        if not isinstance(frame, np.ndarray):
            return None
        if frame.ndim != 3:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            except Exception:
                return None

        debug_mode = getattr(self, 'timer_debug_request', False)
        current_time = time.time()

        if not self.timer_ocr_started:
            return None
        
        # Aggressive OCR during 2:53 → 2:52 transition or until 9-elixir checkpoint
        if aggressive or not self.nine_elixir_checkpoint_done:
            # Run OCR every frame (no throttling)
            pass
        else:
            # After checkpoint, only OCR every 10 seconds
            if self.timer_initialized and (current_time - self.last_timer_check) < self.timer_check_interval:
                return None

        self.last_timer_check = current_time
        ocr_start = time.time()

        frame_h, frame_w = frame.shape[:2]
        if frame_h == 0 or frame_w == 0:
            return None

        x1 = int(round(frame_w * 0.75))
        y1 = 0
        x2 = frame_w
        y2 = min(frame_h, int(round(frame_h * 0.25)))
        if x2 <= x1 or y2 <= y1:
            self.last_timer_ocr_debug = {
                'reason': 'roi_invalid',
                'roi': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1},
                'frame_shape': frame.shape,
            }
            return None

        timer_crop = frame[y1:y2, x1:x2]
        if timer_crop.size == 0:
            self.last_timer_ocr_debug = {
                'reason': 'roi_empty',
                'roi': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1},
            }
            return None

        self.timer_roi_bounds = (x1, y1, x2 - x1, y2 - y1)
        self.timer_merged_bbox = (x1, y1, x2, y2)

        if debug_mode:
            try:
                cv2.imwrite('debug_timer_roi.png', timer_crop)
            except Exception:
                pass

        gray = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug_mode:
            try:
                cv2.imwrite('debug_timer_roi_processed.png', thresh)
            except Exception:
                pass

        detected_chunks = []
        best_confidence = 0.0
        attempt_log = []
        detected_source = None

        def consume(texts, scores):
            nonlocal detected_chunks, best_confidence
            if not texts:
                return
            for idx, text in enumerate(texts):
                text_str = str(text).strip()
                if not text_str:
                    continue
                detected_chunks.append(text_str)
                if scores and idx < len(scores):
                    try:
                        conf = float(scores[idx])
                    except Exception:
                        conf = None
                    if conf is not None:
                        best_confidence = max(best_confidence, conf)

        def extract(result):
            texts, scores = [], []
            if not result:
                return texts, scores
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    texts = [str(t) for t in first.get('rec_texts', [])]
                    scores = first.get('rec_scores', [])
                else:
                    for group in result:
                        if not isinstance(group, (list, tuple)):
                            continue
                        for entry in group:
                            if isinstance(entry, (list, tuple)) and len(entry) > 1:
                                payload = entry[1]
                                if isinstance(payload, (list, tuple)) and payload:
                                    texts.append(str(payload[0]))
                                    if len(payload) > 1:
                                        scores.append(payload[1])
                                else:
                                    texts.append(str(payload))
            return texts, scores

        attempts = [
            ('predict_thresh', 'predict', thresh),
            ('ocr_thresh', 'ocr', thresh),
            ('predict_raw', 'predict', timer_crop),
            ('ocr_raw', 'ocr', timer_crop),
        ]

        for label, method, image in attempts:
            if detected_chunks:
                break
            try:
                fn = getattr(self.ocr, method)
                result = fn(image)
                attempt_log.append({
                    'attempt': label,
                    'result_type': type(result).__name__,
                    'result_len': len(result) if isinstance(result, (list, tuple)) else None,
                })
                texts, scores = extract(result)
                if texts:
                    consume(texts, scores)
                    detected_source = label
            except Exception as exc:
                attempt_log.append({'attempt': label, 'error': str(exc)})

        detected_text = ' '.join(detected_chunks).strip()
        debug_info = {
            'roi': {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1},
            'detected_text': detected_text,
            'best_confidence': best_confidence,
            'attempts': attempt_log,
            'source': detected_source,
        }

        timer_str = None
        minutes = seconds = 0
        if detected_text:
            for pattern in [r'(\d{1,2}):(\d{2})', r'Time left:\s*(\d{1,2}):(\d{2})', r'(\d{1,2}):(\d{2})\s*left']:
                match_obj = re.search(pattern, detected_text, re.IGNORECASE)
                if match_obj:
                    minutes = int(match_obj.group(1))
                    seconds = int(match_obj.group(2))
                    timer_str = f"{minutes:02d}:{seconds:02d}"
                    break
        debug_info['timer_str'] = timer_str
        self.last_timer_ocr_debug = debug_info

        if timer_str is None:
            ocr_dur = (time.time() - ocr_start) * 1000
            if ocr_dur > 300:
                LOGGER.warning(f"Timer OCR took {ocr_dur:.0f}ms and returned no result")
            return None

        # Store last timer for aggressive OCR decision
        self._last_timer_for_checkpoint = timer_str
        
        # 9-elixir checkpoint at 2:52 for proper alignment
        if not self.nine_elixir_checkpoint_done and self.elixir_regeneration_started:
            if minutes == 2 and seconds == 52:
                self._check_nine_elixir_alignment(minutes, seconds)

        if not self.timer_initialized:
            self.timer_initialized = True
            total_seconds = minutes * 60 + seconds
            self.match_start_time = current_time - (180 - total_seconds)
            self.timer_countdown_active = True
            self.status_update.emit(f"Match timer initialized at {timer_str}")
            # Request the main UI to run a full-window snapshot to prime slot detection
            try:
                self.request_snapshot.emit()
            except Exception:
                pass
            self._card_debug_capture_time = current_time + 1.0
        self._last_timer_display = timer_str
        self.timer_update.emit(timer_str)
        return timer_str

    def _update_timer_countdown(self):
        if not self.timer_initialized or self.match_start_time is None:
            return
        current_time = time.time()
        elapsed = current_time - self.match_start_time
        remaining = max(0, int(round(180 - elapsed)))
        minutes = remaining // 60
        seconds = remaining % 60
        formatted = f"{minutes:02d}:{seconds:02d}"
        if formatted != getattr(self, '_last_timer_display', None):
            self._last_timer_display = formatted
            self.timer_update.emit(formatted)
            # self.status_update.emit(f"[TIME] Timer display updated to {formatted}")
    def preprocess_timer_region(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _init_card_detection(self) -> None:
        """Load card templates and initialize ResNet if available."""
        self.card_templates.clear()
        self.card_templates_gray.clear()
        self.card_aliases = {}
        self.unknown_template = None

        def store_template(key: str, image: np.ndarray) -> None:
            self.card_templates[key] = image
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = image if image.ndim == 2 else cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
            self.card_templates_gray[key] = gray

        base_dir = CARD_IMAGES_DIR / "card_ed"
        if base_dir.exists():
            for path in sorted(base_dir.glob("*.png")):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                name = path.stem
                store_template(name, image)
                self.card_aliases[name] = name
                if self.unknown_template is None and name.lower() in {"unknown", "unknown_png"}:
                    self.unknown_template = image

        evo_dir = CARD_IMAGES_DIR / "card_ed_evo"
        if evo_dir.exists():
            for path in sorted(evo_dir.glob("*.png")):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                base_name = path.stem
                evo_name = f"{base_name}_evo"
                store_template(evo_name, image)
                self.card_aliases[evo_name] = base_name
                if base_name not in self.card_aliases:
                    self.card_aliases[base_name] = base_name

        # Map alternate artwork cards to their canonical names
        # This handles cards that got updated artwork but we want to treat as the same card
        alternate_mappings = {
            'MuskAlternate': 'Musk',
            '3MAlternate': '3M',
            'MuskAlternate_evo': 'Musk_evo',
        }
        for alternate_name, canonical_name in alternate_mappings.items():
            if alternate_name in self.card_templates:
                self.card_aliases[alternate_name] = canonical_name
                LOGGER.info(f"Mapped alternate artwork: {alternate_name} → {canonical_name}")

        if NEURAL_AVAILABLE and self.card_templates:
            try:
                self.resnet_embedder = ResNetEmbedder()
                payload = {name: {'image': image} for name, image in self.card_templates.items()}
                self.resnet_embedder.precompute_card_embeddings(payload)
                
                # 🔥 Calibrate delay compensation based on measured speed
                self._calibrate_resnet_delay()
                
            except Exception as exc:
                LOGGER.warning("ResNet embedder init failed: %s", exc)
                self.resnet_embedder = None
        else:
            self.resnet_embedder = None

    def _run_initial_identification(self, frame):
        """One-time pass: run detection across all slots to populate initial identities.

        This is intended to run once after `card_slots` have been found. It will
        perform ResNet/template matching for each slot and integrate the results
        using the existing `integrate_detected_cards` method.
        
        IMPORTANT: This bypasses motion gating (skip_motion_gate=True) to do a direct
        ResNet sweep when slots are first discovered.
        """
        try:
            if not self.card_slots:
                return None
            detected = []
            for i, slot in enumerate(self.card_slots[:8]):
                display_idx = i + 1
                # Pass skip_motion_gate=True to bypass motion checks during initial sweep
                name, score = self.detect_card_in_slot(frame, slot, display_idx, skip_motion_gate=True)
                if not name:
                    name = "unknown_png"
                detected.append((name, score))

                # Prime motion baseline: extract center patch (3x3 or 5x5)
                try:
                    x, y, w, h = slot
                    crop = frame[y:y+h, x:x+w]
                    ch, cw = crop.shape[:2]
                    patch_size = 3
                    if getattr(self, 'motion_sensitivity', 1.0) > 1.25:
                        patch_size = 5
                    half = patch_size // 2
                    if ch > half*2 and cw > half*2:
                        cy, cx = ch // 2, cw // 2
                        region = crop[cy-half:cy+half+1, cx-half:cx+half+1].copy()
                        self.slot_previous_frames[display_idx] = region.copy()  # CRITICAL: Extra safety copy
                        LOGGER.info(f"[OK] [v2-COPY-FIX] Primed motion baseline for slot {display_idx} (patch={patch_size}x{patch_size})")
                        # UI message removed - too verbose
                except Exception:
                    pass

            self.integrate_detected_cards(detected, log_source="Initial identification", log_deferrals=True)
            return detected
        except Exception as exc:
            LOGGER.warning("_run_initial_identification failed: %s", exc)
            return None

    def _dump_slot_debug(self, slot_idx: int, frame: np.ndarray, slot, reason: str, name: str = None, score: float = None):
        """Save a small debug image and metadata for a slot when detection fails or is skipped.

        Rate-limited per-slot to avoid disk flooding.
        """
        try:
            now = time.time()
            last = getattr(self, '_last_debug_dump_time', {})
            if last.get(slot_idx, 0.0) + 3.0 > now:
                return
            # Ensure debug folder
            debug_dir = Path('runs') / 'debug_slot_dumps'
            debug_dir.mkdir(parents=True, exist_ok=True)
            x, y, w, h = slot
            crop = frame[y:y+h, x:x+w].copy() if frame is not None and frame.size else None
            ts = int(now)
            img_name = debug_dir / f"slot_{slot_idx}_reason_{reason}_{ts}.png"
            meta_name = debug_dir / f"slot_{slot_idx}_reason_{reason}_{ts}.json"
            if crop is not None and crop.size:
                try:
                    cv2.imwrite(str(img_name), crop)
                except Exception:
                    pass
            meta = {
                'slot': slot_idx,
                'reason': reason,
                'name': name,
                'score': float(score) if score is not None else None,
                'disable_motion_gate': bool(getattr(self, 'disable_motion_gate', False)),
                'resnet_poll_interval': float(getattr(self, 'resnet_poll_interval', 0.0)),
                'initial_identification_done': bool(getattr(self, '_initial_identification_done', False)),
                'identified_slots': list(getattr(self, 'identified_slots', [])),
            }
            try:
                import json
                with open(meta_name, 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass
            last[slot_idx] = now
            self._last_debug_dump_time = last
            LOGGER.info(f"Wrote debug dump for slot {slot_idx} reason={reason} -> {img_name}")
        except Exception as exc:
            LOGGER.warning("_dump_slot_debug failed: %s", exc)
    
    def _calibrate_resnet_delay(self):
        """
        Calibrate RESNET_DETECTION_DELAY based on actual measured inference speed.
        This ensures accurate elixir compensation regardless of hardware.
        """
        if not self.resnet_embedder:
            return
        
        # Get average inference time from benchmark
        avg_inference_ms = self.resnet_embedder.get_average_inference_time()
        
        # Calculate total detection delay for elixir compensation:
        # This is the time from when the player actually plays the card to when we detect it.
        # 
        # Timeline:
        # t=0ms:    Player clicks "play card" → Card appears, animation starts
        # t=0-200ms: Motion check runs and detects change (happens continuously)
        # t=500ms:  Animation completes, card fully visible → ResNet runs
        # t=516ms:  ResNet completes → Card identified!
        #
        # For elixir compensation, we need the delay from t=0 (actual play) to t=516ms (detection):
        # - Animation settle: 500ms (waiting for card to be fully visible)
        # - ResNet inference: measured (actual detection time)
        # - Safety margin: 50ms (buffer for timing variations)
        #
        # Motion check is continuous (not sequential), so we don't add it to the total delay.
        self.RESNET_INFERENCE_TIME = avg_inference_ms / 1000.0  # Convert to seconds
        self.RESNET_DETECTION_DELAY = (
            self.resnet_delay_after_motion +   # 700ms wait for animation to complete (calibrated)
            self.RESNET_INFERENCE_TIME +       # Measured inference time
            0.05                                # 50ms safety margin
        )
        
        LOGGER.info(f"🎯 ResNet delay calibration:")
        LOGGER.info(f"   Animation settle: {self.resnet_delay_after_motion*1000:.0f}ms (card appearance)")
        LOGGER.info(f"   ResNet inference: {self.RESNET_INFERENCE_TIME*1000:.0f}ms (detection)")
        LOGGER.info(f"   Safety margin: 50ms (timing buffer)")
        LOGGER.info(f"   Total delay compensation: {self.RESNET_DETECTION_DELAY*1000:.0f}ms (play→detection)")
        
        self.status_update.emit(
            f"[OK] ResNet calibrated: {avg_inference_ms:.0f}ms inference, "
            f"{self.RESNET_DETECTION_DELAY*1000:.0f}ms total delay compensation"
        )

    def _load_card_costs(self) -> None:
        """Load elixir costs for cards from card_database.json."""
        try:
            with open(CARD_DATABASE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f).get("cards", {})
        except Exception as exc:
            self.card_elixir_costs = {}
            self.status_update.emit(f"Failed to load card elixir costs: {exc}")
            return
        costs = {}
        for name, info in data.items():
            cost_val = info.get("elixir_cost")
            cost = None
            if isinstance(cost_val, (int, float)):
                cost = float(cost_val)
            elif isinstance(cost_val, str):
                match = re.search(r"\d+(?:\.\d+)?", cost_val)
                if match:
                    cost = float(match.group(0))
            if cost is None:
                continue
            costs[name.lower()] = cost
        self.card_elixir_costs = costs

    def _get_card_cost(self, card_name: str) -> float | None:
        if not card_name:
            return None
        canonical = self.card_aliases.get(card_name, card_name) if hasattr(self, 'card_aliases') else card_name
        cost = self.card_elixir_costs.get(canonical.lower())
        if cost is None:
            cost = self.card_elixir_costs.get(card_name.lower())
        return cost

    def _apply_initial_card_cost(self, card_name: str, source: str = "auto", slot_idx: int = None) -> None:
        """
        Apply elixir cost for initial card identification via PRIMARY detection (ResNet).
        
        With motion-gated detection, we can backdate to the EXACT time motion was detected
        instead of using a fixed delay estimate. This significantly improves elixir accuracy.
        
        If slot_idx is provided and we have a motion timestamp:
        - Use actual delay = current_time - motion_detected_time
        Otherwise fall back to fixed delay estimate (~566ms):
        - 500ms for card animation/appearance
        - ~16ms for ResNet inference
        - 50ms safety margin
        """
        cost = self._get_card_cost(card_name)
        if cost is None:
            return
        key = card_name.lower()
        if key in self.card_cycle_costed:
            return
        self.card_cycle_costed.add(key)
        
        # Ensure elixir regeneration has started if VS badge has been seen
        if not self.elixir_regeneration_started and self.vs_badge_seen:
            self._start_elixir_regeneration()
        
        # Get current time
        current_time = time.time()
        
        # IMPROVED: Use actual motion timestamp if available (much more accurate!)
        if slot_idx and slot_idx in self.slot_motion_detected_time:
            motion_time = self.slot_motion_detected_time[slot_idx]
            actual_delay = current_time - motion_time
            LOGGER.info(f"[TIME] Motion-based backdating: Card played {actual_delay:.3f}s ago (slot {slot_idx})")
        else:
            # Fall back to fixed delay estimate
            actual_delay = self.RESNET_DETECTION_DELAY
            if slot_idx:
                LOGGER.warning(f"[TIME] Fixed-delay backdating: slot_idx={slot_idx} but motion timestamp NOT FOUND! Available slots: {list(self.slot_motion_detected_time.keys())}")
            else:
                LOGGER.info(f"[TIME] Fixed-delay backdating: slot_idx is None, using estimated {actual_delay:.3f}s delay")
        
        # Calculate play time (when card was actually played)
        actual_play_time = current_time - actual_delay
        
        # Update to current state first
        self.elixir_counter.update()
        current_elixir = self.elixir_counter.get_enemy_elixir()
        
        # Calculate elixir at time of actual card play
        # Regeneration that occurred during detection delay
        if self.elixir_counter.match_start_time:
            current_phase = self.elixir_counter.get_current_phase()
            regen_rate = ELIXIR_PHASES[current_phase]
            regen_during_detection = actual_delay * regen_rate
            
            # Elixir at play time = current - regeneration during detection
            elixir_at_play = current_elixir - regen_during_detection
            
            # Calculate what elixir should be now:
            # (elixir_at_play - cost) + regen_during_detection
            corrected_elixir = max(0.0, elixir_at_play - cost)
            
            # Update the elixir counter to the corrected value
            # This effectively: subtracts cost from past, adds regen back to present
            expected_total = self.elixir_counter._expected_total()
            self.elixir_counter.spent_elixir = max(0.0, expected_total - corrected_elixir)
            self.elixir_counter.enemy_elixir = corrected_elixir
            
            elixir_after = corrected_elixir
            spent_after = self.elixir_counter.spent_elixir
        else:
            # No regeneration active, just subtract cost normally
            self.elixir_counter.register_spend(cost)
            elixir_after = self.elixir_counter.get_enemy_elixir()
            spent_after = self.elixir_counter.spent_elixir
        
        # Immediately update target for smooth display system
        self.target_enemy_elixir = elixir_after
        
        # For card plays, snap displayed value immediately and emit to GUI
        self.displayed_enemy_elixir = elixir_after
        self.elixir_update.emit(round(elixir_after, 1))  # Immediate GUI update
        
        # Log the delay compensation with actual measured delay
        delay_type = "motion-based" if (slot_idx and slot_idx in self.slot_motion_detected_time) else "estimated"
        self.status_update.emit(
            f"[CARD] PRIMARY (ResNet): {card_name} (-{cost} elixir, {delay_type} {actual_delay*1000:.0f}ms backdate) "
            f"| Elixir: {current_elixir:.1f} → {elixir_after:.1f} | Spent: {spent_after:.1f}"
        )

    def find_card_slots(self, frame):
        if self.slot_locator is None:
            LOGGER.warning("Card slot locator unavailable; cannot locate slots")
            try:
                self.status_update.emit("[Card Debug] Slot locator unavailable")
            except Exception:
                pass
            return self.card_slots or []

        # Rate-limit heavy slot-locator calls to avoid repeated OCR when anchors are flaky
        now = time.time()
        if not hasattr(self, '_last_slot_locator_time'):
            self._last_slot_locator_time = 0.0
        if now - getattr(self, '_last_slot_locator_time', 0.0) < 3.0:
            LOGGER.debug("find_card_slots: called too soon after last attempt; skipping to avoid overload")
            return self.card_slots or []
        self._last_slot_locator_time = now

        start_ts = time.time()
        LOGGER.info(f"[MOTION] find_card_slots START: frame shape={getattr(frame,'shape',None)}, dtype={getattr(frame,'dtype',None)}")
        try:
            # Try to get debug visualization too so we can save a sample when anchors fail
            layout, debug_img = self.slot_locator.locate_slots(frame, return_debug=True)
        except Exception as exc:
            duration = (time.time() - start_ts) * 1000
            LOGGER.warning("Slot locator error after %.0fms: %s", duration, exc)
            return self.card_slots or []

        duration = (time.time() - start_ts) * 1000
        if not layout or not getattr(layout, 'slots', None):
            LOGGER.warning("Slot locator returned no slots after %.0fms", duration)
            # Save debug image for offline inspection (only occasionally)
            try:
                debug_path = f"slot_find_failed_{int(time.time())}.png"
                if debug_img is not None:
                    cv2.imwrite(debug_path, debug_img)
                    LOGGER.info("Saved slot-finder debug image: %s", debug_path)
            except Exception:
                pass
            try:
                self.status_update.emit("[Card Debug] Slot locator returned no slots")
            except Exception:
                pass
            return self.card_slots or []

        slots = self._slots_to_original(layout, frame.shape)
        if not slots:
            LOGGER.warning("Slot locator produced empty slot list after normalization (%.0fms)", duration)
            try:
                self.status_update.emit("[Card Debug] Slot layout empty after normalization")
            except Exception:
                pass
            return self.card_slots or []

        # Save debug overlays so we can visualize slot placement in both spaces
        try:
            debug_dir = Path("frontend")
            debug_dir.mkdir(parents=True, exist_ok=True)

            if debug_img is not None:
                norm_path = debug_dir / "slot_debug_normalized.png"
                cv2.imwrite(str(norm_path), debug_img)

            raw_overlay = frame.copy()
            for idx, (slot_x, slot_y, slot_w, slot_h) in enumerate(slots[:8], start=1):
                cv2.rectangle(raw_overlay, (slot_x, slot_y), (slot_x + slot_w, slot_y + slot_h), (255, 0, 0), 1)
                cv2.putText(
                    raw_overlay,
                    str(idx),
                    (slot_x + 4, max(15, slot_y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            raw_path = debug_dir / "slot_debug_raw.png"
            cv2.imwrite(str(raw_path), raw_overlay)
        except Exception as exc:
            LOGGER.debug("Failed to save slot debug overlays: %s", exc)

        self.card_slots = slots
        LOGGER.info("[OK] find_card_slots: succeeded in %.0fms - %d slots cached", duration, len(slots))
        
        # FRAME DIMENSION TRACKING: Log the frame used for slot finding
        LOGGER.info(f"📏 [SLOT FINDING] Frame used: shape={frame.shape}, will use these slot positions for motion detection")
        for i, (x, y, w, h) in enumerate(slots[:8], start=1):
            LOGGER.info(f"   Slot {i}: x={x}, y={y}, w={w}, h={h}")
        
        return slots

    def _slots_to_original(self, layout, frame_shape):
        orig_w = frame_shape[1]
        orig_h = frame_shape[0]
        normalization = getattr(layout, 'normalization', None)
        if normalization and isinstance(normalization, dict):
            target_w, target_h = normalization.get('target_size', normalization.get('normalized', (orig_w, orig_h)))
            portrait_w, portrait_h = normalization.get('portrait_size', (orig_w, orig_h))
            offset_x = normalization.get('offset_x', 0.0)
            offset_y = normalization.get('offset_y', 0.0)
        else:
            target_w, target_h = (orig_w, orig_h)
            portrait_w, portrait_h = (orig_w, orig_h)
            offset_x = 0.0
            offset_y = 0.0
        scale_to_portrait_x = portrait_w / target_w if target_w else 1.0
        scale_to_portrait_y = portrait_h / target_h if target_h else 1.0
        scale_portrait_to_raw_x = orig_w / portrait_w if portrait_w else 1.0
        scale_portrait_to_raw_y = orig_h / portrait_h if portrait_h else 1.0
        slots = []
        for slot in getattr(layout, 'slots', []) or []:
            try:
                x1, y1, x2, y2 = slot
            except Exception:
                continue
            x_portrait_1 = x1 * scale_to_portrait_x
            x_portrait_2 = x2 * scale_to_portrait_x
            y_portrait_1 = y1 * scale_to_portrait_y
            y_portrait_2 = y2 * scale_to_portrait_y
            x_raw_1 = (x_portrait_1 + offset_x) * scale_portrait_to_raw_x
            x_raw_2 = (x_portrait_2 + offset_x) * scale_portrait_to_raw_x
            y_raw_1 = (y_portrait_1 + offset_y) * scale_portrait_to_raw_y
            y_raw_2 = (y_portrait_2 + offset_y) * scale_portrait_to_raw_y
            x = int(round(x_raw_1)) + CARD_SLOT_HORIZONTAL_SHIFT
            y = int(round(y_raw_1)) + getattr(self, 'dynamic_slot_vertical_offset', 0)
            w = int(round(x_raw_2 - x_raw_1))
            h = int(round(y_raw_2 - y_raw_1))
            if w <= 0 or h <= 0:
                continue
            if y < 0:
                h += y
                y = 0
            if y + h > orig_h:
                h = orig_h - y
            x = max(0, min(orig_w - 1, x))
            y = max(0, min(orig_h - 1, y))
            w = max(1, min(orig_w - x, w))
            h = max(1, min(orig_h - y, h))
            slots.append((x, y, w, h))
        return slots

    def _fallback_slots(self, frame):
        h, w = frame.shape[:2]
        slot_width = max(1, w // 10)
        slot_height = max(1, h // 12)
        gap = max(1, slot_width // 10)
        x = gap
        y = gap
        slots = []
        for _ in range(8):
            slots.append((x, y, slot_width, slot_height))
            x += slot_width + gap
        self.card_slots = slots
        return slots

    def detect_card_in_slot(self, frame, slot, index, skip_motion_gate=False, motion_detected=False):
        LOGGER.info(f"🔎 detect_card_in_slot called for slot {index} (skip_motion_gate={skip_motion_gate}, motion_detected={motion_detected})")
        try:
            x, y, w, h = slot
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            if x >= x2 or y >= y2:
                LOGGER.info(f"[WARN] Slot {index}: Invalid coordinates")
                return None, 0.0
            crop = frame[y:y2, x:x2]
            if crop.size == 0:
                LOGGER.info(f"[WARN] Slot {index}: Empty crop")
                return None, 0.0
            
            # ========== MOTION-GATED DETECTION ==========
            # Track if motion was detected for this slot (for unknown retry logic)
            # Use passed motion_detected flag (from outer loop) OR detect it here
            motion_was_detected = motion_detected
            
            # Skip motion gating during initial identification sweep
            if not skip_motion_gate:
                # If card already identified (not unknown), check for replays with grace period/cooldown
                if index in self.slot_last_card:
                    cached_card = self.slot_last_card[index]
                    # Only check replays if card is truly identified (not "unknown")
                    if cached_card != "unknown" and cached_card is not None:
                        # Card is identified - check for motion (potential replay)
                        has_motion = self._has_slot_changed(index, crop)
                        
                        if has_motion:
                            current_time = time.time()
                            
                            # CHECK 1: Grace period - ignore motion right after identification (settling animation)
                            if index in self.slot_identification_time:
                                time_since_identification = current_time - self.slot_identification_time[index]
                                if time_since_identification < self.initial_motion_grace_period:
                                    LOGGER.debug(
                                        f"[GRACE] Slot {index}: Motion detected but within grace period "
                                        f"({time_since_identification:.2f}s / {self.initial_motion_grace_period}s) - ignoring settling animation"
                                    )
                                    return cached_card, 1.0  # Ignore motion, return cached card
                            
                            # CHECK 2: Replay cooldown - prevent duplicate detections of same replay
                            if index in self.slot_last_replay_time:
                                time_since_last_replay = current_time - self.slot_last_replay_time[index]
                                if time_since_last_replay < self.replay_cooldown:
                                    LOGGER.debug(
                                        f"[COOLDOWN] Slot {index}: Motion detected but within cooldown "
                                        f"({time_since_last_replay:.2f}s / {self.replay_cooldown}s) - ignoring duplicate"
                                    )
                                    return cached_card, 1.0  # Ignore motion, return cached card
                            
                            # Past grace period AND cooldown → this is a genuine REPLAY!
                            self.slot_last_replay_time[index] = current_time
                            LOGGER.info(f"[REPLAY] Slot {index}: '{cached_card}' REPLAYED (grace period passed, cooldown expired)")
                            # Bubble detector will handle elixir cost for replay
                        
                        # No motion or replay logged - return cached card
                        return cached_card, 1.0
            
                # Card not yet identified or still "unknown" → run motion detection
                has_motion = self._has_slot_changed(index, crop)
                motion_was_detected = motion_was_detected or has_motion  # Track for unknown retry logic (OR with passed flag)
        
                # If motion was just detected, check if we should delay ResNet
                if has_motion and index in self.slot_motion_detected_time:
                    time_since_motion = time.time() - self.slot_motion_detected_time[index]
                
                    # If motion detected recently but delay hasn't passed, skip ResNet for now
                    if time_since_motion < self.resnet_delay_after_motion:
                        # Still in animation - don't run ResNet yet, return None to skip this frame
                        LOGGER.info(f"🕐 Slot {index}: Motion detected {time_since_motion:.2f}s ago, waiting for animation...")
                        return None, 0.0  # Don't return cached card, wait for animation
            
                # If no motion detected, return cached card (if exists AND not unknown)
                if not has_motion and index in self.slot_last_card:
                    cached_card = self.slot_last_card[index]
                    # Only return cache if card is actually identified (not "unknown")
                    if cached_card not in ["unknown", None]:
                        return cached_card, 1.0  # High confidence (from previous detection)
            
                # If no motion detected and slot is unknown, skip ResNet entirely
                if not has_motion:
                    LOGGER.info(f"🔒 Slot {index}: No motion detected, skipping ResNet")
                    return None, 0.0  # No change, skip this slot
            
            # Motion detected AND delay passed (or first frame) → run full ResNet detection
            # OR skip_motion_gate=True (initial identification sweep)
            best_name = None
            best_score = 0.0
            if self.resnet_embedder is not None:
                try:
                    matches = self.resnet_embedder.find_best_match(crop, top_k=3)  # Get top-3 for tiebreaker
                except Exception as exc:
                    LOGGER.warning("ResNet match failed: %s", exc)
                    matches = []
                if matches:
                    candidate, score = matches[0]
                    
                    # 🎯 SMART TIEBREAKER: Check if Firecracker and Musk both in top-3 with close scores
                    # If both present with difference ≤ 0.02, prefer Musk (100% accuracy in testing)
                    firecracker_match = None
                    musk_match = None
                    for name, conf in matches[:3]:
                        if name == "Firecracker":
                            firecracker_match = (name, conf)
                        elif name == "Musk":
                            musk_match = (name, conf)
                    
                    # Apply tiebreaker if both cards detected
                    if firecracker_match and musk_match:
                        fc_conf = firecracker_match[1]
                        musk_conf = musk_match[1]
                        diff = abs(fc_conf - musk_conf)
                        
                        if diff <= 0.02:  # Threshold for "essentially tied" detections
                            # Very close scores - choose Musk
                            candidate = "Musk"
                            score = musk_conf
                            LOGGER.info(
                                f"[TIEBREAKER] Slot {index}: Firecracker ({fc_conf:.4f}) vs Musk ({musk_conf:.4f}), diff={diff:.4f} ≤ 0.02 → choosing Musk"
                            )
                    
                    if score >= self.UNKNOWN_EMBED_THRESHOLD:
                        best_name, best_score = candidate, score
            if best_name is None:
                candidate, score = self._template_match_card(crop)
                if candidate and score >= self.UNKNOWN_TEMPLATE_THRESHOLD:
                    best_name, best_score = candidate, score
            if best_name is not None and best_name in getattr(self, 'card_aliases', {}):
                resolved = self.card_aliases[best_name]
                if resolved:
                    best_name = resolved
            
            # CRITICAL LOGIC: If motion was detected but ResNet returned unknown, retry!
            # Motion = something changed = card must exist, so "unknown" is a detection failure
            if motion_was_detected and best_name in ["unknown", "unknown_png", None]:
                # Throttle warning spam - only log once per slot every 5 seconds
                if not hasattr(self, '_slot_unknown_warnings'):
                    self._slot_unknown_warnings = {}
                now = time.time()
                last_warn_time = self._slot_unknown_warnings.get(index, 0)
                if now - last_warn_time > 5.0:
                    LOGGER.warning(f"[RESET] Slot {index}: Motion detected but ResNet returned '{best_name}' - FORCING RETRY (motion means card must exist!)")
                    self._slot_unknown_warnings[index] = now
                # Don't cache unknown result, keep motion detection active for next frame
                # Return None to signal "no detection yet" and keep trying
                return None, 0.0
            
            # Cache the detected card for this slot
            if best_name is not None:
                self.slot_last_card[index] = best_name
                # ONLY enable bubble detection if card is actually identified (not "unknown")
                if best_name not in ["unknown", "unknown_png"]:
                    # Check if this is a problematic card that requires consensus
                    is_problematic = best_name.lower().replace(" ", "").replace("_", "") in self.PROBLEMATIC_CARDS_REQUIRE_CONSENSUS
                    
                    if is_problematic:
                        # Don't mark as identified yet - let consensus handle it
                        LOGGER.info(f"[WARN] Slot {index}: ResNet detected problematic card '{best_name}' (score={best_score:.2f}) - waiting for consensus")
                    else:
                        # Normal card - mark as identified immediately
                        self.identified_slots.add(index)
                        
                        # Track identification time for grace period (NEW - filters settling animation)
                        if index not in self.slot_identification_time:
                            self.slot_identification_time[index] = time.time()
                            LOGGER.info(f"[OK] Slot {index}: First identification of '{best_name}' - starting {self.initial_motion_grace_period}s grace period")
                        
                        # DON'T clear motion timestamp here - it's needed for elixir backdating
                        # Will be cleared in integrate_detected_cards AFTER elixir cost is applied
                        LOGGER.info(f"[OK] Slot {index}: ResNet identified '{best_name}', bubble detection enabled")
                else:
                    LOGGER.info(f"ℹ️ Slot {index}: ResNet returned 'unknown' (score={best_score:.2f}) - will retry with motion detection")
            
            return best_name, best_score
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, 0.0

    def _load_vs_timer_templates(self) -> None:
        """Load VS badge and timer templates from disk."""
        # Use absolute path from shared settings so packaging builds stay consistent
        templates_dir = TEMPLATES_DIR
        vs_dir = templates_dir / "match_start"
        timer_dir = templates_dir / "match_timer"
        vs_templates = []
        if vs_dir.exists():
            for template_file in vs_dir.glob("vs_badge_*.png"):
                template = cv2.imread(str(template_file))
                if template is not None:
                    vs_templates.append((template, template_file.name))
        timer_templates = []
        if timer_dir.exists():
            for template_file in timer_dir.glob("timeleft*.png"):
                template = cv2.imread(str(template_file))
                if template is not None:
                    timer_templates.append((template, template_file.name))
        if vs_templates:
            self.templates['vs_badges'] = vs_templates
        if timer_templates:
            self.templates['timer_left'] = timer_templates
        LOGGER.info(
            "Loaded %d VS badge templates and %d timer templates",
            len(self.templates.get('vs_badges', [])),
            len(self.templates.get('timer_left', [])),
        )

    def _init_timer_ocr(self) -> None:
        """Initialize PaddleOCR backend for timer reading."""
        if not OCR_AVAILABLE:
            LOGGER.info("Timer OCR not available - install paddleocr")
            return
        try:
            LOGGER.info("Timer OCR initialized")
        except Exception as exc:
            LOGGER.warning("Timer OCR failed to initialize: %s", exc)
            self.ocr = None

    def _template_match_card(self, crop):
        if not self.card_templates_gray:
            return None, 0.0
        try:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None, 0.0
        best_name = None
        best_score = -1.0
        for name, template_gray in self.card_templates_gray.items():
            try:
                resized = cv2.resize(crop_gray, (template_gray.shape[1], template_gray.shape[0]))
            except Exception:
                continue
            result = cv2.matchTemplate(resized, template_gray, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score

class ClashRoyaleTracker(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clash Royale Elixir Tracker")
        self.setGeometry(100, 100, 1000, 700)

        self.window_capture = WindowCapture()

        self.tracker_thread = ElixirTracker()

        self.tracker_thread.frame_update.connect(self.update_frame_display)
        self.tracker_thread.elixir_update.connect(self.update_elixir_display)
        self.tracker_thread.card_detected.connect(self.update_card_detection)
        self.tracker_thread.card_cycle_update.connect(self.update_card_cycle_display)
        self.tracker_thread.timer_update.connect(self.update_timer_display)
        self.tracker_thread.status_update.connect(self.update_status)
        # When tracker requests a snapshot (to prime slot detection), run the UI snapshot helper
        self.tracker_thread.request_snapshot.connect(self.detect_cards_once)

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Display
        right_panel = self.create_display_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_control_panel(self):
        """Create the control panel with settings."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)

        # Window selection
        window_group = QGroupBox("Window Selection")
        window_layout = QVBoxLayout(window_group)

        self.window_combo = QComboBox()
        self.refresh_windows_btn = QPushButton("Refresh Windows")
        self.refresh_windows_btn.clicked.connect(self.refresh_windows)

        window_layout.addWidget(QLabel("Select Clash Royale window to spectate:"))
        window_layout.addWidget(self.window_combo)
        window_layout.addWidget(self.refresh_windows_btn)

        # Detect cards button (works even if tracking not started)
        self.detect_cards_btn = QPushButton("Detect Cards (snapshot)")
        # When user clicks the button, allow a blocking call so they get immediate results.
        # Background/requested snapshots (via request_snapshot) will call without blocking.
        self.detect_cards_btn.clicked.connect(lambda checked=False: self.detect_cards_once(block_for_result=True))
        window_layout.addWidget(self.detect_cards_btn)

        # Read Match Timer button (tester - can start OCR without VS badge)
        self.read_timer_btn = QPushButton("Read Match Timer")
        self.read_timer_btn.clicked.connect(self.read_match_timer)
        window_layout.addWidget(self.read_timer_btn)

        layout.addWidget(window_group)

        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Reset")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.reset_tracking)

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        layout.addWidget(controls_group)

        # Debug controls
        debug_group = QGroupBox("Debug Controls")
        debug_layout = QVBoxLayout(debug_group)
        
        # Motion debug checkbox
        self.motion_debug_checkbox = QCheckBox("Enable Motion Debug Logging")
        self.motion_debug_checkbox.setToolTip("Log detailed motion detection info for each slot")
        self.motion_debug_checkbox.stateChanged.connect(self.toggle_motion_debug)
        debug_layout.addWidget(self.motion_debug_checkbox)
        
        # Disable motion gate checkbox
        self.disable_motion_gate_checkbox = QCheckBox("Disable Motion Gate (Always Run ResNet)")
        self.disable_motion_gate_checkbox.setToolTip("Bypass motion detection - run ResNet on every frame (slow)")
        self.disable_motion_gate_checkbox.stateChanged.connect(self.toggle_motion_gate)
        debug_layout.addWidget(self.disable_motion_gate_checkbox)
        
        layout.addWidget(debug_group)

        # Status log
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout(status_group)

        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(150)
        status_layout.addWidget(self.status_log)
        layout.addWidget(status_group)

        layout.addStretch()

        # Initialize window list
        self.refresh_windows()

        return panel
    
    def create_display_panel(self):
        """Create the display panel showing tracking results."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Elixir display
        elixir_group = QGroupBox("Elixir Tracking")
        elixir_layout = QGridLayout(elixir_group)
        
        # Enemy Player (main tracking target)
        elixir_layout.addWidget(QLabel("Enemy Elixir:"), 0, 0)
        self.enemy_elixir_label = QLabel("0.0")
        self.enemy_elixir_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.enemy_elixir_label.setStyleSheet("color: red;")
        elixir_layout.addWidget(self.enemy_elixir_label, 0, 1)
        
        # Match Timer
        elixir_layout.addWidget(QLabel("Match Timer:"), 1, 0)
        self.match_timer_label = QLabel("--:--")
        self.match_timer_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.match_timer_label.setStyleSheet("color: blue;")
        elixir_layout.addWidget(self.match_timer_label, 1, 1)
        
        layout.addWidget(elixir_group)
        
        # Card cycle display
        cycle_group = QGroupBox("Opponent Card Cycle")
        cycle_layout = QVBoxLayout(cycle_group)
        
        # Cards in Hand (1-4)
        cycle_layout.addWidget(QLabel("Cards in Hand:"))
        self.hand_layout = QHBoxLayout()
        self.hand_cards = []
        for i in range(4):
            card_label = QLabel()
            card_label.setMinimumSize(60, 80)
            card_label.setMaximumSize(60, 80)
            card_label.setStyleSheet("border: 2px solid blue; background-color: lightgray;")
            card_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_label.setScaledContents(True)  # Scale image to fit
            self.set_unknown_card_image(card_label)
            self.hand_cards.append(card_label)
            self.hand_layout.addWidget(card_label)
        cycle_layout.addLayout(self.hand_layout)
        
        # Upcoming Cards (5-8)
        cycle_layout.addWidget(QLabel("Upcoming Cards:"))
        self.upcoming_layout = QHBoxLayout()
        self.upcoming_cards = []
        for i in range(4):
            card_label = QLabel()
            card_label.setMinimumSize(60, 80)
            card_label.setMaximumSize(60, 80)
            card_label.setStyleSheet("border: 2px solid orange; background-color: lightgray;")
            card_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_label.setScaledContents(True)  # Scale image to fit
            self.set_unknown_card_image(card_label)
            self.upcoming_cards.append(card_label)
            self.upcoming_layout.addWidget(card_label)
        cycle_layout.addLayout(self.upcoming_layout)
        
        layout.addWidget(cycle_group)
        
        # Frame display
        frame_group = QGroupBox("Live Feed")
        frame_layout = QVBoxLayout(frame_group)
        
        self.frame_label = QLabel("No frame")
        self.frame_label.setMinimumHeight(300)
        self.frame_label.setStyleSheet("border: 1px solid gray;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.frame_label)
        
        layout.addWidget(frame_group)
        
        # Card detection
        card_group = QGroupBox("Card Detection")
        card_layout = QVBoxLayout(card_group)
        
        self.last_card_label = QLabel("No cards detected")
        card_layout.addWidget(self.last_card_label)
        
        layout.addWidget(card_group)
        
        return panel
    
        return panel
    
    def refresh_windows(self):
        """Refresh the list of available windows."""
        self.window_combo.clear()
        windows = self.window_capture.get_window_list()
        
        for hwnd, title in windows:
            self.window_combo.addItem(title, hwnd)
        
        self.log_status(f"Found {len(windows)} windows")
    
    def start_tracking(self):
        """Start the elixir tracking."""
        # Get selected window for normal mode
        selected_index = self.window_combo.currentIndex()
        if selected_index >= 0:
            hwnd = self.window_combo.itemData(selected_index)
            success = self.tracker_thread.set_window(hwnd)
            if not success:
                self.log_status("Failed to set target window")
                return

            # Success
            window_title = self.window_combo.currentText()
            self.log_status(f"Selected window: {window_title}")
        else:
            self.log_status("Please select a window first")
            return
        
        # Always use normal mode
        self.tracker_thread.set_mode("normal")
        
        # Reset VS badge gating state before starting the thread
        self.tracker_thread.waiting_for_vs_clear = False
        self.tracker_thread.vs_badge_seen = False
        self.tracker_thread.game_started = False
        self.tracker_thread.prepare_for_new_match()
        self.update_elixir_display(self.tracker_thread.enemy_elixir)
        self.tracker_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.log_status("Started tracking in normal mode")
    
    def stop_tracking(self):
        """Stop the elixir tracking."""
        self.tracker_thread.stop()
        self.tracker_thread.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_status("Stopped tracking")

    def reset_tracking(self):
        """Reset tracking state without stopping the app thread.

        This clears the opponent card cycle to initialization state and resets
        the enemy elixir back to 5.0 so Start Tracking can look for VS badge again.
        """
        # If the thread is running, stop it cleanly
        if self.tracker_thread.isRunning():
            try:
                self.tracker_thread.stop()
                self.tracker_thread.wait(timeout=1000)
            except Exception:
                pass

        # Reset internal tracker state
        self.tracker_thread.opponent_cards_in_hand = ["unknown_png"] * 4
        self.tracker_thread.opponent_upcoming_cards = ["unknown_png"] * 4
        self.tracker_thread.enemy_elixir = INITIAL_ELIXIR
        self.tracker_thread.match_start_time = None
        self.tracker_thread.current_phase = 'single'
        self.tracker_thread.game_started = False
        self.tracker_thread.waiting_for_vs_clear = False
        self.tracker_thread.vs_badge_seen = False
        
        # Reset motion detection for fresh card identification
        self.tracker_thread.reset_motion_detection()
        self.tracker_thread.reset_slot_locks()
        self.tracker_thread.prepare_for_new_match()
        
        # Reset timer detection state
        self.tracker_thread.timer_merged_bbox = None
        self.tracker_thread.timer_roi_bounds = None
        self.tracker_thread.timer_detection_stage = "initial"
        self.tracker_thread.timer_initialized = False
        self.tracker_thread.timer_ocr_started = False
        self.tracker_thread.vs_badge_seen_time = None
        
        # Reset timer countdown state
        if hasattr(self.tracker_thread, 'timer_countdown_active'):
            self.tracker_thread.timer_countdown_active = False
        if hasattr(self.tracker_thread, '_last_timer_display'):
            self.tracker_thread._last_timer_display = None

        # UI update
        self.update_card_cycle_display(self.tracker_thread.opponent_cards_in_hand, self.tracker_thread.opponent_upcoming_cards)
        self.update_elixir_display(self.tracker_thread.enemy_elixir)

        # Allow starting again
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.log_status("Tracker reset: opponent cycle cleared, enemy elixir reset to 5.0.")
    
    def toggle_motion_debug(self, state):
        """Toggle motion debug logging."""
        enabled = (state == Qt.CheckState.Checked.value)
        self.tracker_thread.motion_debug = enabled
        status = "ENABLED" if enabled else "DISABLED"
        self.log_status(f"Motion debug logging: {status}")
        LOGGER.info(f"🔧 Motion debug logging {status}")
    
    def toggle_motion_gate(self, state):
        """Toggle motion gate (bypass motion detection)."""
        enabled = (state == Qt.CheckState.Checked.value)
        self.tracker_thread.disable_motion_gate = enabled
        status = "DISABLED (ResNet runs every frame)" if enabled else "ENABLED (motion-gated)"
        self.log_status(f"Motion gate: {status}")
        LOGGER.info(f"🔧 Motion gate {status}")

    def detect_cards_once(self, block_for_result: bool = False):
        """Snapshot current screen and populate opponent card cycle for testing.

        This runs even if tracking hasn't been started. It captures one frame
        from the selected window (or current last captured frame) and uses
        the same detection pipeline to fill opponent card arrays. Order is not
        guaranteed; this is a convenience function for quick testing.
        """
        # Try to capture a live frame from window capture if available
        frame = None
        try:
            # If window set, use its capture; otherwise fall back to last_captured_frame
            if self.tracker_thread.window_capture.hwnd:
                frame = self.tracker_thread.window_capture.capture_window(full_window=True)
        except Exception:
            frame = None

        if frame is None:
            frame = self.tracker_thread.last_captured_frame

        if frame is None:
            self.log_status("No frame available to detect cards from.")
            return

        self.tracker_thread.last_captured_frame = frame

        # Attempt to get cached slots. If a background slot-finding job is
        # currently running, either wait briefly (if user requested a blocking call)
        # or schedule a non-blocking retry to avoid freezing the UI.
        slots = self.tracker_thread.card_slots or []
        if not slots and getattr(self.tracker_thread, '_slot_find_future', None):
            future = self.tracker_thread._slot_find_future
            # If the caller asked to block (user clicked button), wait briefly.
            if block_for_result:
                try:
                    # Wait up to 0.5s for background slot finding to complete
                    slots = future.result(timeout=0.5) or []
                except Exception:
                    slots = self.tracker_thread.card_slots or []
            else:
                # Non-blocking: schedule a retry later and return immediately to keep UI responsive
                try:
                    QTimer.singleShot(500, lambda: self.detect_cards_once(block_for_result=False))
                except Exception:
                    pass
                self.log_status("Slot finding in progress - scheduled non-blocking retry")
                return

        # If still no slots, fall back to synchronous call but with a short
        # timeout to avoid freezing the UI for many seconds.
        if not slots:
            try:
                if getattr(self.tracker_thread, '_executor', None):
                    if block_for_result:
                        # Blocking user-requested run: submit and wait briefly
                        fut = self.tracker_thread._executor.submit(self.tracker_thread.find_card_slots, frame)
                        slots = fut.result(timeout=1.0) or []
                    else:
                        # Non-blocking: submit job and schedule a retry instead of waiting
                        try:
                            if not getattr(self.tracker_thread, '_slot_find_future', None) or self.tracker_thread._slot_find_future.done():
                                self.tracker_thread._slot_find_future = self.tracker_thread._executor.submit(self.tracker_thread.find_card_slots, frame.copy())
                                # Attach a done-callback to notify when ready
                                try:
                                    self.tracker_thread._slot_find_future.add_done_callback(lambda fut: self.tracker_thread.status_update.emit("Background slot finding completed"))
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        QTimer.singleShot(700, lambda: self.detect_cards_once(block_for_result=False))
                        self.log_status("Submitted background slot-finder and scheduled retry (non-blocking)")
                        return
                else:
                    # No executor available - run synchronously only if user explicitly requested it
                    if block_for_result:
                        slots = self.tracker_thread.find_card_slots(frame)
                    else:
                        self.log_status("No executor available for non-blocking slot detection; skipping")
                        return
            except Exception:
                slots = self.tracker_thread.card_slots or []

        if not slots:
            self.log_status("No card slots found for detection.")
            return

        detected = []
        # Save a debug overlay showing the slot rectangles that will be classified
        overlay_filename = None
        try:
            overlay = frame.copy()
            for idx, (slot_x, slot_y, slot_w, slot_h) in enumerate(slots[:8]):
                cv2.rectangle(overlay, (slot_x, slot_y), (slot_x + slot_w, slot_y + slot_h), (0, 255, 0), 2)
                cv2.putText(overlay, f"{idx + 1}", (slot_x + 4, max(15, slot_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            overlay_filename = f"detect_cards_overlay_{int(time.time())}.png"
            cv2.imwrite(overlay_filename, overlay)
            self.log_status(f"Saved detect-cards overlay: {overlay_filename}")
        except Exception as exc:
            self.log_status(f"Failed to save detect-cards overlay: {exc}")

        prev_cards = (self.tracker_thread.opponent_cards_in_hand +
                      self.tracker_thread.opponent_upcoming_cards)
        for i, slot in enumerate(slots[:8]):
            name, score = self.tracker_thread.detect_card_in_slot(frame, slot, i)
            if not name:
                name = "unknown_png"
            if prev_cards and i < len(prev_cards):
                previous = prev_cards[i]
                if previous not in {"unknown", "unknown_png"}:
                    if name in {"unknown", "unknown_png"} or score < 0.45:
                        self.log_status(
                            f"Slot {i + 1}: keeping previous detection {previous} (new conf={score:.2f})"
                        )
                        name = previous
                        score = 1.0
            detected.append((name, score))
            self.log_status(f"Slot {i + 1}: {name} (conf={score:.2f})")

        changed = self.tracker_thread.integrate_detected_cards(
            detected, log_source="Detect Cards", log_deferrals=True
        )
        if changed:
            self.log_status("Detect Cards: cycle merged with order enforcement")
        # Removed spam: "Detect Cards: cycle unchanged"

        # Ensure GUI reflects latest state immediately
        self.update_card_cycle_display(
            self.tracker_thread.opponent_cards_in_hand,
            self.tracker_thread.opponent_upcoming_cards
        )

    def read_match_timer(self):
        """Manually trigger timer OCR using new two-stage approach (tester button - no VS badge required)."""
        import cv2, numpy as np, time
        
        if not self.tracker_thread.ocr:
            self.log_status("Timer OCR not available - missing OCR")
            return

        frame = None
        hwnd = getattr(self.tracker_thread.window_capture, 'hwnd', None)
        self.log_status(f"[Timer OCR Debug] Window handle: {hwnd}")
        try:
            if hwnd:
                frame = self.tracker_thread.window_capture.capture_window()
                self.log_status(f"[Timer OCR Debug] capture_window() returned frame: {type(frame)}, shape: {getattr(frame, 'shape', None)}")
        except Exception as e:
            self.log_status(f"[Timer OCR Debug] Exception during capture_window: {e}")
            frame = None

        if frame is None:
            frame = self.tracker_thread.last_captured_frame
            self.log_status(f"[Timer OCR Debug] Using last_captured_frame: {type(frame)}, shape: {getattr(frame, 'shape', None)}")

        if frame is None:
            self.log_status("No frame available for timer OCR test")
            return

        # Check if frame is all black or empty
        if frame.size == 0 or np.sum(frame) == 0:
            self.log_status("[Timer OCR Debug] Frame is empty or all black. Window may be minimized or hidden.")
            # Save frame for inspection
            try:
                cv2.imwrite("debug_timer_frame_empty.png", frame)
                self.log_status("[Timer OCR Debug] Saved empty frame as debug_timer_frame_empty.png")
            except Exception as e:
                self.log_status(f"[Timer OCR Debug] Failed to save empty frame: {e}")
            return

        # Save frame for inspection
        try:
            cv2.imwrite("debug_timer_frame.png", frame)
            self.log_status("[Timer OCR Debug] Saved frame as debug_timer_frame.png")
        except Exception as e:
            self.log_status(f"[Timer OCR Debug] Failed to save frame: {e}")

        # Use new two-stage timer detection approach
        self.log_status("[Timer OCR Debug] Using new two-stage timer detection...")
        
        # Force timer OCR to run (bypass VS badge requirement)
        self.tracker_thread.timer_ocr_started = True
        self.tracker_thread.vs_badge_seen_time = time.time() - 3
        
        # Reset timer detection state to initial stage
        self.tracker_thread.timer_detection_stage = "initial"
        self.tracker_thread.timer_merged_bbox = None
        self.tracker_thread.timer_roi_bounds = None
        
        # Run the new timer detection with debug overlays enabled
        self.tracker_thread.timer_debug_request = True
        try:
            self.tracker_thread.detect_match_timer(frame)
        finally:
            self.tracker_thread.timer_debug_request = False
        
        # Save debug images from the new approach
        try:
            # Save ROI if it was extracted
            if hasattr(self.tracker_thread, 'timer_roi_bounds') and self.tracker_thread.timer_roi_bounds:
                roi_x_start, roi_y_start, roi_w, roi_h = self.tracker_thread.timer_roi_bounds
                roi = frame[roi_y_start:roi_y_start+roi_h, roi_x_start:roi_x_start+roi_w]
                cv2.imwrite("debug_timer_roi.png", roi)
                self.log_status("[Timer OCR Debug] Saved ROI as debug_timer_roi.png")
                
                # Save processed ROI
                processed_roi = self.tracker_thread.preprocess_timer_region(roi)
                cv2.imwrite("debug_timer_roi_processed.png", processed_roi)
                self.log_status("[Timer OCR Debug] Saved processed ROI as debug_timer_roi_processed.png")
            
            # Save merged bbox crop if available
            if hasattr(self.tracker_thread, 'timer_merged_bbox') and self.tracker_thread.timer_merged_bbox:
                x_min, y_min, x_max, y_max = self.tracker_thread.timer_merged_bbox
                timer_crop = frame[y_min:y_max, x_min:x_max]
                cv2.imwrite("debug_timer_crop.png", timer_crop)
                self.log_status("[Timer OCR Debug] Saved merged bbox crop as debug_timer_crop.png")

                # Save overlay with merged bbox
                overlay = frame.copy()
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imwrite("debug_timer_overlay.png", overlay)
                self.log_status("[Timer OCR Debug] Saved overlay with merged bbox as debug_timer_overlay.png")
                
        except Exception as e:
            self.log_status(f"[Timer OCR Debug] Failed to save debug images: {e}")

        # After running timer detection, log OCR debug info
        if hasattr(self.tracker_thread, 'last_timer_ocr_debug'):
            self.log_status(f"Timer OCR debug: {self.tracker_thread.last_timer_ocr_debug}")
        else:
            self.log_status("Timer OCR debug: No debug info available")

        # Reset the flags so normal operation continues
        self.tracker_thread.timer_ocr_started = False
        self.tracker_thread.vs_badge_seen_time = None

        self.log_status("Timer OCR test completed")
    
    def update_frame_display(self, frame):
        """Update the frame display with latest capture."""
        # Convert OpenCV frame to Qt format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.frame_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.frame_label.setPixmap(scaled_pixmap)
    
    def update_elixir_display(self, enemy_elixir):
        """Update elixir counter display with decimal precision."""
        LOGGER.info(f"📊 GUI Update: Elixir={enemy_elixir:.3f} → Display will show {enemy_elixir:.1f}")
        self.enemy_elixir_label.setText(f"{enemy_elixir:.1f}")
    
    def update_timer_display(self, timer_text):
        """Update match timer display."""
        self.match_timer_label.setText(timer_text)
    
    def update_card_detection(self, card_name, confidence):
        """Update card detection display."""
        self.last_card_label.setText(f"Detected: {card_name} ({confidence:.2f})")
        self.log_status(f"Card detected: {card_name} (confidence: {confidence:.2f})")
    
    def update_card_cycle_display(self, cards_in_hand, upcoming_cards):
        """Update the card cycle display with actual card images."""
        # Update cards in hand (positions 1-4)
        for i, card_name in enumerate(cards_in_hand):
            if i < len(self.hand_cards):
                if card_name == "unknown_png":
                    self.set_unknown_card_image(self.hand_cards[i])
                    self.hand_cards[i].setStyleSheet("border: 2px solid blue; background-color: lightgray;")
                else:
                    self.set_card_image(self.hand_cards[i], card_name)
                    self.hand_cards[i].setStyleSheet("border: 2px solid blue; background-color: lightblue;")
        
        # Update upcoming cards (positions 5-8)
        for i, card_name in enumerate(upcoming_cards):
            if i < len(self.upcoming_cards):
                if card_name == "unknown_png":
                    self.set_unknown_card_image(self.upcoming_cards[i])
                    self.upcoming_cards[i].setStyleSheet("border: 2px solid orange; background-color: lightgray;")
                else:
                    self.set_card_image(self.upcoming_cards[i], card_name)
                    self.upcoming_cards[i].setStyleSheet("border: 2px solid orange; background-color: lightyellow;")
    
    def set_card_image(self, label, card_name):
        """Set the card image for a label."""
        from pathlib import Path
        

        # Try to load the card image
        base_dir = CARD_IMAGES_DIR / "card_ed"
        evo_dir = CARD_IMAGES_DIR / "card_ed_evo"
        lower_name = card_name.lower()

        search_candidates = []
        seen = set()

        def add_candidate(directory, name):
            if not name:
                return
            candidate = directory / f"{name}.png"
            key = str(candidate).lower()
            if key in seen:
                return
            seen.add(key)
            search_candidates.append(candidate)

        # Default tooltip shows underscores as spaces
        display_name = card_name.replace('_', ' ')

        # Handle evolution variants first so their artwork is preferred
        evo_suffixes = ["_evo", "-evo", "_evolution", "-evolution"]
        matched_suffix = None
        for suffix in evo_suffixes:
            if lower_name.endswith(suffix):
                matched_suffix = suffix
                break

        base_name = card_name
        if matched_suffix:
            base_name = card_name[:-len(matched_suffix)]
            display_name = f"{base_name.replace('_', ' ')} (Evolution)"
            add_candidate(evo_dir, base_name)
            add_candidate(evo_dir, base_name.lower())

        # Always allow direct lookups in the base directory too
        add_candidate(base_dir, card_name)
        add_candidate(base_dir, card_name.lower())
        if matched_suffix:
            add_candidate(base_dir, base_name)
            add_candidate(base_dir, base_name.lower())

        for card_image_path in search_candidates:
            if card_image_path.exists():
                pixmap = QPixmap(str(card_image_path))
                if not pixmap.isNull():
                    # Scale the image to fit the label (60x80)
                    scaled_pixmap = pixmap.scaled(
                        60, 80,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    label.setPixmap(scaled_pixmap)
                    label.setText("")
                    label.setToolTip(f"{display_name} (Card Image)")
                    return
        # Fallback to text if image not found
        label.setText(display_name)
        label.setPixmap(QPixmap())  # Clear any existing pixmap
        label.setToolTip(f"{display_name} (Text Fallback)")
        # Reset stylesheet for text display
        base_style = "border: 2px solid blue; background-color: lightblue;" if "blue" in label.styleSheet() else "border: 2px solid orange; background-color: lightyellow;"
        label.setStyleSheet(base_style + " font-size: 8px; color: black;")
    
    def set_unknown_card_image(self, label):
        """Set the unknown card placeholder."""
        label.setText("?")
        label.setPixmap(QPixmap())  # Clear any existing pixmap
        label.setToolTip("Unknown card")
        # Keep the border color but add text styling
        base_style = label.styleSheet().split(';')[0] + ";" + label.styleSheet().split(';')[1] + ";"  # Keep border and background
        label.setStyleSheet(base_style + " font-size: 20px; color: gray; font-weight: bold;")
    
    def update_status(self, message):
        """Update status log."""
        self.log_status(message)
    
    def log_status(self, message):
        """Add message to status log."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.status_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.tracker_thread.isRunning():
            self.tracker_thread.stop()
            self.tracker_thread.wait()
        event.accept()
