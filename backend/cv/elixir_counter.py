"""
Elixir counter with phase-based regeneration that degrades gracefully when
optional neural components are unavailable.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from config.constants import (
    ELIXIR_PHASES,
    INITIAL_ELIXIR,
    MAX_ELIXIR,
    DEFAULT_SPELL_COST,
    SPELL_COSTS,
)

try:
    from cv.spell_detector import SpellDetector  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    class SpellDetector:  # type: ignore
        """Fallback spell detector that returns default costs."""

        def detect_spell_around_stopwatch(self, frame: Any, event: Any) -> int:
            return DEFAULT_SPELL_COST

        def add_event(self, event: Any, spell_cost: int) -> None:
            pass

try:
    from cv.neural_card_detector import CardDetector  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    class CardDetector:  # type: ignore
        """Fallback card detector that provides no detections."""

        def reset_identifications(self) -> None:
            pass

        def detect_cards_from_frame(self, frame: Any) -> List[Dict[str, Any]]:
            return []

        def _find_stopwatches(self, frame: Any) -> List[Any]:
            return []

        def get_identified_cards(self) -> Dict[Any, Any]:
            return {}


class ElixirCounter:
    """Phase-aware elixir model with optional spell/card integrations."""

    def __init__(self) -> None:
        self.enemy_elixir: float = INITIAL_ELIXIR
        self.current_phase: str = "single"
        self.last_update: float = time.time()
        self.match_start_time: Optional[float] = None
        self.bonus_elixir: float = 0.0
        self.spell_detector = SpellDetector()
        self.card_detector = CardDetector()
        self._card_detection_debug_count = 0
        self.spent_elixir = 0.0
        self._initial_calculated_elixir = INITIAL_ELIXIR  # Will be recalculated on start
        self.phase_times = {
            "single": (0, 120),
            "double": (120, 180),
            "triple": (180, float("inf")),
        }

    def reset(self) -> None:
        self.enemy_elixir = INITIAL_ELIXIR
        self.current_phase = "single"
        self.last_update = time.time()
        self.match_start_time = None
        self.bonus_elixir = 0.0
        self.spent_elixir = 0.0
        if hasattr(self.card_detector, "reset_identifications"):
            self.card_detector.reset_identifications()

    def start_regeneration(self, backdate_seconds: float = 0.0) -> None:
        """Start elixir regeneration with reverse-calculated initial value.
        
        Instead of backdating, we calculate what the initial elixir should be
        such that it reaches exactly 9.0 elixir at the expected checkpoint time.
        This eliminates drift and makes the counter perfectly smooth.
        
        Math:
        - Checkpoint time: 3.15 seconds (time from match start to first detection)
        - Regen rate: 0.357 elixir/s (1 elixir every 2.8 seconds)
        - Target at checkpoint: 9.0 elixir
        - Elixir generated: 3.15s × 0.357 = 1.125 elixir
        - Initial elixir needed: 9.0 - 1.125 = 7.875 elixir
        
        Result: Counter starts at 7.875, regenerates smoothly, and hits exactly 9.0 at 3.15s
        """
        if self.match_start_time is not None:
            return
        
        # Match start time is NOW (no backdating)
        self.match_start_time = time.time()
        self.last_update = time.time()
        self.current_phase = "single"
        self.spent_elixir = 0.0
        
        # Reverse calculation: What initial elixir gives us 9.0 at checkpoint?
        checkpoint_time = 3.15  # seconds from match start to first detection
        regen_rate = ELIXIR_PHASES["single"]  # 0.357 elixir/s (1 per 2.8s)
        elixir_generated_by_checkpoint = checkpoint_time * regen_rate  # 1.125
        target_at_checkpoint = 9.0
        
        # Initial: 9.0 - 1.125 = 7.875 elixir
        self.enemy_elixir = target_at_checkpoint - elixir_generated_by_checkpoint
        
        # Store as baseline for calculations
        self._initial_calculated_elixir = self.enemy_elixir

    def start_regeneration_from_loading(self) -> None:
        """Start regeneration without backdating - use reverse calculation instead."""
        self.start_regeneration(backdate_seconds=0.0)

    def is_regenerating(self) -> bool:
        return self.match_start_time is not None

    def update(self) -> None:
        expected_total = self._expected_total()
        self.enemy_elixir = max(0.0, min(expected_total - self.spent_elixir, MAX_ELIXIR))
        self.last_update = time.time()

    def _update_phase(self, match_time: float) -> None:
        for phase, (start_time, end_time) in self.phase_times.items():
            if start_time <= match_time < end_time:
                self.current_phase = phase
                break

    def add_bonus(self, amount: float) -> None:
        if amount <= 0:
            return
        max_bonus = MAX_ELIXIR - INITIAL_ELIXIR
        self.bonus_elixir = min(max_bonus, self.bonus_elixir + amount)
        self.update()

    def process_enemy_event(self, event: Any, frame: Optional[Any] = None) -> None:
        spell_cost = DEFAULT_SPELL_COST
        if frame is not None and hasattr(self.spell_detector, "detect_spell_around_stopwatch"):
            try:
                detected = self.spell_detector.detect_spell_around_stopwatch(frame, event)
                if detected:
                    spell_cost = detected
            except Exception:
                spell_cost = DEFAULT_SPELL_COST
        elif isinstance(event, dict):
            spell_cost = SPELL_COSTS.get(event.get("spell"), DEFAULT_SPELL_COST)
        if spell_cost > 0:
            self.register_spend(spell_cost)
            if hasattr(self.spell_detector, "add_event"):
                try:
                    self.spell_detector.add_event(event, spell_cost)
                except Exception:
                    pass

    def process_card_detection(self, frame: Any) -> None:
        if not self.is_regenerating():
            return
        self._card_detection_debug_count += 1
        try:
            detected_cards = self.card_detector.detect_cards_from_frame(frame)
        except Exception:
            detected_cards = []
        if detected_cards:
            pass  # Placeholder for future card handling logic

    def _expected_total(self) -> float:
        """Calculate expected total elixir (initial + generated - spent).
        
        Uses the reverse-calculated initial elixir value instead of INITIAL_ELIXIR constant.
        """
        if self.match_start_time is None:
            return INITIAL_ELIXIR + self.bonus_elixir
        
        current_time = time.time()
        elapsed_since_start = current_time - self.match_start_time
        self._update_phase(elapsed_since_start)
        regen_rate = ELIXIR_PHASES[self.current_phase]
        
        # Use the calculated initial elixir (which makes us hit 9.0 at checkpoint)
        initial_elixir = getattr(self, '_initial_calculated_elixir', INITIAL_ELIXIR)
        total_generated = elapsed_since_start * regen_rate
        return initial_elixir + total_generated + self.bonus_elixir

    def register_spend(self, amount: float) -> None:
        if amount <= 0:
            return
        expected_total = self._expected_total()
        self.spent_elixir = min(expected_total, self.spent_elixir + amount)
        self.enemy_elixir = max(0.0, min(expected_total - self.spent_elixir, MAX_ELIXIR))

    def get_enemy_elixir(self) -> float:
        return self.enemy_elixir

    def get_current_phase(self) -> str:
        return self.current_phase

    def get_phase_multiplier(self) -> str:
        return {
            "single": "1x",
            "double": "2x",
            "triple": "3x",
        }.get(self.current_phase, "1x")

    def adjust_elixir(self, new_elixir: float) -> None:
        expected_total = self._expected_total()
        self.enemy_elixir = max(0.0, min(new_elixir, MAX_ELIXIR))
        self.spent_elixir = max(0.0, min(expected_total, expected_total - self.enemy_elixir))


