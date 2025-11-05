"""
Elixir counter with phase-based regeneration and card detection
"""
import time
from config.constants import (ELIXIR_PHASES, INITIAL_ELIXIR, MAX_ELIXIR, 
                             SPELL_COSTS, DEFAULT_SPELL_COST)
from cv.spell_detector import SpellDetector
from cv.neural_card_detector import CardDetector  # Now uses neural detection

class ElixirCounter:
    def __init__(self):
        self.enemy_elixir = INITIAL_ELIXIR
        self.current_phase = 'single'
        self.last_update = time.time()
        self.match_start_time = None
        self.bonus_elixir = 0.0
        self.spell_detector = SpellDetector()
        self.card_detector = CardDetector()  # Initialize card detector
        
        # Phase timing (standard Clash Royale match phases)
        self.phase_times = {
            'single': (0, 120),      # 0-2 minutes: single elixir
            'double': (120, 180),    # 2-3 minutes: double elixir  
            'triple': (180, float('inf'))  # 3+ minutes: triple elixir (overtime)
        }
        
    def reset(self):
        """Reset elixir counter for new match"""
        self.enemy_elixir = INITIAL_ELIXIR
        self.current_phase = 'single'
        self.last_update = time.time()
        self.match_start_time = None  # Clear match start time - no regeneration until match starts
        self.bonus_elixir = 0.0
        
        # Reset neural card detector smart tracking
        if hasattr(self.card_detector, 'reset_identifications'):
            self.card_detector.reset_identifications()
        
    def start_regeneration(self, backdate_seconds=3.15):
        """Start elixir regeneration from 5.0 with 3.15s backdate (loading screen optimized)"""
        # Only start if not already started (handles multiple detections gracefully)
        if self.match_start_time is None:
            self.enemy_elixir = INITIAL_ELIXIR  # Start at 5.0
            # Apply backdate for loading screen timing
            self.match_start_time = time.time() - backdate_seconds
            self.last_update = time.time()
            self.current_phase = 'single'
            
            print(f"💜 Elixir regeneration started! Starting at {self.enemy_elixir} elixir (backdated {backdate_seconds}s)")
        else:
            print(f"💜 Elixir already regenerating (current: {self.enemy_elixir:.1f})")
            
    def start_regeneration_from_loading(self):
        """Start elixir regeneration from loading screen detection (simplified to use 3.15s)"""
        # Use 3.15s backdate consistently for loading screen
        self.start_regeneration(backdate_seconds=3.15)
    
    def is_regenerating(self):
        """Check if elixir regeneration is active"""
        return self.match_start_time is not None
        
    def update(self):
        """Update elixir based on time progression"""
        if self.match_start_time is None:
            return
        
        current_time = time.time()
        
        # More precise regeneration using elapsed time from match start
        elapsed_since_start = current_time - self.match_start_time
        
        # Update current phase based on match time
        self._update_phase(elapsed_since_start)
        
        # Calculate total elixir that should be generated based on current phase
        regen_rate = ELIXIR_PHASES[self.current_phase]
        total_generated = min(elapsed_since_start * regen_rate, MAX_ELIXIR - INITIAL_ELIXIR)
        target_elixir = INITIAL_ELIXIR + total_generated + self.bonus_elixir

        # Update to target elixir (more accurate than incremental updates)
        self.enemy_elixir = min(target_elixir, MAX_ELIXIR)
        
        self.last_update = current_time
        
    def _update_phase(self, match_time):
        """Update the current elixir phase"""
        for phase, (start_time, end_time) in self.phase_times.items():
            if start_time <= match_time < end_time:
                if self.current_phase != phase:
                    self.current_phase = phase
                    print(f"Elixir phase changed to: {phase}")
                break
                
    def add_bonus(self, amount):
        """Grant bonus elixir that persists across subsequent updates."""
        if amount <= 0:
            return
        max_bonus = MAX_ELIXIR - INITIAL_ELIXIR
        self.bonus_elixir = min(max_bonus, self.bonus_elixir + amount)
        if self.match_start_time is None:
            self.enemy_elixir = min(INITIAL_ELIXIR + self.bonus_elixir, MAX_ELIXIR)
        else:
            elapsed_since_start = time.time() - self.match_start_time
            regen_rate = ELIXIR_PHASES[self.current_phase]
            total_generated = min(elapsed_since_start * regen_rate, max_bonus)
            target_elixir = INITIAL_ELIXIR + total_generated + self.bonus_elixir
            self.enemy_elixir = min(target_elixir, MAX_ELIXIR)

    def process_enemy_event(self, event, frame=None):
        """Process an enemy elixir spend event"""
        # Use spell detection if frame is provided
        if frame is not None:
            spell_cost = self.spell_detector.detect_spell_around_stopwatch(frame, event)
        else:
            spell_cost = self._estimate_spell_cost(event)
        
        if spell_cost > 0:
            self.enemy_elixir = max(0, self.enemy_elixir - spell_cost)
            print(f"Enemy spent {spell_cost} elixir, remaining: {self.enemy_elixir:.1f}")
            
            # Add to spell detector history
            self.spell_detector.add_event(event, spell_cost)
            
    def _estimate_spell_cost(self, event):
        """Estimate spell cost from event (MVP implementation)"""
        # For MVP, return default cost
        # In full implementation, this would analyze the visual around the stopwatch
        return DEFAULT_SPELL_COST
    
    def process_card_detection(self, frame):
        """Process automated card detection from game frame using neural embeddings"""
        if not self.is_regenerating():
            return  # Don't process if match hasn't started
            
        # Debug: Print every 60 frames (once per second) but run detection every frame
        if not hasattr(self, '_card_detection_debug_count'):
            self._card_detection_debug_count = 0
        self._card_detection_debug_count += 1
        
        if self._card_detection_debug_count % 60 == 0:
            print(f"🧠 Neural card detection running (frame {self._card_detection_debug_count})")
            
        # Use neural detection instead of template matching
        detected_cards = self.card_detector.detect_cards_from_frame(frame)
        
        # Show detection results with smart tracking info
        for detection in detected_cards:
            slot_num = detection['slot_number']
            card_name = detection['card_name']
            confidence = detection['confidence']
            elixir_cost = detection['elixir_cost']
            
            print(f"🧠 Slot {slot_num}: {card_name} (confidence: {confidence:.1%})")
            print(f"💜 Elixir cost: {elixir_cost} | Method: {detection.get('detection_method', 'neural')}")
            
            # TODO: Later we'll add elixir deduction here
            # old_elixir = self.enemy_elixir
            # self.enemy_elixir = max(0, self.enemy_elixir - elixir_cost)
            # print(f"💜 Enemy elixir: {old_elixir:.1f} → {self.enemy_elixir:.1f} (spent {elixir_cost})")
        
        # Show smart tracking summary every 5 seconds
        if self._card_detection_debug_count % 300 == 0:  # Every 5 seconds at 60fps
            identified = self.card_detector.get_identified_cards()
            if identified:
                print("🎯 Smart Tracking Summary:")
                for slot, info in identified.items():
                    print(f"   Slot {slot}: {info['card_name']} ({info['confidence']:.1%})")
                print("---")
        
        if detected_cards:
            print("---")
        
    def get_enemy_elixir(self):
        """Get current enemy elixir count"""
        return self.enemy_elixir
        
    def get_current_phase(self):
        """Get current elixir phase"""
        return self.current_phase
        
    def get_phase_multiplier(self):
        """Get current phase multiplier for display"""
        phase_multipliers = {
            'single': '1x',
            'double': '2x', 
            'triple': '3x'
        }
        return phase_multipliers.get(self.current_phase, '1x')
        
    def adjust_elixir(self, new_elixir):
        """Adjust elixir value (for timer cross-checking corrections)"""
        old_elixir = self.enemy_elixir
        self.enemy_elixir = max(0, min(new_elixir, MAX_ELIXIR))
        print(f"🔧 Elixir adjusted: {old_elixir:.1f} → {self.enemy_elixir:.1f}")
        
    def reset(self):
        """Reset elixir counter to initial state"""
        self.enemy_elixir = INITIAL_ELIXIR
        self.match_start_time = None
        self.bonus_elixir = 0.0
        self.last_update = time.time()
        self.current_phase = 'single'
