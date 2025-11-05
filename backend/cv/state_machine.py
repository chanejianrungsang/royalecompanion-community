"""
State machine for tracking match state and elixir
"""
import time
from enum import Enum
from cv.elixir_counter import ElixirCounter
from cv.loading_screen_detector import LoadingScreenDetector

class GameState(Enum):
    IDLE = "idle"
    MATCH = "match"

class StateMachine:
    def __init__(self):
        self.state = GameState.IDLE
        self.last_state_change = time.time()
        
        # Initialize detectors - simplified to only use loading screen
        self.elixir_counter = ElixirCounter()
        self.loading_detector = LoadingScreenDetector()
        
        # State tracking
        self.confidence = 1.0
        self.last_activity = time.time()
        
        # Prevent duplicate triggers (only need loading screen now)
        self.loading_screen_triggered = False
        
    def process_frame(self, frame):
        """Process a single frame through the state machine"""
        current_time = time.time()
        
        # Debug: Print frame processing every 500 frames (reduced frequency)
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        if self.frame_count % 500 == 0:
            elixir_status = f"elixir regenerating: {self.elixir_counter.is_regenerating()}" if hasattr(self, 'elixir_counter') else "no elixir counter"
            print(f"🔄 Frame {self.frame_count}, state: {self.state.value}, {elixir_status}")
        
        # Priority 1: Check for loading screen 100% (only detection method)
        loading_detected = self.loading_detector.detect_loading_complete(frame)
        
        # Simplified: Only use loading screen for match detection and elixir start
        if self.state == GameState.IDLE:
            # Enter match state only on loading screen detection
            if loading_detected:
                self._enter_match_state()
            
            # FALLBACK: If we detect any stopwatch while IDLE, we're probably in a match!
            # This handles cases where user starts app mid-match
            elif hasattr(self, 'elixir_counter') and self.frame_count % 60 == 0:  # Check every second
                # Quick stopwatch check to see if we're in a match
                stopwatch_locations = self.elixir_counter.card_detector._find_stopwatches(frame)
                if stopwatch_locations:
                    print(f"🎮 Stopwatch detected while IDLE - entering MATCH state!")
                    self._enter_match_state()
                
        # Start elixir regeneration ONLY from loading screen detection
        if loading_detected and not self.loading_screen_triggered and not self.elixir_counter.is_regenerating():
            # Loading screen detected - use 3.15s backdate as requested
            self.elixir_counter.start_regeneration(backdate_seconds=3.15)
            self.loading_screen_triggered = True
            print("📱 Loading 100% detected - elixir regeneration started (backdated 3.15s)!")
                    
        if self.state == GameState.MATCH:
            # Always process match events when in MATCH state 
            # (elixir regeneration should continue regardless of UI detection)
            self._process_match_frame(frame)
            
            # Debug: Print that we're processing match frames
            if hasattr(self, 'frame_count') and self.frame_count % 300 == 0:  # Every 5 seconds at 60fps
                print(f"🎮 Processing match frame {self.frame_count} - elixir regenerating: {self.elixir_counter.is_regenerating()}")
            
            # Simple timeout check - exit match state after 5 minutes of no loading screen
            if current_time - self.last_activity > 300.0:  # 5 minute timeout
                self._enter_idle_state()
                
    def _enter_match_state(self):
        """Enter MATCH state"""
        self.state = GameState.MATCH
        self.last_state_change = time.time()
        self.last_activity = time.time()
        
        # Initialize elixir counter but don't reset if regeneration has already started
        # This prevents killing regeneration that was started by loading screen detection
        if not self.elixir_counter.is_regenerating():
            self.elixir_counter.reset()
            print("Entered MATCH state - waiting for VS/FIGHT detection to start tracking")
        else:
            print("Entered MATCH state - elixir regeneration already active, continuing")
        
    def _enter_idle_state(self):
        """Enter IDLE state"""
        self.state = GameState.IDLE
        self.last_state_change = time.time()
        
        # Reset elixir counter and loading detection when leaving match
        self.elixir_counter.reset()
        self.loading_detector.reset_detection()  # Reset loading screen cooldown
        
        # Reset trigger flag for next match
        self.loading_screen_triggered = False
        
        print("Entered IDLE state - tracking stopped")
        
    def _process_match_frame(self, frame):
        """Process frame during match state - simplified for loading screen only"""
        # Update elixir regeneration
        self.elixir_counter.update()
        
        # ALWAYS process card detection when in MATCH state
        # Players can place troops at any time, not just when elixir is regenerating!
        self.elixir_counter.process_card_detection(frame)
                
    def _validate_timer(self, frame):
        """Validate elixir timing using timer template matching"""
        try:
            current_elixir = self.elixir_counter.get_enemy_elixir()
            elixir_start_time = self.elixir_counter.match_start_time
            
            validation = self.timer_validator.validate_elixir_timing(
                frame, current_elixir, elixir_start_time
            )
            
            if validation and abs(validation['elixir_difference']) > 1.0:
                print(f"⚠️  Timer validation: Elixir {validation['elixir_difference']:+.1f} off expected")
                
                # Optionally adjust elixir counter for large discrepancies
                if abs(validation['elixir_difference']) > 2.0:
                    self.elixir_counter.adjust_elixir(validation['expected_elixir'])
                    print(f"🔧 Adjusted elixir to {validation['expected_elixir']:.1f}")
                    
            self._last_timer_check = time.time()
            
        except Exception as e:
            print(f"⚠️ Timer validation error: {e}")
            self._last_timer_check = time.time()
        
    def get_enemy_elixir(self):
        """Get current enemy elixir count"""
        if self.state == GameState.MATCH:
            return self.elixir_counter.get_enemy_elixir()
        return 5.0  # Always return 5.0 when not in match (strict requirement)
        
    def get_confidence(self):
        """Get current detection confidence"""
        return self.confidence
        
    def get_state(self):
        """Get current state"""
        return self.state
        
    def reset(self):
        """Reset the state machine"""
        self.state = GameState.IDLE
        self.elixir_counter.reset()
        self.confidence = 1.0
