"""
Match timer detector for cross-checking elixir timing accuracy
"""
import cv2
import numpy as np
import os
from pathlib import Path
import time
from config.settings import TEMPLATES_DIR

class MatchTimerDetector:
    def __init__(self):
        self.template_dir = TEMPLATES_DIR / "match_timer"
        self.timer_templates = []
        self.last_detection_time = 0
        self.match_start_time = None
        self.last_timer_value = None
        
        # Load timer templates
        self._load_templates()
        
    def _load_templates(self):
        """Load match timer templates"""
        self.timer_templates = []
        
        if not self.template_dir.exists():
            print("⏰ No match timer templates directory found")
            return
            
        # Load all timer template images
        for template_file in self.template_dir.glob("*.png"):
            try:
                template = cv2.imread(str(template_file))
                if template is not None:
                    self.timer_templates.append({
                        'image': template,
                        'name': template_file.stem,
                        'path': template_file
                    })
                    print(f"⏰ Loaded timer template: {template_file.name}")
            except Exception as e:
                print(f"⏰ Error loading timer template {template_file}: {e}")
                
        print(f"⏰ Loaded {len(self.timer_templates)} timer templates")
        
    def detect_timer(self, frame):
        """
        Detect match timer in frame
        Returns: dict with timer info or None
        """
        if not self.timer_templates:
            return None
            
        current_time = time.time()
        
        # Convert frame to correct color format
        if len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        best_match = None
        best_confidence = 0.0
        
        # Search for timer templates
        for template_info in self.timer_templates:
            template = template_info['image']
            
            # Multi-scale template matching
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                # Resize template
                if scale != 1.0:
                    template_scaled = cv2.resize(template, None, fx=scale, fy=scale)
                else:
                    template_scaled = template
                    
                if template_scaled.shape[0] > frame_bgr.shape[0] or template_scaled.shape[1] > frame_bgr.shape[1]:
                    continue
                    
                # Template matching
                result = cv2.matchTemplate(frame_bgr, template_scaled, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = {
                        'template_name': template_info['name'],
                        'confidence': max_val,
                        'location': max_loc,
                        'scale': scale,
                        'template_size': template_scaled.shape[:2],
                        'detection_time': current_time
                    }
        
        # Only return if confidence is high enough
        if best_match and best_confidence > 0.6:  # Threshold for timer detection
            self.last_detection_time = current_time
            self.last_timer_value = best_match
            
            # Extract timer value from template name if possible
            timer_seconds = self._extract_timer_seconds(best_match['template_name'])
            if timer_seconds is not None:
                best_match['timer_seconds'] = timer_seconds
                
            return best_match
            
        return None
        
    def _extract_timer_seconds(self, template_name):
        """
        Extract timer value in seconds from template name
        Expected format: timer_2m30s or timer_150s
        """
        try:
            # Handle formats like "timer_2m30s"
            if 'm' in template_name and 's' in template_name:
                time_part = template_name.split('_')[-1]  # Get "2m30s"
                minutes_part, seconds_part = time_part.split('m')
                minutes = int(minutes_part)
                seconds = int(seconds_part.replace('s', ''))
                return minutes * 60 + seconds
                
            # Handle formats like "timer_150s"
            elif 's' in template_name:
                time_part = template_name.split('_')[-1]  # Get "150s"
                seconds = int(time_part.replace('s', ''))
                return seconds
                
        except (ValueError, IndexError):
            pass
            
        return None
        
    def start_timer_tracking(self, initial_timer_seconds=None):
        """Start tracking match timer"""
        self.match_start_time = time.time()
        if initial_timer_seconds:
            # Backdate start time based on detected timer value
            # If we detect 2m30s, match started 30s ago (assuming 3min match)
            estimated_elapsed = 180 - initial_timer_seconds  # 3min = 180s
            self.match_start_time -= estimated_elapsed
            
        print(f"⏰ Match timer tracking started (backdated by {estimated_elapsed if initial_timer_seconds else 0}s)")
        
    def get_estimated_match_time(self):
        """Get estimated match time based on our tracking"""
        if self.match_start_time is None:
            return None
            
        elapsed = time.time() - self.match_start_time
        return max(0, 180 - elapsed)  # 3min countdown
        
    def cross_check_elixir_timing(self, elixir_start_time, current_elixir):
        """
        Cross-check elixir timing against match timer
        Returns suggested elixir adjustment
        """
        if self.match_start_time is None or elixir_start_time is None:
            return None
            
        # Calculate how much time has passed since elixir started
        elixir_elapsed = time.time() - elixir_start_time
        
        # Calculate how much time has passed since match started  
        match_elapsed = time.time() - self.match_start_time
        
        # In Clash Royale, elixir starts generating right when match starts
        # So they should be very close
        time_difference = abs(elixir_elapsed - match_elapsed)
        
        if time_difference > 1.0:  # More than 1 second difference
            return {
                'time_difference': time_difference,
                'suggested_adjustment': match_elapsed - elixir_elapsed,
                'match_elapsed': match_elapsed,
                'elixir_elapsed': elixir_elapsed
            }
            
        return None
