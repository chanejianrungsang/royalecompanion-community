"""
Simple timer validation using template matching for specific timer values
"""
import cv2
import time
from pathlib import Path
from config.settings import TEMPLATES_DIR

class SimpleTimerValidator:
    def __init__(self):
        self.template_dir = TEMPLATES_DIR / "match_timer"
        self.timer_templates = []
        self.last_check_time = 0
        
        # Load timer templates with known values
        self._load_timer_templates()
        
    def _load_timer_templates(self):
        """Load timer templates with known time values"""
        self.timer_templates = []
        
        if not self.template_dir.exists():
            print("⏰ No timer templates directory found")
            return
            
        for template_file in self.template_dir.glob("*.png"):
            try:
                template = cv2.imread(str(template_file))
                if template is not None:
                    # Extract timer value from filename if possible
                    timer_seconds = self._extract_time_from_filename(template_file.stem)
                    
                    self.timer_templates.append({
                        'image': template,
                        'name': template_file.stem,
                        'timer_seconds': timer_seconds,
                        'path': template_file
                    })
                    print(f"⏰ Loaded timer template: {template_file.name} ({timer_seconds}s)")
            except Exception as e:
                print(f"⏰ Error loading timer template {template_file}: {e}")
                
        print(f"⏰ Loaded {len(self.timer_templates)} timer templates")
        
    def _extract_time_from_filename(self, filename):
        """
        Extract timer seconds from filename
        Examples: timer_2m30s -> 150, timer_156s -> 156
        """
        import re
        
        # Handle formats like "timer_2m30s"
        match = re.search(r'(\d+)m(\d+)s', filename)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
            
        # Handle formats like "timer_150s"
        match = re.search(r'(\d+)s', filename)
        if match:
            return int(match.group(1))
            
        # Default fallback
        return None
        
    def validate_elixir_timing(self, frame, current_elixir, elixir_start_time):
        """
        Validate elixir timing by checking for specific timer templates
        Returns validation result or None
        """
        current_time = time.time()
        
        # Only check every 10 seconds to avoid spam
        if current_time - self.last_check_time < 10.0:
            return None
            
        if not self.timer_templates or elixir_start_time is None:
            return None
            
        # Convert frame to BGR for template matching
        if len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        best_match = None
        best_confidence = 0.0
        
        # Try to match timer templates
        for template_info in self.timer_templates:
            if template_info['timer_seconds'] is None:
                continue
                
            template = template_info['image']
            
            # Multi-scale template matching
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                if scale != 1.0:
                    template_scaled = cv2.resize(template, None, fx=scale, fy=scale)
                else:
                    template_scaled = template
                    
                if template_scaled.shape[0] > frame_bgr.shape[0] or template_scaled.shape[1] > frame_bgr.shape[1]:
                    continue
                    
                result = cv2.matchTemplate(frame_bgr, template_scaled, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = template_info.copy()
                    best_match['confidence'] = max_val
                    
        # Validate if we found a good match
        if best_match and best_confidence > 0.7:  # High confidence threshold
            timer_seconds = best_match['timer_seconds']
            
            # Calculate expected elixir based on timer
            match_elapsed = 180 - timer_seconds  # 3min = 180s countdown
            expected_elixir = 5.0 + (match_elapsed * (1/2.8))
            expected_elixir = min(expected_elixir, 10.0)
            
            # Calculate actual elixir elapsed time
            elixir_elapsed = current_time - elixir_start_time
            
            validation_result = {
                'timer_detected': timer_seconds,
                'match_elapsed': match_elapsed,
                'elixir_elapsed': elixir_elapsed,
                'expected_elixir': expected_elixir,
                'actual_elixir': current_elixir,
                'elixir_difference': current_elixir - expected_elixir,
                'time_difference': elixir_elapsed - match_elapsed,
                'confidence': best_confidence,
                'template_name': best_match['name']
            }
            
            print(f"⏰ Timer validation: {timer_seconds}s detected (conf: {best_confidence:.2f})")
            print(f"   Expected elixir: {expected_elixir:.1f}, Actual: {current_elixir:.1f}")
            print(f"   Difference: {validation_result['elixir_difference']:+.1f} elixir")
            
            self.last_check_time = current_time
            return validation_result
            
        self.last_check_time = current_time
        return None
