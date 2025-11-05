"""
Loading screen detector for detecting match start at 100% loading
"""
import cv2
import numpy as np
from pathlib import Path
import time
from config.settings import TEMPLATES_DIR

class LoadingScreenDetector:
    def __init__(self):
        self.template_dir = TEMPLATES_DIR / "loading_screen"
        self.loading_templates = []
        self.last_detection_time = 0
        self.confidence_threshold = 0.9  # High threshold for precise 100% detection
        self.detection_cooldown = 10.0  # 10 second cooldown between detections
        
        # Load loading screen templates
        self._load_templates()
        
    def _load_templates(self):
        """Load loading screen 100% templates"""
        self.loading_templates = []
        
        if not self.template_dir.exists():
            print("📱 No loading screen templates directory found")
            return
            
        # Load all loading screen template images
        for template_file in self.template_dir.glob("*.png"):
            try:
                template = cv2.imread(str(template_file))
                if template is not None:
                    self.loading_templates.append({
                        'image': template,
                        'name': template_file.stem,
                        'path': template_file
                    })
                    print(f"📱 Loaded loading screen template: {template_file.name}")
            except Exception as e:
                print(f"📱 Error loading loading screen template {template_file}: {e}")
                
        print(f"📱 Loaded {len(self.loading_templates)} loading screen templates")
        
    def detect_loading_complete(self, frame):
        """
        Detect if loading screen shows 100% complete
        Returns: detection info dict or None
        """
        if not self.loading_templates:
            return None
            
        current_time = time.time()
        
        # Enforce cooldown to prevent duplicate detections
        if self.last_detection_time > 0 and (current_time - self.last_detection_time) < self.detection_cooldown:
            return None
        
        # Convert frame to correct color format for OpenCV
        if len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        best_match = None
        best_confidence = 0.0
        
        # Search for loading screen templates
        for template_info in self.loading_templates:
            template = template_info['image']
            
            # Multi-scale template matching for better detection
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                # Resize template if needed
                if scale != 1.0:
                    template_scaled = cv2.resize(template, None, fx=scale, fy=scale)
                else:
                    template_scaled = template
                    
                # Skip if template is larger than frame
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
        
        # Only return if confidence is high enough for 100% detection
        if best_match and best_confidence > self.confidence_threshold:
            self.last_detection_time = current_time
            print(f"📱 Loading 100% detected! (confidence: {best_confidence:.3f})")
            return best_match
        elif best_match and best_confidence > 0.6:
            # Log partial matches for debugging
            print(f"📱 Loading partial match: {best_confidence:.3f} (threshold: {self.confidence_threshold})")
            
        return None
        
    def is_recently_detected(self, time_window=2.0):
        """Check if loading screen was detected recently"""
        if self.last_detection_time == 0:
            return False
        return time.time() - self.last_detection_time < time_window
        
    def reset_detection(self):
        """Reset detection state - used when entering idle state"""
        self.last_detection_time = 0
