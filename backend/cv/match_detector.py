"""
Match detection to determine when a Clash Royale match is active
"""
import cv2
import numpy as np
from config.settings import TEMPLATES_DIR
from cv.template_manager import TemplateManager

class MatchDetector:
    def __init__(self):
        self.template_manager = TemplateManager()
        self.match_indicators = {
            'vs_badge_detected': False,
            'fight_text_detected': False,
            'ui_elements': False
        }
        
        # Detection parameters
        self.confidence_threshold = 0.45
        self.min_match_elements = 1  # Minimum indicators needed
        
        # Match start timing
        self.vs_detected_time = None
        self.fight_detected_time = None
        
        # Cooldown to prevent spam
        self.last_vs_detection = 0
        self.vs_detection_cooldown = 5.0  # 5 second cooldown between VS detections
        
        # Debug: Check for templates at startup
        self._check_templates()
        
    def detect_match_start(self, frame):
        """Detect specific match start elements (VS badge, FIGHT text)"""
        try:
            # Detect VS badge (primary indicator)
            vs_detected = self._detect_vs_badge(frame)
            
            # FIGHT text is optional - only check if VS was detected for confirmation
            fight_detected = False
            if vs_detected:
                fight_detected = self._detect_fight_text(frame)
            
            # Debug: Print detection attempts every 200 frames (reduced frequency)
            if not hasattr(self, 'detection_count'):
                self.detection_count = 0
            self.detection_count += 1
            if self.detection_count % 200 == 0:
                print(f"Match detection attempt {self.detection_count}: VS={vs_detected}, FIGHT={fight_detected}")
            
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            if vs_detected and self.vs_detected_time is None:
                self.vs_detected_time = current_time
                print("VS badge detected - match starting!")
                
            if fight_detected and self.fight_detected_time is None:
                self.fight_detected_time = current_time
                print("FIGHT text detected - elixir counting begins!")
                
            return {
                'vs_detected': vs_detected,
                'fight_detected': fight_detected,
                'vs_time': self.vs_detected_time,
                'fight_time': self.fight_detected_time
            }
        except Exception as e:
            print(f"ERROR in detect_match_start: {e}")
            import traceback
            traceback.print_exc()
            return {
                'vs_detected': False,
                'fight_detected': False,
                'vs_time': None,
                'fight_time': None
            }
        
    def _detect_vs_badge(self, frame):
        """Detect VS badge in center screen"""
        try:
            vs_dir = TEMPLATES_DIR / "match_start"
            if not vs_dir.exists():
                print(f"VS detection: Template directory not found: {vs_dir}")
                return False

            vs_templates = list(vs_dir.glob("vs_badge*.png"))
            if not vs_templates:
                print("VS detection: No VS badge templates found")
                return False
                
            confidence = self.confidence_threshold
            height, width = frame.shape[:2]
            search_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
            best_match = {'score': 0.0, 'template': None, 'scale': 1.0}
            
            if hasattr(self, 'detection_count') and self.detection_count % 500 == 0:
                print(f"VS detection: Frame {height}x{width}, searching entire frame, checking {len(vs_templates)} templates")
            
            for template_path in vs_templates:
                try:
                    template_color = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template_color is None:
                        continue
                    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
                    
                    if hasattr(self, 'detection_count') and self.detection_count % 100 == 0:
                        print(f"Checking template: {template_path}, size: {template_color.shape}")
                    
                    for scale in scale_factors:
                        scaled_template = cv2.resize(
                            template_gray,
                            (
                                max(1, int(template_gray.shape[1] * scale)),
                                max(1, int(template_gray.shape[0] * scale))
                            ),
                            interpolation=cv2.INTER_AREA
                        )
                        
                        if scaled_template.shape[0] > height or scaled_template.shape[1] > width:
                            continue
                        
                        result = cv2.matchTemplate(search_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > best_match['score']:
                            best_match = {'score': max_val, 'template': template_path.name, 'scale': scale}
                        
                        if max_val >= confidence:
                            current_time = cv2.getTickCount() / cv2.getTickFrequency()
                            if current_time - self.last_vs_detection > self.vs_detection_cooldown:
                                print(
                                    f"VS badge detected using {template_path.name} "
                                    f"(scale {scale:.2f}, confidence: {max_val:.3f})"
                                )
                                self.last_vs_detection = current_time
                                return True
                except Exception as e:
                    print(f"VS template error with {template_path}: {e}")
            
            if best_match['template'] and hasattr(self, 'detection_count') and self.detection_count % 500 == 0:
                print(
                    f"Best VS match so far {best_match['score']:.3f} "
                    f"(template={best_match['template']}, scale={best_match['scale']:.2f})"
                )
            
            return False
        except Exception as e:
            print(f"ERROR in _detect_vs_badge: {e}")
            return False
        
    def _detect_fight_text(self, frame):
        """Detect FIGHT text"""
        fight_dir = TEMPLATES_DIR / "match_start"
        if not fight_dir.exists():
            return False
            
        fight_templates = list(fight_dir.glob("fight_text*.png"))
        
        if not fight_templates:
            return False
            
        # Focus on center region where FIGHT text appears
        height, width = frame.shape[:2]
        center_region = frame[height//3:2*height//3, width//4:3*width//4]
        
        for template_path in fight_templates:
            try:
                # Load template image
                template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                if template is None:
                    continue
                    
                # Try multiple confidence levels
                for confidence in [0.8, 0.7, 0.6]:
                    # Perform template matching
                    result = cv2.matchTemplate(center_region, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val >= confidence:
                        print(f"FIGHT text detected using template: {template_path} (confidence: {max_val:.3f})")
                        return True
            except Exception as e:
                print(f"FIGHT template error with {template_path}: {e}")
                
        return False
        
    def _detect_timer_region(self, frame):
        """Detect match timer/clock in typical location"""
        height, width = frame.shape[:2]
        
        # Timer is typically in the top center
        timer_region = frame[0:height//6, width//3:2*width//3]
        
        # Convert to grayscale
        gray = cv2.cvtColor(timer_region, cv2.COLOR_RGB2GRAY)
        
        # Look for circular/clock-like shapes
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50
        )
        
        if circles is not None:
            return len(circles[0]) > 0
            
        # Alternative: look for digital timer patterns (rectangles with text-like features)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Reasonable timer size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 1.5 < aspect_ratio < 4.0:  # Timer-like aspect ratio
                    return True
                    
        return False
        
    def _detect_ui_elements(self, frame):
        """Detect typical Clash Royale match UI elements"""
        height, width = frame.shape[:2]
        
        # Look for elixir bar region (bottom of screen)
        elixir_region = frame[4*height//5:height, 0:width]
        
        # Convert to HSV to look for purple/pink elixir color
        hsv = cv2.cvtColor(elixir_region, cv2.COLOR_RGB2HSV)
        
        # Purple/pink range for elixir
        purple_lower = np.array([140, 50, 50])
        purple_upper = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        
        # If we find significant purple/pink pixels, likely elixir bar
        purple_pixels = cv2.countNonZero(purple_mask)
        if purple_pixels > elixir_region.shape[0] * elixir_region.shape[1] * 0.01:  # 1% threshold
            return True
            
        # Alternative: look for card slots at bottom
        gray_elixir = cv2.cvtColor(elixir_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_elixir, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular regions that could be card slots
        card_slots = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Card slot size range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.6 < aspect_ratio < 1.4:  # Roughly square card slots
                    card_slots += 1
                    
        return card_slots >= 2  # Need at least 2 card-like shapes
        
    def detect_victory_defeat(self, frame):
        """Detect victory/defeat screen to end match"""
        # This would look for victory/defeat banners
        # For MVP, we'll rely on timeout detection in state machine
        return False
    
    def detect_match(self, frame):
        """Detect if a Clash Royale match is currently active (legacy method)"""
        indicators = 0
        
        # Look for timer/clock elements (simplified detection)
        if self._detect_timer_region(frame):
            indicators += 1
            
        # Look for typical match UI elements
        if self._detect_ui_elements(frame):
            indicators += 1
            
        # Consider match active if we have enough indicators
        return indicators >= self.min_match_elements
        
    def reset_match_start(self):
        """Reset match start detection"""
        self.vs_detected_time = None
        self.fight_detected_time = None
        self.last_vs_detection = 0  # Reset cooldown
        
    def _check_templates(self):
        """Debug: Check if templates exist at startup"""
        vs_dir = TEMPLATES_DIR / "match_start"
        if vs_dir.exists():
            vs_templates = [f.name for f in vs_dir.glob("vs_badge*.png")]
            fight_templates = [f.name for f in vs_dir.glob("fight_text*.png")]
            print(f"Found {len(vs_templates)} VS badge templates: {vs_templates}")
            print(f"Found {len(fight_templates)} FIGHT text templates: {fight_templates}")
        else:
            print(f"Template directory not found: {vs_dir}")
