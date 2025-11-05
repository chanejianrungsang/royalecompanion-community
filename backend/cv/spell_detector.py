"""
Spell detection using heuristics around stopwatch events
"""
import cv2
import numpy as np
import time
from config.constants import SPELL_COSTS, DEFAULT_SPELL_COST, STOPWATCH_DETECTION_WINDOW

class SpellDetector:
    def __init__(self):
        self.spell_patterns = {}
        self.recent_events = []
        
        # Initialize basic spell patterns (MVP implementation)
        self._init_spell_patterns()
        
    def _init_spell_patterns(self):
        """Initialize basic patterns for common spells"""
        # For MVP, we'll use simple heuristics based on visual effects
        self.spell_patterns = {
            'fireball': {
                'cost': 4,
                'color_signature': 'orange_explosion',
                'shape': 'circular_blast'
            },
            'arrows': {
                'cost': 3,
                'color_signature': 'white_streaks',
                'shape': 'linear_volleys'
            },
            'zap': {
                'cost': 2,
                'color_signature': 'blue_lightning',
                'shape': 'fork_pattern'
            },
            'the_log': {
                'cost': 2,
                'color_signature': 'brown_cylinder',
                'shape': 'rolling_horizontal'
            },
            'rocket': {
                'cost': 6,
                'color_signature': 'red_trail',
                'shape': 'vertical_missile'
            }
        }
        
    def detect_spell_around_stopwatch(self, frame, stopwatch_event):
        """Detect spell type around a red stopwatch event"""
        if stopwatch_event['color'] != 'red':
            return DEFAULT_SPELL_COST
            
        # Extract region around stopwatch
        x, y = stopwatch_event['x'], stopwatch_event['y']
        search_radius = 150  # pixels
        
        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(frame.shape[1], x + search_radius)
        y2 = min(frame.shape[0], y + search_radius)
        
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return DEFAULT_SPELL_COST
            
        # Try to identify spell type
        spell_type = self._analyze_spell_effects(region)
        
        if spell_type in SPELL_COSTS:
            return SPELL_COSTS[spell_type]
        else:
            return DEFAULT_SPELL_COST
            
    def _analyze_spell_effects(self, region):
        """Analyze visual effects to identify spell type (MVP)"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Simple heuristics for common spells
        
        # Check for fireball (orange explosion)
        if self._detect_fireball_pattern(region, hsv):
            return 'fireball'
            
        # Check for arrows (white streaks)
        if self._detect_arrows_pattern(region, gray):
            return 'arrows'
            
        # Check for zap (blue lightning)
        if self._detect_zap_pattern(region, hsv):
            return 'zap'
            
        # Check for log (brown rolling cylinder)
        if self._detect_log_pattern(region, hsv):
            return 'the_log'
            
        # Check for rocket (red trail)
        if self._detect_rocket_pattern(region, hsv):
            return 'rocket'
            
        return 'unknown'
        
    def _detect_fireball_pattern(self, region, hsv):
        """Detect fireball explosion pattern"""
        # Look for orange/red circular explosion
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Find circular contours
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Significant explosion size
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:  # Reasonably circular
                        return True
        return False
        
    def _detect_arrows_pattern(self, region, gray):
        """Detect arrows volley pattern"""
        # Look for linear white streaks
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=5)
        
        if lines is not None and len(lines) >= 3:  # Multiple arrow streaks
            return True
        return False
        
    def _detect_zap_pattern(self, region, hsv):
        """Detect zap lightning pattern"""
        # Look for bright blue/white zigzag patterns
        blue_lower = np.array([100, 50, 200])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Look for branching/jagged patterns
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Check for jagged/irregular shape (lightning)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) > 6:  # Many vertices = jagged
                return True
        return False
        
    def _detect_log_pattern(self, region, hsv):
        """Detect log rolling pattern"""
        # Look for brown horizontal cylinder
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([20, 200, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if aspect_ratio > 2.0:  # Horizontal cylinder shape
                return True
        return False
        
    def _detect_rocket_pattern(self, region, hsv):
        """Detect rocket trail pattern"""
        # Look for red/orange vertical trail
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w
            
            if aspect_ratio > 2.0:  # Vertical trail shape
                return True
        return False
        
    def add_event(self, event, spell_cost):
        """Add a detected spell event to history"""
        event_data = {
            'timestamp': time.time(),
            'position': (event['x'], event['y']),
            'cost': spell_cost,
            'confidence': event.get('confidence', 1.0)
        }
        
        self.recent_events.append(event_data)
        
        # Keep only recent events
        current_time = time.time()
        self.recent_events = [
            e for e in self.recent_events 
            if current_time - e['timestamp'] < 10.0  # 10 second history
        ]
