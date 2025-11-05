"""
Stopwatch detection using template matching and HSV color verification
"""
import cv2
import numpy as np
import time
from config.constants import RED_HSV_RANGES, BLUE_HSV_RANGE
from cv.template_manager import TemplateManager

class StopwatchDetector:
    def __init__(self):
        self.last_detections = []
        self.detection_history = []
        
        # Template manager for improved detection
        self.template_manager = TemplateManager()
        
        # Template matching parameters
        self.template_threshold = 0.7
        self.min_stopwatch_size = 20
        self.max_stopwatch_size = 100
        
        # Check if we have templates available
        self.has_red_templates = self.template_manager.has_templates('red_stopwatch')
        self.has_blue_templates = self.template_manager.has_templates('blue_stopwatch')
        self.has_gold_templates = self.template_manager.has_templates('gold_bezel')
        
    def detect_stopwatches(self, frame):
        """Detect red and blue stopwatches in frame"""
        events = []
        current_time = time.time()
        
        # Try template matching first (more accurate)
        template_events = self._detect_with_templates(frame)
        events.extend(template_events)
        
        # If no templates or low confidence, fall back to HSV detection
        if not template_events or max([e.get('confidence', 0) for e in template_events], default=0) < 0.8:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Detect red stopwatches
            red_events = self._detect_colored_stopwatches(frame, hsv, 'red')
            events.extend(red_events)
            
            # Detect blue stopwatches  
            blue_events = self._detect_colored_stopwatches(frame, hsv, 'blue')
            events.extend(blue_events)
        
        # Filter out duplicate/overlapping detections
        events = self._filter_overlapping_events(events)
        
        # Add timestamps
        for event in events:
            event['timestamp'] = current_time
            
        self.last_detections = events
        return events
        
    def _detect_with_templates(self, frame):
        """Detect stopwatches using template matching"""
        events = []
        
        # Detect red stopwatches with templates
        if self.has_red_templates:
            red_matches = self.template_manager.match_template(frame, 'red_stopwatch', self.template_threshold)
            for match in red_matches:
                event = {
                    'color': 'red',
                    'x': match['x'] + match['width'] // 2,
                    'y': match['y'] + match['height'] // 2,
                    'width': match['width'],
                    'height': match['height'],
                    'confidence': match['confidence'],
                    'detection_method': 'template'
                }
                events.append(event)
                
        # Detect blue stopwatches with templates
        if self.has_blue_templates:
            blue_matches = self.template_manager.match_template(frame, 'blue_stopwatch', self.template_threshold)
            for match in blue_matches:
                event = {
                    'color': 'blue',
                    'x': match['x'] + match['width'] // 2,
                    'y': match['y'] + match['height'] // 2,
                    'width': match['width'],
                    'height': match['height'],
                    'confidence': match['confidence'],
                    'detection_method': 'template'
                }
                events.append(event)
                
        # Also try gold bezel detection (works for both colors)
        if self.has_gold_templates:
            gold_matches = self.template_manager.match_template(frame, 'gold_bezel', self.template_threshold)
            for match in gold_matches:
                # Determine color by checking the center area
                center_x = match['x'] + match['width'] // 2
                center_y = match['y'] + match['height'] // 2
                color = self._determine_stopwatch_color(frame, center_x, center_y)
                
                if color:
                    event = {
                        'color': color,
                        'x': center_x,
                        'y': center_y,
                        'width': match['width'],
                        'height': match['height'],
                        'confidence': match['confidence'] * 0.9,  # Slightly lower confidence for gold bezel
                        'detection_method': 'gold_bezel'
                    }
                    events.append(event)
                    
        return events
        
    def _determine_stopwatch_color(self, frame, x, y):
        """Determine stopwatch color by checking center pixel"""
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            # Extract small region around center
            region_size = 10
            y1 = max(0, y - region_size)
            y2 = min(frame.shape[0], y + region_size)
            x1 = max(0, x - region_size)
            x2 = min(frame.shape[1], x + region_size)
            
            region = frame[y1:y2, x1:x2]
            if region.size > 0:
                hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                
                # Check for red
                red_mask1 = cv2.inRange(hsv_region, RED_HSV_RANGES[0][0], RED_HSV_RANGES[0][1])
                red_mask2 = cv2.inRange(hsv_region, RED_HSV_RANGES[1][0], RED_HSV_RANGES[1][1])
                red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
                
                # Check for blue
                blue_mask = cv2.inRange(hsv_region, BLUE_HSV_RANGE[0], BLUE_HSV_RANGE[1])
                blue_pixels = cv2.countNonZero(blue_mask)
                
                if red_pixels > blue_pixels and red_pixels > 5:
                    return 'red'
                elif blue_pixels > 5:
                    return 'blue'
                    
        return None
        
    def _detect_colored_stopwatches(self, frame, hsv, color):
        """Detect stopwatches of a specific color"""
        events = []
        
        # Create color mask
        if color == 'red':
            # Red has two ranges in HSV
            mask1 = cv2.inRange(hsv, RED_HSV_RANGES[0][0], RED_HSV_RANGES[0][1])
            mask2 = cv2.inRange(hsv, RED_HSV_RANGES[1][0], RED_HSV_RANGES[1][1])
            mask = cv2.bitwise_or(mask1, mask2)
        elif color == 'blue':
            mask = cv2.inRange(hsv, BLUE_HSV_RANGE[0], BLUE_HSV_RANGE[1])
        else:
            return events
            
        # Find contours in color mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by size
            area = cv2.contourArea(contour)
            if area < self.min_stopwatch_size * self.min_stopwatch_size:
                continue
            if area > self.max_stopwatch_size * self.max_stopwatch_size:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (stopwatches should be roughly circular)
            aspect_ratio = w / h
            if not (0.7 <= aspect_ratio <= 1.3):
                continue
                
            # Additional validation: check for circular shape
            if self._validate_stopwatch_shape(contour, x, y, w, h):
                event = {
                    'color': color,
                    'x': x + w // 2,
                    'y': y + h // 2,
                    'width': w,
                    'height': h,
                    'confidence': self._calculate_confidence(contour, area),
                    'detection_method': 'hsv_color'
                }
                events.append(event)
                
        return events
        
    def _validate_stopwatch_shape(self, contour, x, y, w, h):
        """Validate that the contour looks like a stopwatch"""
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
        
        # Stopwatches should be reasonably circular
        return circularity > 0.3
        
    def _calculate_confidence(self, contour, area):
        """Calculate confidence score for detection"""
        # Simple confidence based on area and shape
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
            
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Confidence increases with circularity
        confidence = min(1.0, circularity * 2)
        return confidence
        
    def _filter_overlapping_events(self, events):
        """Remove overlapping detections"""
        if len(events) <= 1:
            return events
            
        filtered = []
        for event in events:
            is_duplicate = False
            
            for existing in filtered:
                # Calculate distance between centers
                dx = event['x'] - existing['x']
                dy = event['y'] - existing['y']
                distance = np.sqrt(dx * dx + dy * dy)
                
                # If too close, consider it a duplicate
                overlap_threshold = max(event['width'], event['height']) / 2
                if distance < overlap_threshold:
                    # Keep the one with higher confidence
                    if event['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
                        
            if not is_duplicate:
                filtered.append(event)
                
        return filtered
        
    def get_detection_history(self, time_window=5.0):
        """Get recent detection history"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        return [d for d in self.detection_history if d['timestamp'] > cutoff_time]
