"""
Template-based detection utilities
"""
import cv2
import numpy as np
import os
from pathlib import Path
from config.settings import TEMPLATES_DIR

class TemplateManager:
    def __init__(self):
        self.templates = {}
        self.template_dir = TEMPLATES_DIR
        self.load_templates()
        
    def load_templates(self):
        """Load all template images"""
        if not self.template_dir.exists():
            print(f"Template directory not found: {self.template_dir}")
            return
            
        # Load stopwatch templates
        self.load_stopwatch_templates()
        
        # Load UI element templates
        self.load_ui_templates()
        
        # Load spell effect templates
        self.load_spell_templates()
        
    def load_stopwatch_templates(self):
        """Load red and blue stopwatch templates"""
        stopwatch_files = {
            'red_stopwatch': ['red_stopwatch_1.png', 'red_stopwatch_2.png', 'red_stopwatch_3.png'],
            'blue_stopwatch': ['blue_stopwatch_1.png', 'blue_stopwatch_2.png', 'blue_stopwatch_3.png'],
            'gold_bezel': ['gold_bezel_1.png', 'gold_bezel_2.png']  # The gold ring around stopwatches
        }
        
        for template_type, filenames in stopwatch_files.items():
            self.templates[template_type] = []
            for filename in filenames:
                template_path = self.template_dir / "stopwatches" / filename
                if template_path.exists():
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        # Store multiple scales
                        self.templates[template_type].extend(self._create_scaled_templates(template))
                        
    def load_ui_templates(self):
        """Load UI element templates for match detection"""
        ui_files = {
            'match_timer': ['timer_digital.png', 'timer_analog.png'],
            'elixir_bar': ['elixir_empty.png', 'elixir_full.png'],
            'card_slot': ['card_slot_empty.png', 'card_slot_ready.png'],
            'victory_banner': ['victory.png'],
            'defeat_banner': ['defeat.png']
        }
        
        for template_type, filenames in ui_files.items():
            self.templates[template_type] = []
            for filename in filenames:
                template_path = self.template_dir / "ui" / filename
                if template_path.exists():
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        self.templates[template_type].extend(self._create_scaled_templates(template))
                        
    def load_spell_templates(self):
        """Load spell effect templates"""
        spell_files = {
            'fireball_explosion': ['fireball_1.png', 'fireball_2.png'],
            'arrows_volley': ['arrows_1.png', 'arrows_2.png'],
            'zap_lightning': ['zap_1.png', 'zap_2.png'],
            'log_cylinder': ['log_1.png', 'log_2.png'],
            'rocket_trail': ['rocket_1.png', 'rocket_2.png'],
            'poison_cloud': ['poison_1.png', 'poison_2.png'],
            'freeze_effect': ['freeze_1.png', 'freeze_2.png']
        }
        
        for template_type, filenames in spell_files.items():
            self.templates[template_type] = []
            for filename in filenames:
                template_path = self.template_dir / "spells" / filename
                if template_path.exists():
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        self.templates[template_type].extend(self._create_scaled_templates(template))
                        
    def _create_scaled_templates(self, template, scales=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]):
        """Create multiple scaled versions of a template"""
        scaled_templates = []
        h, w = template.shape[:2]
        
        for scale in scales:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w > 10 and new_h > 10:  # Minimum size check
                scaled = cv2.resize(template, (new_w, new_h))
                scaled_templates.append({
                    'template': scaled,
                    'scale': scale,
                    'size': (new_w, new_h)
                })
                
        return scaled_templates
        
    def match_template(self, image, template_type, threshold=0.7):
        """Match templates against image"""
        if template_type not in self.templates:
            return []
            
        matches = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        for template_data in self.templates[template_type]:
            template = template_data['template']
            gray_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY) if len(template.shape) == 3 else template
            
            # Template matching
            result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                match = {
                    'x': pt[0],
                    'y': pt[1],
                    'width': template_data['size'][0],
                    'height': template_data['size'][1],
                    'confidence': result[pt[1], pt[0]],
                    'scale': template_data['scale'],
                    'template_type': template_type
                }
                matches.append(match)
                
        # Remove overlapping matches
        return self._filter_overlapping_matches(matches)
        
    def _filter_overlapping_matches(self, matches, overlap_threshold=0.5):
        """Remove overlapping template matches"""
        if len(matches) <= 1:
            return matches
            
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for match in matches:
            is_overlapping = False
            
            for existing in filtered:
                # Calculate overlap
                x1 = max(match['x'], existing['x'])
                y1 = max(match['y'], existing['y'])
                x2 = min(match['x'] + match['width'], existing['x'] + existing['width'])
                y2 = min(match['y'] + match['height'], existing['y'] + existing['height'])
                
                if x2 > x1 and y2 > y1:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    match_area = match['width'] * match['height']
                    overlap_ratio = overlap_area / match_area
                    
                    if overlap_ratio > overlap_threshold:
                        is_overlapping = True
                        break
                        
            if not is_overlapping:
                filtered.append(match)
                
        return filtered
        
    def get_available_templates(self):
        """Get list of available template types"""
        return list(self.templates.keys())
        
    def has_templates(self, template_type):
        """Check if templates are available for a type"""
        return template_type in self.templates and len(self.templates[template_type]) > 0
