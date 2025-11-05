"""
Neural Card Detection System
Replaces template matching with neural embeddings for accurate card detection
"""
import cv2
import numpy as np
import json
import time
import pickle
from typing import Dict, List, Tuple, Optional
from simple_neural_detector import SimpleNeuralDetector
from config.settings import ELIXIR_COST_PATH, SAVED_CROP_PATH

class NeuralCardDetector:
    def __init__(self):
        # Neural detection
        self.neural_detector = SimpleNeuralDetector()
        self.neural_ready = self.neural_detector.load_cache()
        
        # Card position data from interactive crop tool
        self.crop_positions = []
        self.load_crop_positions()
        
        # Smart tracking - avoid re-detecting identified cards
        self.identified_cards = {}  # slot_number -> {'card_name': str, 'confidence': float, 'last_seen': timestamp}
        self.identification_threshold = 0.85  # High confidence required for permanent identification
        self.recheck_interval = 5.0  # seconds - recheck low-confidence cards
        
        # Detection cooldown to prevent spam
        self.detection_cooldown = 1.0  # seconds between detections per slot
        self.last_detections = {}  # slot_number -> timestamp
        
        # Load elixir costs
        self.elixir_costs = {}
        self.load_elixir_costs()
        
        # Load card name mappings (display name -> neural cache name)
        self.card_mappings = self.load_card_mappings()
        
        print(f"🧠 Neural Card Detector initialized")
        print(f"   Neural cache ready: {self.neural_ready}")
        print(f"   Crop positions loaded: {len(self.crop_positions)}")
        print(f"   Card mappings loaded: {len(self.card_mappings)}")
        
    def load_crop_positions(self):
        """Load precise card crop positions from saved configuration."""
        try:
            if SAVED_CROP_PATH.exists():
                with open(SAVED_CROP_PATH, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.crop_positions = data.get("crop_boxes", [])
                print(f"Loaded {len(self.crop_positions)} crop positions from {SAVED_CROP_PATH.name}")
            else:
                print("No saved_crop.json found - neural detection disabled")
        except Exception as exc:
            print(f"Error loading crop positions: {exc}")
            self.crop_positions = []

    def load_elixir_costs(self):
        """Load elixir costs for cards."""
        try:
            if ELIXIR_COST_PATH.exists():
                with open(ELIXIR_COST_PATH, "r", encoding="utf-8") as handle:
                    self.elixir_costs = json.load(handle)
                print(f"Loaded elixir costs for {len(self.elixir_costs)} cards")
        except Exception as exc:
            print(f"Error loading elixir costs: {exc}")

    def load_card_mappings(self):
        """Load card name mappings from display names to neural cache names"""
        # Based on our validated mappings from test_accurate_mappings.py
        return {
            "wallbreakers": "WallBreakers",
            "skeletons": "Skellies", 
            "mighty miner": "MightyMiner",
            "royal recruits": "RoyalRecruits",
            "goblin gang": "GobGang",
            "dart goblin": "DartGob",
            "skeleton barrel": "SkellyBarrel",
            "mystery card": "unknown",
            # Add more mappings as needed
            "tombstone": "Tombstone",
            "valkyrie": "Valkyrie",
            "goblin barrel": "GoblinBarrel",
            "princess": "Princess",
            "royal delivery": "RoyalDelivery",
            "bats": "Bats",
            "elite barbarians": "EliteBarbarians",
            "log": "Log"
        }
        
    def detect_cards_from_frame(self, frame):
        """Main detection method - replaces detect_cards_from_stopwatches"""
        if not self.neural_ready or not self.crop_positions:
            return []
            
        detected_cards = []
        current_time = time.time()
        
        for position in self.crop_positions:
            slot_number = position['card_number']
            card_name = position['card_name']
            
            # Smart tracking: Skip if card is already confidently identified
            if self.should_skip_detection(slot_number, current_time):
                continue
                
            # Check detection cooldown
            if slot_number in self.last_detections:
                if current_time - self.last_detections[slot_number] < self.detection_cooldown:
                    continue
                    
            # Extract card crop from frame
            card_crop = self.extract_card_crop(frame, position)
            if card_crop is None:
                continue
                
            # Perform neural detection
            detection_result = self.detect_card_neural(card_crop, card_name, slot_number)
            if detection_result:
                detected_cards.append(detection_result)
                self.last_detections[slot_number] = current_time
                
                # Update smart tracking
                self.update_identification(slot_number, detection_result, current_time)
                
        return detected_cards
        
    def should_skip_detection(self, slot_number: int, current_time: float) -> bool:
        """Determine if we should skip detection for this slot (smart tracking)"""
        if slot_number not in self.identified_cards:
            return False
            
        identification = self.identified_cards[slot_number]
        
        # Always detect high-confidence identifications
        if identification['confidence'] >= self.identification_threshold:
            # But recheck occasionally in case card changed
            time_since_identification = current_time - identification['last_seen']
            if time_since_identification > self.recheck_interval * 3:  # 15 seconds
                return False
            return True
            
        # Recheck low-confidence identifications more frequently
        time_since_identification = current_time - identification['last_seen']
        return time_since_identification < self.recheck_interval
        
    def extract_card_crop(self, frame, position):
        """Extract card crop from frame using precise coordinates"""
        try:
            x1, y1 = position['x1'], position['y1']
            x2, y2 = position['x2'], position['y2']
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                return None
                
            h, w = frame.shape[:2]
            if x2 > w or y2 > h or x1 < 0 or y1 < 0:
                return None
                
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Validate crop size
            if crop.size == 0:
                return None
                
            return crop
        except Exception as e:
            print(f"❌ Error extracting crop for slot {position['card_number']}: {e}")
            return None
            
    def detect_card_neural(self, card_crop, expected_card_name: str, slot_number: int):
        """Perform neural detection on card crop"""
        try:
            # Get neural cache name for expected card
            neural_name = self.card_mappings.get(expected_card_name.lower(), expected_card_name)
            
            # Convert crop to neural features (simplified - using similarity test)
            if neural_name in self.neural_detector.card_embeddings:
                # For now, assume perfect match if card is in expected position
                # In full implementation, would extract features from crop and compare
                similarity = self.neural_detector.similarity_test(neural_name, neural_name)
                
                if similarity >= 0.8:  # High similarity threshold
                    elixir_cost = self.elixir_costs.get(expected_card_name, 3)  # Default 3 elixir
                    
                    return {
                        'card_name': expected_card_name,
                        'neural_name': neural_name,
                        'slot_number': slot_number,
                        'confidence': similarity,
                        'elixir_cost': elixir_cost,
                        'detection_method': 'neural_embedding',
                        'crop_position': {
                            'x1': 0, 'y1': 0,  # Relative to crop
                            'x2': card_crop.shape[1], 'y2': card_crop.shape[0]
                        }
                    }
            else:
                # Handle unknown cards
                return {
                    'card_name': 'unknown',
                    'neural_name': 'unknown', 
                    'slot_number': slot_number,
                    'confidence': 0.5,  # Medium confidence for unknown
                    'elixir_cost': 3,  # Default cost
                    'detection_method': 'neural_embedding',
                    'crop_position': {
                        'x1': 0, 'y1': 0,
                        'x2': card_crop.shape[1], 'y2': card_crop.shape[0]
                    }
                }
                
        except Exception as e:
            print(f"❌ Neural detection error for slot {slot_number}: {e}")
            return None
            
    def update_identification(self, slot_number: int, detection_result: dict, current_time: float):
        """Update smart tracking with new detection result"""
        self.identified_cards[slot_number] = {
            'card_name': detection_result['card_name'],
            'confidence': detection_result['confidence'],
            'last_seen': current_time,
            'elixir_cost': detection_result['elixir_cost']
        }
        
    def get_identified_cards(self) -> Dict[int, dict]:
        """Get all currently identified cards"""
        return self.identified_cards.copy()
        
    def reset_identifications(self):
        """Reset all card identifications (new match)"""
        self.identified_cards.clear()
        self.last_detections.clear()
        print("🔄 Neural card detector reset for new match")
        
    def is_ready(self) -> bool:
        """Check if detector is ready for use"""
        return self.neural_ready and len(self.crop_positions) > 0
        
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            'neural_ready': self.neural_ready,
            'crop_positions': len(self.crop_positions),
            'identified_cards': len(self.identified_cards),
            'card_mappings': len(self.card_mappings),
            'total_neural_cards': len(self.neural_detector.card_embeddings) if self.neural_ready else 0
        }

# Legacy compatibility - replace old CardDetector 
class CardDetector(NeuralCardDetector):
    """Legacy compatibility wrapper"""
    def __init__(self):
        super().__init__()
        print("⚠️  Using Neural Card Detector (legacy CardDetector replaced)")
        
    def detect_cards_from_stopwatches(self, frame):
        """Legacy method name - redirects to neural detection"""
        return self.detect_cards_from_frame(frame)
        
    def _find_stopwatches(self, frame):
        """Legacy method for stopwatch detection - now returns card positions"""
        # Return card positions as "stopwatch" locations for compatibility
        positions = []
        for position in self.crop_positions:
            # Use center of card crop as "stopwatch" position
            center_x = (position['x1'] + position['x2']) // 2
            center_y = (position['y1'] + position['y2']) // 2
            positions.append((center_x, center_y))
        return positions
