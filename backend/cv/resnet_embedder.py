#!/usr/bin/env python3
"""
ResNet-based image embedder for card recognition with caching.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
import time
import logging
from config.settings import EMBEDDINGS_CACHE_DIR

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ResNetEmbedder:
    """ResNet-based image embedder for card recognition with caching."""

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        try:
            # Always use CPU for now (GPU support can be added later)
            self.device = torch.device('cpu')
            logger.info(f"🖥️ ResNet using device: {self.device}")

            # Try to load ResNet18 with pretrained weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

            # Remove the final classification layer to get features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            
            # 🔥 Apply INT8 quantization for 2-3x speedup on CPU
            logger.info("🔧 Applying INT8 quantization for CPU optimization...")
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.info("✅ CPU quantization applied successfully")
                self.quantized = True
            except Exception as e:
                logger.warning(f"⚠️ Quantization failed, using FP32: {e}")
                self.quantized = False

            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.embeddings_cache = {}
            self.cache_dir = EMBEDDINGS_CACHE_DIR
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
            # Performance tracking
            self.inference_times = []
            self.total_inferences = 0
            
            # Benchmark the model
            self._benchmark_inference_speed()
            
            logger.info("✅ ResNet embedder initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ResNet: {e}")
            raise
    
    def _benchmark_inference_speed(self):
        """Benchmark inference speed to measure actual performance."""
        logger.info("🔬 Benchmarking ResNet inference speed...")
        
        # Create dummy 224x224 RGB image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Warmup run (first run is always slower)
        _ = self.get_embedding(dummy_image)
        
        # Benchmark runs
        times = []
        for i in range(10):
            start = time.time()
            _ = self.get_embedding(dummy_image)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        logger.info(f"📊 Inference benchmark results:")
        logger.info(f"   Average: {avg_time:.1f}ms")
        logger.info(f"   Min: {min_time:.1f}ms")
        logger.info(f"   Max: {max_time:.1f}ms")
        logger.info(f"   Quantized: {'Yes (INT8)' if self.quantized else 'No (FP32)'}")
        
        # Store average for delay compensation
        self.avg_inference_time_ms = avg_time
        
        return avg_time

    def get_embedding(self, image):
        """Get embedding vector for an image with performance tracking."""
        start_time = time.time()
        
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply preprocessing
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(tensor)
                # Flatten to 1D vector
                embedding = embedding.squeeze().cpu().numpy()
            
            # Track inference time
            elapsed_ms = (time.time() - start_time) * 1000
            self.inference_times.append(elapsed_ms)
            self.total_inferences += 1
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            return embedding
        return None
    
    def get_average_inference_time(self):
        """Get average inference time in milliseconds."""
        if not self.inference_times:
            return self.avg_inference_time_ms  # Return benchmark result
        return np.mean(self.inference_times)

    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def save_embedding(self, card_name, embedding):
        """Save embedding to disk."""
        cache_file = self.cache_dir / f"{card_name}.npy"
        np.save(cache_file, embedding)

    def load_embedding(self, card_name):
        """Load embedding from disk if exists."""
        cache_file = self.cache_dir / f"{card_name}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def precompute_card_embeddings(self, card_templates, force_recompute=False):
        """Precompute embeddings for all card templates, with caching."""
        total_cards = 0

        for card_name, template_data in card_templates.items():
            # Check if already cached
            if not force_recompute:
                cached_embedding = self.load_embedding(card_name)
                if cached_embedding is not None:
                    self.embeddings_cache[card_name] = cached_embedding
                    total_cards += 1
                    continue

            # Compute new embedding
            image = template_data.get('image')
            if image is not None:
                embedding = self.get_embedding(image)
                if embedding is not None:
                    self.embeddings_cache[card_name] = embedding
                    self.save_embedding(card_name, embedding)
                    total_cards += 1

        return total_cards

    def find_best_match(self, crop_image, top_k=1):
        """Find best matching card for a crop image."""
        if not self.embeddings_cache:
            return []

        crop_embedding = self.get_embedding(crop_image)
        if crop_embedding is None:
            return []

        similarities = []
        for card_name, card_embedding in self.embeddings_cache.items():
            similarity = self.cosine_similarity(crop_embedding, card_embedding)
            similarities.append((card_name, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
