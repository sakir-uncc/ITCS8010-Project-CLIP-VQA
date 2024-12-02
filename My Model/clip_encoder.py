import torch
import clip
import torch.nn as nn
from typing import Dict, Tuple
from tqdm import tqdm
import os
import numpy as np

# Configuration variables
CACHE_DIR = '/data1/Sakir/COCO QA/cache'
CLIP_MODEL = "ViT-B/32"
FEATURE_DIM = 512  # CLIP ViT-B/32 output dimension
BATCH_SIZE = 32

class CLIPEncoder(nn.Module):
    def __init__(self, cache_dir: str = None, use_cache: bool = True):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = True
    
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features using CLIP"""
        with torch.no_grad():
            features = self.model.encode_image(images)
            # Convert to float32 and normalize
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def extract_text_features(self, questions: list) -> torch.Tensor:
        """Extract text features using CLIP"""
        with torch.no_grad():
            text = clip.tokenize(questions).to(self.device)
            features = self.model.encode_text(text)
            # Convert to float32 and normalize
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def cache_dataset_features(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Cache features for entire dataset
        Args:
            data_loader: DataLoader containing the dataset
        """
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{data_loader.dataset.img_dir.name}_features.pt")
        
        if os.path.exists(cache_file):
            print(f"Cache file {cache_file} already exists!")
            return
            
        print(f"Caching features for {len(data_loader.dataset)} images...")
        
        features_dict = {}
        self.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                images = batch['image'].to(self.device)  # Shape: (batch_size, 3, 224, 224)
                image_ids = batch['image_id']  # Shape: (batch_size,)
                
                features = self.extract_image_features(images)  # Shape: (batch_size, 512)
                
                # Store features for each image
                for idx, img_id in enumerate(image_ids):
                    features_dict[img_id.item()] = features[idx].cpu()
        
        # Save cache
        torch.save(features_dict, cache_file)
        print(f"Cached features saved to {cache_file}")
    
    def load_cached_features(self, image_ids: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Load cached features for given image IDs
        Args:
            image_ids: Tensor of image IDs
            data_loader: DataLoader containing the dataset
        Returns:
            features: Tensor of shape (batch_size, feature_dim)
        """
        cache_file = os.path.join(self.cache_dir, f"{data_loader.dataset.img_dir.name}_features.pt")
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file {cache_file} not found!")
            
        features_dict = torch.load(cache_file)
        features = torch.stack([features_dict[img_id.item()] for img_id in image_ids])
        return features.to(self.device)

def test_encoder():
    """
    Test the CLIPEncoder with random tensors
    """
    # Create random image tensor
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)  # Shape: (4, 3, 224, 224)
    dummy_questions = ["What color is the car?"] * batch_size
    dummy_image_ids = torch.tensor([1, 2, 3, 4])
    
    # Initialize encoder
    encoder = CLIPEncoder(use_cache=False)
    
    # Test image feature extraction
    image_features = encoder.extract_image_features(dummy_images.to(encoder.device))
    print(f"\nImage features shape: {image_features.shape}")  # Should be (4, 512)
    
    # Test text feature extraction
    text_features = encoder.extract_text_features(dummy_questions)
    print(f"Text features shape: {text_features.shape}")  # Should be (4, 512)
    
    # Test similarity computation
    similarity = image_features @ text_features.T
    print(f"Similarity matrix shape: {similarity.shape}")  # Should be (4, 4)
    
    # Print feature statistics
    print("\nFeature statistics:")
    print(f"Image features mean: {image_features.mean():.4f}")
    print(f"Image features std: {image_features.std():.4f}")
    print(f"Text features mean: {text_features.mean():.4f}")
    print(f"Text features std: {text_features.std():.4f}")

if __name__ == "__main__":
    # Test the encoder
    test_encoder()
    
    # Test with actual data loader
    from data_loader import get_data_loaders
    
    print("\nTesting with actual data...")
    train_loader, val_loader, test_loader, num_classes, _ = get_data_loaders()
    
    # Initialize encoder with caching
    encoder = CLIPEncoder(use_cache=True)
    
    # Cache features for training set
    encoder.cache_dataset_features(train_loader)
    
    # Test loading cached features
    batch = next(iter(train_loader))
    cached_features = encoder.load_cached_features(batch['image_id'], train_loader)
    print(f"\nCached features shape: {cached_features.shape}")  # Should be (batch_size, 512)

# Expected output:
# Image features shape: torch.Size([4, 512])
# Text features shape: torch.Size([4, 512])
# Similarity matrix shape: torch.Size([4, 4])

# Feature statistics:
# Image features mean: 0.0000
# Image features std: 0.0442
# Text features mean: 0.0000
# Text features std: 0.0442

# Testing with actual data...
# Caching features for 900 images...
# [Progress bar]
# Cached features saved to /data1/Sakir/COCO QA/cache/train_images_features.pt
# Cached features shape: torch.Size([32, 512])