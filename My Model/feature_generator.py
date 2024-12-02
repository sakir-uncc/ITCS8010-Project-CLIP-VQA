import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Dict

# Configuration variables
CLIP_FEATURE_DIM = 512  # CLIP ViT-B/32 output dimension
HIDDEN_DIM = 256       # Hidden dimension for branch processing
OUTPUT_DIM = 128       # Output dimension for each branch

class ZeroShotBranch(nn.Module):
    """Base class for zero-shot feature extraction using CLIP"""
    def __init__(self, clip_model, prompts: List[str], output_dim: int):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = clip_model
        self.prompts = prompts
        
        # Generate and cache text embeddings for prompts
        with torch.no_grad():
            text = clip.tokenize(prompts).to(self.device)  # Shape: (num_prompts, 77)
            text_features = self.clip_model.encode_text(text)  # Shape: (num_prompts, 512)
            # Convert from float16 to float32 and move to device
            self.text_features = text_features.float().to(self.device)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        # Projection layer to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(len(prompts), output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        ).to(self.device)
    
    def forward(self, image_features):
        """
        Args:
            image_features: Normalized CLIP image features (batch_size, CLIP_FEATURE_DIM)
        Returns:
            branch_features: Branch-specific features (batch_size, output_dim)
        """
        # Ensure image_features are float32 and on correct device
        image_features = image_features.float().to(self.device)
        
        # Compute similarity with all prompts
        similarities = image_features @ self.text_features.T  # Shape: (batch_size, num_prompts)
        
        # Project to output dimension
        branch_features = self.projection(similarities)  # Shape: (batch_size, output_dim)
        return branch_features

class WhatBranch(ZeroShotBranch):
    """Zero-shot branch for object presence"""
    def __init__(self, clip_model, output_dim=OUTPUT_DIM):
        # Define prompts for common objects and scenes
        prompts = [
            "a photo of a person",
            "a photo of an animal",
            "a photo of a vehicle",
            "a photo of furniture",
            "a photo of food",
            "a photo of electronics",
            "a photo of clothing",
            "a photo of sports equipment",
            "a photo containing buildings",
            "a photo of nature"
        ]
        super().__init__(clip_model, prompts, output_dim)

class WhereBranch(ZeroShotBranch):
    """Zero-shot branch for spatial information"""
    def __init__(self, clip_model, output_dim=OUTPUT_DIM):
        # Define prompts for spatial relationships
        prompts = [
            "an object in the center of the image",
            "an object on the left side",
            "an object on the right side",
            "an object at the top",
            "an object at the bottom",
            "objects in the foreground",
            "objects in the background",
            "objects close to the camera",
            "objects far from the camera",
            "objects spread across the image"
        ]
        super().__init__(clip_model, prompts, output_dim)

class AttributeBranch(ZeroShotBranch):
    """Zero-shot branch for object attributes"""
    def __init__(self, clip_model, output_dim=OUTPUT_DIM):
        # Define prompts for common attributes
        prompts = [
            "a red colored object",
            "a blue colored object",
            "a green colored object",
            "a large object",
            "a small object",
            "a shiny object",
            "a textured object",
            "a modern looking object",
            "an old looking object",
            "a transparent object"
        ]
        super().__init__(clip_model, prompts, output_dim)

class HowBranch(ZeroShotBranch):
    """Zero-shot branch for object relationships"""
    def __init__(self, clip_model, output_dim=OUTPUT_DIM):
        # Define prompts for object relationships
        prompts = [
            "objects next to each other",
            "objects stacked on top of each other",
            "objects interacting with each other",
            "objects aligned in a row",
            "objects grouped together",
            "objects scattered apart",
            "one object containing another",
            "objects in contact",
            "objects following each other",
            "objects facing each other"
        ]
        super().__init__(clip_model, prompts, output_dim)

class FeatureGenerator(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize branches
        self.what_branch = WhatBranch(clip_model)
        self.where_branch = WhereBranch(clip_model)
        self.attribute_branch = AttributeBranch(clip_model)
        self.how_branch = HowBranch(clip_model)
        
        # Fusion layer for object-level features
        self.object_fusion = nn.Sequential(
            nn.Linear(OUTPUT_DIM * 3, OUTPUT_DIM),  # 3 branches: what, where, attribute
            nn.ReLU(),
            nn.LayerNorm(OUTPUT_DIM)
        ).to(self.device)
        
    def to(self, device):
        """Override to() to handle custom components"""
        super().to(device)
        self.device = device
        self.what_branch = self.what_branch.to(device)
        self.where_branch = self.where_branch.to(device)
        self.attribute_branch = self.attribute_branch.to(device)
        self.how_branch = self.how_branch.to(device)
        self.object_fusion = self.object_fusion.to(device)
        return self
        
    def forward(self, clip_features):
        """
        Args:
            clip_features: Normalized CLIP features (batch_size, CLIP_FEATURE_DIM)
        Returns:
            dict containing:
            - global_features: Original CLIP features
            - object_features: Fused what/where/attribute features
            - relation_features: How branch features
        """
        # Ensure input is on correct device
        clip_features = clip_features.to(self.device)
        
        # Process through each branch
        what_features = self.what_branch(clip_features)      # Shape: (batch_size, OUTPUT_DIM)
        where_features = self.where_branch(clip_features)    # Shape: (batch_size, OUTPUT_DIM)
        attr_features = self.attribute_branch(clip_features) # Shape: (batch_size, OUTPUT_DIM)
        how_features = self.how_branch(clip_features)        # Shape: (batch_size, OUTPUT_DIM)
        
        # Concatenate object-level features
        object_concat = torch.cat(                          # Shape: (batch_size, OUTPUT_DIM * 3)
            [what_features, where_features, attr_features], 
            dim=1
        )
        
        # Fuse object-level features
        object_features = self.object_fusion(object_concat) # Shape: (batch_size, OUTPUT_DIM)
        
        return {
            'global_features': clip_features,               # Shape: (batch_size, CLIP_FEATURE_DIM)
            'object_features': object_features,             # Shape: (batch_size, OUTPUT_DIM)
            'relation_features': how_features               # Shape: (batch_size, OUTPUT_DIM)
        }

def test_feature_generator():
    """Test the FeatureGenerator with real CLIP model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Create random image features (assuming they're from CLIP)
    batch_size = 4
    clip_features = torch.randn(batch_size, CLIP_FEATURE_DIM, dtype=torch.float32).to(device)
    # Normalize features as CLIP would
    clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
    
    # Initialize feature generator
    generator = FeatureGenerator(clip_model)
    generator = generator.to(device)
    
    # Generate features
    with torch.no_grad():
        features = generator(clip_features)
    
    # Print shapes and statistics
    print("\nFeature shapes:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
        print(f"dtype: {feat.dtype}")
        
    # Print similarity scores for one example
    print("\nSimilarity scores for first image:")
    for branch in [generator.what_branch, generator.where_branch, 
                  generator.attribute_branch, generator.how_branch]:
        sims = clip_features[0:1].float() @ branch.text_features.T
        print(f"\n{branch.__class__.__name__} similarities:")
        for prompt, sim in zip(branch.prompts, sims[0]):
            print(f"{prompt}: {sim.item():.3f}")

if __name__ == "__main__":
    # Test the feature generator
    test_feature_generator()

# Expected output:
# Feature shapes:
# global_features: torch.Size([4, 512])
# object_features: torch.Size([4, 128])
# relation_features: torch.Size([4, 128])

# Similarity scores for first image:
# WhatBranch similarities:
# a photo of a person: 0.245
# a photo of an animal: 0.178
# ...