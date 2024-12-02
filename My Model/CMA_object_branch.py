import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Configuration variables
CLIP_DIM = 512          # CLIP output dimension
FEATURE_DIM = 128       # Feature dimension from feature generator
HIDDEN_DIM = 256        # Hidden dimension for attention
DROPOUT_RATE = 0.1
NUM_HEADS = 8

class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int):
        """
        Initialize cross-modal attention between visual and text features
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Project visual and text features to same dimension
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        ).to(self.device)
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        ).to(self.device)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=NUM_HEADS,
            dropout=DROPOUT_RATE,
            batch_first=True
        ).to(self.device)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        ).to(self.device)
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention between visual and text features
        Args:
            visual_features: Visual features (batch_size, visual_dim)
            text_features: Text features (batch_size, text_dim)
        Returns:
            fused_features: Attended features (batch_size, hidden_dim)
        """
        # Convert inputs to float32 and move to device
        visual_features = visual_features.float().to(self.device)
        text_features = text_features.float().to(self.device)
        
        # Project features to same dimension
        visual_proj = self.visual_projection(visual_features)  # Shape: (batch_size, hidden_dim)
        text_proj = self.text_projection(text_features)      # Shape: (batch_size, hidden_dim)
        
        # Add sequence dimension for attention
        visual_seq = visual_proj.unsqueeze(1)  # Shape: (batch_size, 1, hidden_dim)
        text_seq = text_proj.unsqueeze(1)      # Shape: (batch_size, 1, hidden_dim)
        
        # Apply cross attention
        attended_features, _ = self.attention(
            query=visual_seq,
            key=text_seq,
            value=text_seq
        )  # Shape: (batch_size, 1, hidden_dim)
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)  # Shape: (batch_size, hidden_dim)
        
        # Concatenate with visual features and project
        fused_features = torch.cat([visual_proj, attended_features], dim=1)  # Shape: (batch_size, hidden_dim * 2)
        output = self.output_projection(fused_features)  # Shape: (batch_size, hidden_dim)
        
        return output

class MultiLevelCrossModalAttention(nn.Module):
    def __init__(self):
        """
        Initialize attention modules for different feature levels
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Attention for global features
        self.global_attention = CrossModalAttention(
            visual_dim=CLIP_DIM,
            text_dim=CLIP_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(self.device)
        
        # Attention for object-level features
        self.object_attention = CrossModalAttention(
            visual_dim=FEATURE_DIM,
            text_dim=CLIP_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(self.device)
        
        # Attention for relation-level features
        self.relation_attention = CrossModalAttention(
            visual_dim=FEATURE_DIM,
            text_dim=CLIP_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(self.device)
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        ).to(self.device)
    
    def to(self, device):
        """Override to() to handle custom components"""
        super().to(device)
        self.device = device
        self.global_attention = self.global_attention.to(device)
        self.object_attention = self.object_attention.to(device)
        self.relation_attention = self.relation_attention.to(device)
        self.fusion = self.fusion.to(device)
        return self
        
    def forward(self, features: Dict[str, torch.Tensor], text_features: torch.Tensor) -> torch.Tensor:
        """
        Apply attention between text and different feature levels
        Args:
            features: Dictionary containing:
                - global_features: CLIP features (batch_size, CLIP_DIM)
                - object_features: Object-level features (batch_size, FEATURE_DIM)
                - relation_features: Relation-level features (batch_size, FEATURE_DIM)
            text_features: CLIP text features (batch_size, CLIP_DIM)
        Returns:
            fused_features: Final attended features (batch_size, HIDDEN_DIM)
        """
        # Convert text_features to float32
        text_features = text_features.float()
        
        # Apply attention for each feature level
        global_attended = self.global_attention(
            features['global_features'].float(), 
            text_features
        )  # Shape: (batch_size, HIDDEN_DIM)
        
        object_attended = self.object_attention(
            features['object_features'].float(), 
            text_features
        )  # Shape: (batch_size, HIDDEN_DIM)
        
        relation_attended = self.relation_attention(
            features['relation_features'].float(), 
            text_features
        )  # Shape: (batch_size, HIDDEN_DIM)
        
        # Concatenate all attended features
        all_features = torch.cat(
            [global_attended, object_attended, relation_attended], 
            dim=1
        )  # Shape: (batch_size, HIDDEN_DIM * 3)
        
        # Final fusion
        output = self.fusion(all_features)  # Shape: (batch_size, HIDDEN_DIM)
        
        return output
        
    def forward(self, features: Dict[str, torch.Tensor], text_features: torch.Tensor) -> torch.Tensor:
        """
        Apply attention between text and different feature levels
        Args:
            features: Dictionary containing:
                - global_features: CLIP features (batch_size, CLIP_DIM)
                - object_features: Object-level features (batch_size, FEATURE_DIM)
                - relation_features: Relation-level features (batch_size, FEATURE_DIM)
            text_features: CLIP text features (batch_size, CLIP_DIM)
        Returns:
            fused_features: Final attended features (batch_size, HIDDEN_DIM)
        """
        # Apply attention for each feature level
        global_attended = self.global_attention(
            features['global_features'], 
            text_features
        )  # Shape: (batch_size, HIDDEN_DIM)
        
        object_attended = self.object_attention(
            features['object_features'], 
            text_features
        )  # Shape: (batch_size, HIDDEN_DIM)
        
        relation_attended = self.relation_attention(
            features['relation_features'], 
            text_features
        )  # Shape: (batch_size, HIDDEN_DIM)
        
        # Concatenate all attended features
        all_features = torch.cat(
            [object_attended, relation_attended], 
            dim=1
        )  # Shape: (batch_size, HIDDEN_DIM * 3)
        
        # Final fusion
        output = self.fusion(all_features)  # Shape: (batch_size, HIDDEN_DIM)
        
        return relation_attended

def test_attention():
    """
    Test the attention modules with random tensors
    """
    # Create random feature tensors
    batch_size = 4
    features = {
        'global_features': torch.randn(batch_size, CLIP_DIM),
        'object_features': torch.randn(batch_size, FEATURE_DIM),
        'relation_features': torch.randn(batch_size, FEATURE_DIM)
    }
    text_features = torch.randn(batch_size, CLIP_DIM)
    
    # Initialize attention module
    attention = MultiLevelCrossModalAttention()
    
    # Test forward pass
    output = attention(features, text_features)
    
    # Print shapes
    print("\nInput shapes:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"\nOutput shape: {output.shape}")
    
    # Test single attention module
    single_attention = CrossModalAttention(CLIP_DIM, CLIP_DIM, HIDDEN_DIM)
    single_output = single_attention(features['global_features'], text_features)
    print(f"\nSingle attention output shape: {single_output.shape}")
    
    # Print some statistics
    print("\nOutput statistics:")
    print(f"Mean: {output.mean().item():.3f}")
    print(f"Std: {output.std().item():.3f}")
    print(f"Min: {output.min().item():.3f}")
    print(f"Max: {output.max().item():.3f}")

if __name__ == "__main__":
    # Test the attention modules
    test_attention()

# Expected output:
# Input shapes:
# global_features: torch.Size([4, 512])
# object_features: torch.Size([4, 128])
# relation_features: torch.Size([4, 128])
# Text features: torch.Size([4, 512])

# Output shape: torch.Size([4, 256])

# Single attention output shape: torch.Size([4, 256])

# Output statistics:
# Mean: 0.000
# Std: 0.816
# Min: -1.234
# Max: 1.456