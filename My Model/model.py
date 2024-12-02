import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from clip_encoder import CLIPEncoder
from feature_generator import FeatureGenerator
from CMA_object_branch import MultiLevelCrossModalAttention

# Configuration variables
CLIP_DIM = 512          # CLIP output dimension
HIDDEN_DIM = 256        # Hidden dimension for attention/fusion
DROPOUT_RATE = 0.1

class VQAModel(nn.Module):
    def __init__(self, num_classes: int, cache_dir: str = None):
        """
        Initialize the complete VQA model
        Args:
            num_classes: Number of possible answers
            cache_dir: Directory for caching CLIP features
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.clip_encoder = CLIPEncoder(cache_dir=cache_dir if cache_dir else None)
        self.feature_generator = FeatureGenerator(self.clip_encoder.model).to(self.device)
        self.cross_modal_attention = MultiLevelCrossModalAttention().to(self.device)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, num_classes)
        ).to(self.device)
    
    def to(self, device):
        """Override to() to handle custom components"""
        super().to(device)
        self.device = device
        self.clip_encoder = self.clip_encoder.to(device)
        self.feature_generator = self.feature_generator.to(device)
        self.cross_modal_attention = self.cross_modal_attention.to(device)
        self.classifier = self.classifier.to(device)
        return self

        
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            batch: Dictionary containing:
                - image: Image tensor (batch_size, 3, 224, 224)
                - question: List of question strings
                - answer_label: Answer labels (batch_size,) [optional]
        Returns:
            Dictionary containing:
                - logits: Predicted logits (batch_size, num_classes)
                - loss: Classification loss if answer_label provided
        """
        # Extract CLIP features
        image_features = self.clip_encoder.extract_image_features(batch['image'])
        text_features = self.clip_encoder.extract_text_features(batch['question'])
        # Shape: image_features, text_features -> (batch_size, CLIP_DIM)
        
        # Generate multi-branch features
        branch_features = self.feature_generator(image_features)
        # Shape: {
        #   'global_features': (batch_size, CLIP_DIM),
        #   'object_features': (batch_size, FEATURE_DIM),
        #   'relation_features': (batch_size, FEATURE_DIM)
        # }
        
        # Apply cross-modal attention
        attended_features = self.cross_modal_attention(branch_features, text_features)
        # Shape: attended_features -> (batch_size, HIDDEN_DIM)
        
        # Classification
        logits = self.classifier(attended_features)
        # Shape: logits -> (batch_size, num_classes)
        
        output = {'logits': logits}
        
        # Calculate loss if labels provided
        if 'answer_label' in batch:
            loss = F.cross_entropy(logits, batch['answer_label'])
            output['loss'] = loss
        
        return output
    
    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None, 
                       epoch: int = None, best_val_acc: float = None):
        """
        Save model checkpoint
        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state to save
            epoch: Current epoch number
            best_val_acc: Best validation accuracy
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, optimizer: torch.optim.Optimizer = None):
        """
        Load model checkpoint
        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into
        Returns:
            epoch: Epoch number
            best_val_acc: Best validation accuracy
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['epoch'], checkpoint['best_val_acc']

def test_model():
    """
    Test the model with random tensors
    """
    # Create random batch
    batch_size = 4
    num_classes = 100
    
    batch = {
        'image': torch.randn(batch_size, 3, 224, 224),
        'question': ['What is in the image?'] * batch_size,
        'answer_label': torch.randint(0, num_classes, (batch_size,))
    }
    
    # Initialize model
    model = VQAModel(num_classes=num_classes)
    
    # Move batch to device
    batch['image'] = batch['image'].to(model.device)
    batch['answer_label'] = batch['answer_label'].to(model.device)
    
    # Forward pass
    output = model(batch)
    
    # Print shapes and statistics
    print("\nInput shapes:")
    print(f"Image: {batch['image'].shape}")
    print(f"Questions: {len(batch['question'])} strings")
    print(f"Labels: {batch['answer_label'].shape}")
    
    print("\nOutput shapes:")
    print(f"Logits: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test checkpointing
    print("\nTesting checkpointing...")
    optimizer = torch.optim.Adam(model.parameters())
    model.save_checkpoint('test_checkpoint.pth', optimizer, epoch=1, best_val_acc=0.75)
    loaded_epoch, loaded_acc = model.load_checkpoint('test_checkpoint.pth', optimizer)
    print(f"Loaded checkpoint - Epoch: {loaded_epoch}, Best acc: {loaded_acc}")

if __name__ == "__main__":
    # Test the model
    test_model()

# Expected output:
# Input shapes:
# Image: torch.Size([4, 3, 224, 224])
# Questions: 4 strings
# Labels: torch.Size([4])

# Output shapes:
# Logits: torch.Size([4, 100])
# Loss: 4.6052

# Testing checkpointing...
# Loaded checkpoint - Epoch: 1, Best acc: 0.75