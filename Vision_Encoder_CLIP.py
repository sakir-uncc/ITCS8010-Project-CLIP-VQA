import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

# Configuration variables
BATCH_SIZE = 8
IMAGE_SIZE = 224
HIDDEN_DIM = 768  # CLIP-ViT-Base hidden dimension
MODEL_NAME = "openai/clip-vit-base-patch32"  # Pretrained model name
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

class PretrainedVisionEncoder(nn.Module):
    """Vision Encoder using pretrained CLIP model"""
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        
        # Load pretrained CLIP model
        self.clip = CLIPModel.from_pretrained(model_name)
        
        # We only need the vision model
        self.vision_model = self.clip.vision_model
        
        # Load the official preprocessor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Additional transform for tensor conversion
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP-specific normalization
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def freeze_parameters(self):
        """Freeze all parameters in the model"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def preprocess_images(self, images):
        """
        Preprocess images using the CLIP processor
        Args:
            images: List of PIL images or tensor of shape (B, C, H, W)
        Returns:
            Preprocessed tensor of shape (B, C, H, W)
        """
        if torch.is_tensor(images):
            images = images.to('cpu')
            # If input is already a tensor, convert to numpy
            if images.dim() == 4:  # (B, C, H, W)
                images = [Image.fromarray(
                    (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ) for img in images]
            else:
                raise ValueError("Expected 4D tensor of shape (B, C, H, W)")
        
        # Process images using the CLIP processor
        processed = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        return processed['pixel_values']
    
    def forward(self, x, output_attentions=False):
        """
        Forward pass through the vision encoder
        Args:
            x: Input tensor of shape (B, C, H, W) or list of PIL images
            output_attentions: Whether to output attention weights
        Returns:
            dict containing:
                - embeddings: Tensor of shape (B, N, hidden_dim)
                - pooled: Tensor of shape (B, hidden_dim)
                - attentions: Optional tuple of attention weights
        """
        # Preprocess images
        x = self.preprocess_images(x)
        
        # Move to device
        x = x.to(DEVICE)
        
        # Get vision model outputs
        outputs = self.vision_model(
            x,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        return {
            'embeddings': outputs.last_hidden_state,  # Sequence of patch embeddings
            'pooled': outputs.pooler_output,         # Pooled CLS token
            'attentions': outputs.attentions if output_attentions else None
        }

# Test the implementation
if __name__ == "__main__":
    # Initialize model
    model = PretrainedVisionEncoder().to(DEVICE)
    print("Model initialized and moved to", DEVICE)
    
    # Test with random tensor
    print("\nTesting with random tensor:")
    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    outputs = model(x)
    print(f"Embedding shape: {outputs['embeddings'].shape}")
    print(f"Pooled shape: {outputs['pooled'].shape}")
    
    # Test with attention outputs
    print("\nTesting with attention outputs:")
    outputs_with_attention = model(x, output_attentions=True)
    if outputs_with_attention['attentions'] is not None:
        print(f"Number of attention layers: {len(outputs_with_attention['attentions'])}")
        print(f"Attention shape for each layer: {outputs_with_attention['attentions'][0].shape}")
    
    # Test with a single synthetic image
    print("\nTesting with synthetic image:")
    synthetic_image = Image.fromarray(
        (np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8)
    )
    outputs_single = model([synthetic_image])
    print(f"Single image embedding shape: {outputs_single['embeddings'].shape}")
    print(f"Single image pooled shape: {outputs_single['pooled'].shape}")