import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer, CLIPTextModel
import numpy as np

# Model Configuration
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 77  # CLIP's default max sequence length
MODEL_NAME = "openai/clip-vit-base-patch32"
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

class TextProcessor(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(TextProcessor, self).__init__()
        
        # Initialize CLIP text model and tokenizer
        self.clip = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Get the actual hidden size from the model config
        self.hidden_size = self.clip.config.hidden_size
        self.num_attention_heads = self.clip.config.num_attention_heads
        
        # Positional encoding layer with correct hidden size
        self.positional_encoding = PositionalEncoding(self.hidden_size, MAX_SEQ_LENGTH)
        
        # Dependency parsing - using the correct hidden size
        self.dependency_parser = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1
        )

    def freeze_parameters(self):
        """Freeze all parameters in the model"""
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, questions):
        """
        Args:
            questions (list): List of question strings
        Returns:
            dict: Dictionary containing text embeddings, positional encodings, and dependency parse
        """
        # Tokenize questions
        encoded = self.tokenizer(
            questions,
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        )

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        
        # Get CLIP text embeddings
        clip_output = self.clip(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        
        text_embeddings = clip_output.last_hidden_state
        
        # Add positional encoding
        pos_encoded = self.positional_encoding(text_embeddings)
        
        # Generate dependency parsing
        dep_features = self.dependency_parser(pos_encoded)
        dep_parse = torch.matmul(dep_features, dep_features.transpose(-2, -1))
        
        return {
            'text_embeddings': text_embeddings,      # Shape: (B, 77, hidden_size)
            'positional_encodings': pos_encoded,     # Shape: (B, 77, hidden_size)
            'dependency_parse': dep_parse,           # Shape: (B, 77, 77)
            'pooled_output': clip_output.pooler_output if hasattr(clip_output, 'pooler_output') else text_embeddings[:, 0]
        }

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix with the correct hidden size
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-np.log(10000.0) / hidden_size))
        
        pe = torch.zeros(1, max_seq_length, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, hidden_size)
        Returns:
            Tensor of shape (batch_size, seq_length, hidden_size)
        """
        return x + self.pe[:, :x.size(1)]

def test_text_processor():
    """
    Test function to verify the TextProcessor implementation using random inputs
    """
    # Initialize model
    processor = TextProcessor().to(DEVICE)
    print(f"Model initialized and moved to {DEVICE}")
    
    # Get the actual hidden size from the model
    hidden_size = processor.hidden_size
    print(f"Model hidden size: {hidden_size}")
    
    # Create random test questions
    test_questions = [
        "What color is the car?",
        "How many people are in the image?",
        "What is the weather like?"
    ] * (BATCH_SIZE // 3 + 1)
    test_questions = test_questions[:BATCH_SIZE]
    
    # Process questions
    with torch.no_grad():
        outputs = processor(test_questions)
    
    # Print output shapes
    print("\nOutput tensor shapes:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # Verify shapes
    assert outputs['text_embeddings'].shape == (BATCH_SIZE, MAX_SEQ_LENGTH, hidden_size)
    assert outputs['positional_encodings'].shape == (BATCH_SIZE, MAX_SEQ_LENGTH, hidden_size)
    assert outputs['dependency_parse'].shape == (BATCH_SIZE, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH)
    assert outputs['pooled_output'].shape == (BATCH_SIZE, hidden_size)
    print("\nAll shape assertions passed!")

if __name__ == "__main__":
    test_text_processor()