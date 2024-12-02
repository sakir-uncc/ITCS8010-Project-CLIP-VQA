import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from pathlib import Path
import clip
import os
import numpy as np

# Configuration variables
TRAIN_CSV = '/data1/Sakir/COCO QA/datasets/coco-qa/processed/train.csv'
TEST_CSV = '/data1/Sakir/COCO QA/datasets/coco-qa/processed/test.csv'
TRAIN_IMG_DIR = '/data1/Sakir/COCO QA/datasets/coco-qa/processed/train_images'
TEST_IMG_DIR = '/data1/Sakir/COCO QA/datasets/coco-qa/processed/test_images'
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_SPLIT = 0.1  # 10% of training data for validation
RANDOM_SEED = 42  # For reproducibility

class VQADataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        """
        Args:
            df: Pandas DataFrame containing the data
            img_dir: Directory with all the images
            transform: CLIP image transform
            is_test: Whether this is test set
        """
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        
        # Create answer to index mapping
        if not is_test:
            self.answer_to_idx = self._create_answer_mapping()
            self.num_classes = len(self.answer_to_idx)
    
    def _create_answer_mapping(self):
        """Create a mapping of unique answers to indices"""
        unique_answers = sorted(self.df['answer'].unique())
        return {answer: idx for idx, answer in enumerate(unique_answers)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict containing:
            - image: torch.Tensor of shape (3, 224, 224)
            - question: string
            - answer_label: torch.Tensor of shape (1,) - index of correct answer
            - type: int - question type
            - image_id: int
        """
        row = self.df.iloc[idx]
        
        # Load and transform image
        img_path = self.img_dir / f"{row['image_id']:012d}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # Shape: (3, 224, 224)
        
        # Get question
        question = row['question']
        
        # Create sample dict
        sample = {
            'image': image,  # Shape: (3, 224, 224)
            'question': question,
            'type': torch.tensor(row['type'], dtype=torch.long),
            'image_id': row['image_id']
        }
        
        # Add answer label if not test set
        if not self.is_test:
            answer_idx = self.answer_to_idx[row['answer']]
            sample['answer_label'] = torch.tensor(answer_idx, dtype=torch.long)
            sample['answer_text'] = row['answer']
        
        return sample

def create_splits(train_df, val_split=0.1, random_seed=42):
    """
    Create train and validation splits while maintaining class distribution
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Calculate split sizes
    val_size = int(len(train_df) * val_split)
    train_size = len(train_df) - val_size
    
    # Create indices for splitting
    indices = np.random.permutation(len(train_df))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Split the dataframe
    train_df_split = train_df.iloc[train_indices].reset_index(drop=True)
    val_df_split = train_df.iloc[val_indices].reset_index(drop=True)
    
    return train_df_split, val_df_split

def get_data_loaders():
    """
    Creates and returns train, validation, and test data loaders
    Returns:
        train_loader: DataLoader
        val_loader: DataLoader
        test_loader: DataLoader
        num_classes: int - number of unique answers
    """
    # Load CLIP image transform
    _, preprocess = clip.load("ViT-B/32", device="cpu")
    
    # Read the CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Create train/val splits
    train_df_split, val_df_split = create_splits(train_df, VAL_SPLIT, RANDOM_SEED)
    
    print(f"Total training samples: {len(train_df)}")
    print(f"Training split: {len(train_df_split)}")
    print(f"Validation split: {len(val_df_split)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = VQADataset(train_df_split, TRAIN_IMG_DIR, preprocess)
    val_dataset = VQADataset(val_df_split, TRAIN_IMG_DIR, preprocess)  # Note: Using same image dir as train
    test_dataset = VQADataset(test_df, TEST_IMG_DIR, preprocess, is_test=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes, train_dataset.answer_to_idx

# Test the data loader
if __name__ == "__main__":
    # Create random tensors for testing
    dummy_image = torch.randn(3, 224, 224)  # Random image tensor
    dummy_question = "What color is the car?"
    dummy_answer = "red"
    dummy_type = torch.tensor(1)
    
    print("Testing data loader with random tensors:")
    print(f"Image shape: {dummy_image.shape}")
    print(f"Question: {dummy_question}")
    print(f"Answer: {dummy_answer}")
    print(f"Type: {dummy_type}")
    
    # Test actual data loading
    train_loader, val_loader, test_loader, num_classes, answer_to_idx = get_data_loaders()
    
    print("\nTesting actual data loading:")
    print(f"Number of classes: {num_classes}")
    
    # Print number of batches in each loader
    print(f"\nNumber of batches:")
    print(f"Training: {len(train_loader)}")
    print(f"Validation: {len(val_loader)}")
    print(f"Testing: {len(test_loader)}")
    
    # Get a batch from train loader
    batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    print(f"Image batch shape: {batch['image'].shape}")  # Should be (BATCH_SIZE, 3, 224, 224)
    print(f"Type batch shape: {batch['type'].shape}")    # Should be (BATCH_SIZE,)
    print(f"Answer label shape: {batch['answer_label'].shape}")  # Should be (BATCH_SIZE,)
    
    # Print sample question-answer pair
    print("\nSample question-answer pair:")
    print(f"Question: {batch['question'][0]}")
    print(f"Answer: {batch['answer_text'][0]}")
    print(f"Answer label: {batch['answer_label'][0]}")
    
    # Print first few answer mappings
    print("\nSample answer mappings:")
    for answer, idx in list(answer_to_idx.items())[:5]:
        print(f"{answer}: {idx}")

# Testing data loader with random tensors:
# Image shape: torch.Size([3, 224, 224])
# Question: What color is the car?
# Answer: red
# Type: 1
# Total training samples: 999
# Training split: 900
# Validation split: 99
# Test samples: 499

# Testing actual data loading:
# Number of classes: 223

# Number of batches:
# Training: 29
# Validation: 4
# Testing: 16

# Sample batch shapes:
# Image batch shape: torch.Size([32, 3, 224, 224])
# Type batch shape: torch.Size([32])
# Answer label shape: torch.Size([32])

# Sample question-answer pair:
# Question: what is the color of the stove
# Answer: black
# Answer label: 26

# Sample answer mappings:
# airliner: 0
# airplane: 1
# airplanes: 2
# apple: 3
# apples: 4