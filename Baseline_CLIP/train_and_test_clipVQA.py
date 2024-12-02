import torch
import clip
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim

class VQADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Create answer vocabulary from training data
        self.answer_vocab = sorted(self.df['answer'].unique())
        self.answer_to_idx = {ans: idx for idx, ans in enumerate(self.answer_vocab)}
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']:012d}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'question': row['question'],
            'answer': row['answer'],
            'answer_idx': self.answer_to_idx[row['answer']],
            'type': row['type'],
            'image_id': row['image_id']
        }

class VQAModel(nn.Module):
    def __init__(self, clip_model, num_answers, hidden_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.hidden_dim = hidden_dim
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Additional layers for VQA
        self.fusion = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim + clip_model.text_projection.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers)
        )
        
    def forward(self, images, questions):
        # Get CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            question_features = self.clip_model.encode_text(questions)
            
            # Convert from float16 to float32
            image_features = image_features.float()
            question_features = question_features.float()
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        question_features = question_features / question_features.norm(dim=-1, keepdim=True)
        
        # Concatenate features
        combined_features = torch.cat([image_features, question_features], dim=1)
        
        # Pass through fusion layers
        output = self.fusion(combined_features)
        return output

def train_vqa_model(num_epochs=10, batch_size=32, learning_rate=1e-4):
    # Setup device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Create datasets
    train_dataset = VQADataset(
        csv_path='/data1/Sakir/COCO QA/datasets/coco-qa/processed/train.csv',  # Update with your actual path
        img_dir='/data1/Sakir/COCO QA/datasets/coco-qa/processed/train_images',  # Update with your actual path
        transform=preprocess
    )
    
    val_dataset = VQADataset(
        csv_path='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test.csv',  # Update with your actual path
        img_dir='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test_images',  # Update with your actual path
        transform=preprocess
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and move to device
    model = VQAModel(clip_model, len(train_dataset.answer_vocab))
    model = model.to(device)
    
    # Ensure model is in float32
    model = model.float()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = batch['image'].to(device)
            questions = clip.tokenize(batch['question']).to(device)
            answer_indices = batch['answer_idx'].to(device)
            
            # Forward pass
            outputs = model(images, questions)
            loss = criterion(outputs, answer_indices)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == answer_indices).sum().item()
            train_total += len(predictions)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                questions = clip.tokenize(batch['question']).to(device)
                answer_indices = batch['answer_idx'].to(device)
                
                outputs = model(images, questions)
                loss = criterion(outputs, answer_indices)
                
                val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == answer_indices).sum().item()
                val_total += len(predictions)
        
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_vqa_model.pth')
    
    return model

def evaluate_trained_model(model, test_dataset, batch_size=32):
    device = next(model.parameters()).device
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            questions = clip.tokenize(batch['question']).to(device)
            answer_indices = batch['answer_idx'].to(device)
            types = batch['type']
            
            outputs = model(images, questions)
            predictions = outputs.argmax(dim=1)
            
            correct += (predictions == answer_indices).sum().item()
            total += len(predictions)
            
            for pred, ans, typ in zip(predictions, answer_indices, types):
                type_total[typ.item()] += 1
                if pred == ans:
                    type_correct[typ.item()] += 1
    
    overall_accuracy = 100 * correct / total
    type_accuracies = {t: 100 * type_correct[t] / type_total[t] for t in type_total.keys()}
    
    print(f"\nTest Accuracy: {overall_accuracy:.2f}%")
    print("\nAccuracy by question type:")
    for type_id, accuracy in type_accuracies.items():
        print(f"Type {type_id}: {accuracy:.2f}%")
    
    return overall_accuracy, type_accuracies

if __name__ == "__main__":
    # Train the model
    trained_model = train_vqa_model()
    
    # # Create test dataset and evaluate
    # test_dataset = VQADataset(
    #     csv_path='/path/to/test.csv',
    #     img_dir='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test_images',
    #     transform=preprocess
    # )
    
    # accuracy, type_accuracies = evaluate_trained_model(trained_model, test_dataset)