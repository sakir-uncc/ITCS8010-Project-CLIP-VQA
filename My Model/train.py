import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import time
from datetime import datetime

from model import VQAModel
from data_loader import get_data_loaders

# Configuration variables
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
CHECKPOINT_DIR = '/data1/Sakir/COCO QA/checkpoints_Object_relation'
LOG_DIR = '/data1/Sakir/COCO QA/logs_Object_relation'
CACHE_DIR = '/data1/Sakir/COCO QA/cache_Object_relation'

def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer,
                device: torch.device) -> tuple:
    """
    Train for one epoch
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        # Move batch to device
        batch['image'] = batch['image'].to(device)
        batch['answer_label'] = batch['answer_label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        loss = output['loss']
        logits = output['logits']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch['answer_label']).sum().item()
        total += len(batch['answer_label'])
        
        # Update running loss
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model: nn.Module, 
            val_loader: DataLoader,
            device: torch.device) -> tuple:
    """
    Validate the model
    Returns:
        val_loss: Average validation loss
        val_acc: Validation accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            batch['image'] = batch['image'].to(device)
            batch['answer_label'] = batch['answer_label'].to(device)
            
            # Forward pass
            output = model(batch)
            loss = output['loss']
            logits = output['logits']
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch['answer_label']).sum().item()
            total += len(batch['answer_label'])
            
            # Update running loss
            total_loss += loss.item()
    
    val_loss = total_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def train():
    """Main training function"""
    # Create directories if they don't exist
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader, num_classes, answer_to_idx = get_data_loaders()
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = VQAModel(num_classes=num_classes, cache_dir=CACHE_DIR)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize tracking variables
    best_val_acc = 0
    start_epoch = 0
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")
    
    print("Starting training...")
    print(f"Logging to {log_file}")
    
    with open(log_file, 'w') as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write("-" * 50 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Log results
        log_message = (f"Epoch {epoch + 1}/{NUM_EPOCHS}\n"
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%\n"
                      f"{'-' * 50}\n")
        
        with open(log_file, 'a') as f:
            f.write(log_message)
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
        model.save_checkpoint(
            checkpoint_path,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            model.save_checkpoint(
                best_model_path,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

def test_training():
    """Test the training pipeline with small random dataset"""
    # Get dataloaders with small batch size
    train_loader, val_loader, test_loader, num_classes, _ = get_data_loaders()
    
    # Initialize model
    model = VQAModel(num_classes=num_classes, cache_dir=CACHE_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Test one training epoch
    print("\nTesting training epoch...")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    # Test validation
    print("\nTesting validation...")
    val_loss, val_acc = validate(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Test checkpointing
    print("\nTesting checkpointing...")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "test_checkpoint.pth")
    model.save_checkpoint(checkpoint_path, optimizer, epoch=0, best_val_acc=val_acc)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    # Test the training pipeline
    print("Testing training pipeline...")
    test_training()
    
    # Start actual training
    print("\nStarting actual training...")
    train()