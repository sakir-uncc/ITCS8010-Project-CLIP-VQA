import torch
import clip
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

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

def evaluate_clip_prediction():
    """Evaluate CLIP in a prediction setting (without seeing answers during inference)"""
    # Load CLIP model and preprocessing
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Create test dataset
    test_dataset = VQADataset(
        csv_path='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test.csv',
        img_dir='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test_images',
        transform=preprocess
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Statistics
    correct = 0
    total = 0
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    
    # Pre-encode all possible answers
    answers = test_dataset.answer_vocab
    answer_tokens = clip.tokenize([f"a photo of {ans}" for ans in answers]).to(device)
    with torch.no_grad():
        answer_features = model.encode_text(answer_tokens)
        answer_features = answer_features / answer_features.norm(dim=-1, keepdim=True)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            questions = batch['question']
            answer_indices = batch['answer_idx'].to(device)
            types = batch['type']
            
            # Encode images
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode questions
            question_tokens = clip.tokenize([f"{q}?" for q in questions]).to(device)
            question_features = model.encode_text(question_tokens)
            question_features = question_features / question_features.norm(dim=-1, keepdim=True)
            
            # Combine image and question features
            combined_features = (image_features + question_features) / 2
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities with all possible answers
            similarity = (100.0 * combined_features @ answer_features.T)
            
            # Get predictions
            predictions = similarity.argmax(dim=-1)
            
            # Calculate accuracy
            correct_predictions = (predictions == answer_indices)
            correct += correct_predictions.sum().item()
            total += len(predictions)
            
            # Calculate type-wise accuracy
            for pred, ans, typ in zip(predictions, answer_indices, types):
                type_total[typ.item()] += 1
                if pred == ans:
                    type_correct[typ.item()] += 1
            
            # Print some examples
            if total <= 64:  # Print first two batches only
                for i in range(len(questions)):
                    pred_answer = test_dataset.idx_to_answer[predictions[i].item()]
                    true_answer = test_dataset.idx_to_answer[answer_indices[i].item()]
                    print(f"\nQuestion: {questions[i]}")
                    print(f"Predicted: {pred_answer}")
                    print(f"True: {true_answer}")
                    print(f"Correct: {pred_answer == true_answer}")
    
    # Calculate accuracies
    overall_accuracy = 100 * correct / total
    type_accuracies = {t: 100 * type_correct[t] / type_total[t] 
                      for t in type_total.keys()}
    
    # Print results
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print("\nAccuracy by question type:")
    for type_id, accuracy in type_accuracies.items():
        print(f"Type {type_id}: {accuracy:.2f}%")
    
    return overall_accuracy, type_accuracies

if __name__ == "__main__":
    print("\nEvaluating CLIP in prediction mode...")
    accuracy, type_accuracies = evaluate_clip_prediction()