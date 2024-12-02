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
            'type': row['type'],
            'image_id': row['image_id']
        }

def print_type_samples(df, num_samples=3):
    print("\nQuestion-Answer Samples by Type:")
    print("-" * 80)
    
    for type_id in df['type'].unique():
        type_samples = df[df['type'] == type_id].sample(min(num_samples, len(df[df['type'] == type_id])))
        print(f"\nType {type_id} samples:")
        for _, sample in type_samples.iterrows():
            print(f"Image ID: {sample['image_id']}")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print("-" * 40)

def evaluate_clip():
    # Load CLIP model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Create test dataset
    test_dataset = VQADataset(
        csv_path='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test.csv',
        img_dir='/data1/Sakir/COCO QA/datasets/coco-qa/processed/test_images',
        transform=preprocess
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    # Create arrays to store predictions and types
    all_predictions = []
    all_types = []
    
    # Store correct and incorrect examples for each type
    type_examples = defaultdict(lambda: {'correct': [], 'incorrect': []})
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            questions = batch['question']
            answers = batch['answer']
            types = batch['type']
            image_ids = batch['image_id']
            
            # Combine question and answer into text pairs
            text_inputs = []
            for q, a in zip(questions, answers):
                text = f"Question: {q} Answer: {a}"
                text_inputs.append(text)
            
            # Encode images and text
            image_features = model.encode_image(images)
            text_features = model.encode_text(clip.tokenize(text_inputs).to(device))
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get predictions
            values, predictions = similarity.max(dim=-1)
            
            # Calculate batch accuracy
            batch_correct = (predictions == torch.arange(len(predictions)).to(device))
            correct += batch_correct.sum().item()
            total += len(predictions)
            
            # Store predictions and types for per-type accuracy
            all_predictions.extend(batch_correct.cpu().numpy())
            all_types.extend(types.numpy())
            
            # Store examples
            for i, (is_correct, type_id, q, a, img_id) in enumerate(zip(
                batch_correct.cpu().numpy(), 
                types.numpy(), 
                questions,
                answers,
                image_ids
            )):
                example = {
                    'image_id': img_id,
                    'question': q,
                    'answer': a
                }
                if is_correct:
                    type_examples[type_id]['correct'].append(example)
                else:
                    type_examples[type_id]['incorrect'].append(example)
    
    accuracy = 100 * correct / total
    
    # Calculate type-wise accuracy
    all_predictions = np.array(all_predictions)
    all_types = np.array(all_types)
    
    type_accuracies = {}
    types = np.unique(all_types)
    for type_id in types:
        type_mask = all_types == type_id
        type_correct = all_predictions[type_mask].sum()
        type_total = type_mask.sum()
        type_accuracies[type_id] = 100 * type_correct / type_total
    
    return accuracy, type_accuracies, type_examples

if __name__ == "__main__":
    # Print samples from the test set before evaluation
    test_df = pd.read_csv('/data1/Sakir/COCO QA/datasets/coco-qa/processed/test.csv')
    print_type_samples(test_df)
    
    print("\nEvaluating CLIP...\n")
    accuracy, type_accuracies, type_examples = evaluate_clip()
    
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("\nAccuracy by question type:")
    for type_id, type_acc in type_accuracies.items():
        print(f"\nType {type_id}:")
        print(f"Accuracy: {type_acc:.2f}%")
        print(f"Sample correct prediction:")
        if type_examples[type_id]['correct']:
            example = type_examples[type_id]['correct'][0]
            print(f"Image ID: {example['image_id']}")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['answer']}")
        print(f"\nSample incorrect prediction:")
        if type_examples[type_id]['incorrect']:
            example = type_examples[type_id]['incorrect'][0]
            print(f"Image ID: {example['image_id']}")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['answer']}")
        print("-" * 40)