"""
Evaluation script for strep throat classification model.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset import StrepThroatDataset
from model import StrepThroatClassifier


def get_transforms():
    """Get validation transforms."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # With clinical features
                images, clinical, labels = batch
                images = images.to(device)
                clinical = clinical.to(device)
                labels = labels.to(device)
                
                outputs = model(images, clinical)
            else:  # Without clinical features
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate strep throat classifier')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--csv_path', type=str, 
                       default='data/sample_dataset_100.csv',
                       help='Path to CSV file')
    parser.add_argument('--images_dir', type=str, 
                       default='data/images',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--use_clinical', action='store_true', default=True,
                       help='Use clinical features')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = StrepThroatDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        use_clinical_features=args.use_clinical,
        transform=get_transforms()
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2)
    
    print(f"Total samples: {len(dataset)}")
    
    # Create model
    model = StrepThroatClassifier(
        num_clinical_features=7,
        use_clinical_features=args.use_clinical,
        pretrained=False
    ).to(device)
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    # Evaluate
    print("\nEvaluating...")
    preds, labels, probs = evaluate(model, dataloader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, 
                                target_names=['Negative', 'Positive']))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        labels, preds,
        os.path.join(args.output_dir, 'evaluation_confusion_matrix.png')
    )
    
    # Save predictions
    results_df = pd.DataFrame({
        'image_name': dataset.df['ImageName'].values,
        'true_label': dataset.df['label'].values,
        'predicted_label': ['Positive' if p == 1 else 'Negative' for p in preds],
        'probability_positive': [prob[1] for prob in probs],
        'probability_negative': [prob[0] for prob in probs]
    })
    results_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }])
    metrics_df.to_csv(os.path.join(args.output_dir, 'evaluation_metrics.csv'), index=False)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("="*50)


if __name__ == '__main__':
    main()
