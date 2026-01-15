"""
Training script for strep throat classification model.
"""

import os
import argparse


def get_transforms(is_train=True):
    """Get data augmentation transforms."""
    from torchvision import transforms

    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    import torch
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        if len(batch) == 3:  # With clinical features
            images, clinical, labels = batch
            images = images.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, clinical)
        else:  # Without clinical features
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    import torch
    from tqdm import tqdm
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for positive class
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get probabilities using softmax and extract positive class (class 1)
            probs = torch.softmax(outputs, dim=1)
            pos_probs = probs[:, 1].cpu().numpy()  # Probability of positive class
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(pos_probs)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Compute ROC-AUC with edge case handling
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        # Handle case where only one class is present in y_true
        if "Only one class present" in str(e) or "Need at least 2 classes" in str(e):
            print(f"Warning: Cannot compute ROC-AUC - only one class present in validation set")
            roc_auc = None
        else:
            raise
    
    return epoch_loss, epoch_acc, precision, recall, f1, roc_auc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train strep throat classifier')
    parser.add_argument('--csv_path', type=str, 
                       default='data/sample_dataset_100.csv',
                       help='Path to CSV file')
    parser.add_argument('--images_dir', type=str, 
                       default='data/images',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and results')
    parser.add_argument('--mode', type=str, default='with_clinical',
                       choices=['with_clinical', 'image_only'],
                       help='Training mode: with_clinical or image_only')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Determine if clinical features should be used
    use_clinical = (args.mode == 'with_clinical')

    # Heavy imports (kept here so `python train.py --help` works even if optional deps are broken)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    import pandas as pd
    import matplotlib.pyplot as plt
    from dataset import StrepThroatDataset
    from model import StrepThroatClassifier
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode} (use_clinical={use_clinical})")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = StrepThroatDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        use_clinical_features=use_clinical,
        transform=get_transforms(is_train=True)
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Use validation transforms for validation set
    val_dataset.dataset.transform = get_transforms(is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = StrepThroatClassifier(
        num_clinical_features=7,
        use_clinical_features=use_clinical,
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_roc_aucs = []  # Store ROC-AUC per epoch
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_roc_aucs.append(val_roc_auc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        if val_roc_auc is not None:
            print(f"Val ROC-AUC: {val_roc_auc:.4f}")
        else:
            print(f"Val ROC-AUC: N/A (only one class present)")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with val acc: {best_val_acc:.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1 Score:  {val_f1:.4f}")
    if val_roc_auc is not None:
        print(f"  ROC-AUC:   {val_roc_auc:.4f}")
    else:
        print(f"  ROC-AUC:   N/A (only one class present)")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_roc_auc': [auc if auc is not None else float('nan') for auc in val_roc_aucs]
    })
    metrics_df.to_csv(os.path.join(args.output_dir, 'training_metrics.csv'), index=False)
    
    # Save final metrics
    final_metrics = {
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1,
        'roc_auc': val_roc_auc if val_roc_auc is not None else float('nan')
    }
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(args.output_dir, 'final_metrics.csv'), index=False
    )
    
    print(f"\nResults saved to {args.output_dir}/")
    print("="*50)


if __name__ == '__main__':
    main()
