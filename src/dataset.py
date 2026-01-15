"""
Dataset class for loading throat images and clinical symptoms.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class StrepThroatDataset(Dataset):
    """
    Dataset for strep throat classification.
    
    Args:
        csv_path: Path to CSV file with image names, labels, and symptoms
        images_dir: Directory containing the images
        use_clinical_features: Whether to include clinical symptoms
        transform: Image transformations to apply
    """
    
    def __init__(self, csv_path, images_dir, use_clinical_features=True, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.use_clinical_features = use_clinical_features
        self.transform = transform
        
        # Clinical feature columns (excluding ImageName and label)
        self.clinical_features = [
            'Hoarseness', 'Rhinorrhea', 'sorethroat', 'Congestion',
            'Knownrecentcontact', 'Headache', 'Fever'
        ]
        
        # Convert labels to binary (Positive=1, Negative=0)
        self.df['label_encoded'] = (self.df['label'] == 'Positive').astype(int)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_name = row['ImageName']
        img_path = os.path.join(self.images_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        
        # Get clinical features if requested
        if self.use_clinical_features:
            clinical = torch.tensor(
                [row[feat] for feat in self.clinical_features],
                dtype=torch.float32
            )
            return image, clinical, label
        
        return image, label
