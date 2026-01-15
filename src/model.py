"""
Deep learning model for strep throat classification.
Uses a CNN backbone (ResNet) with optional clinical feature fusion.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StrepThroatClassifier(nn.Module):
    """
    Model for classifying strep throat from images and optional clinical features.
    
    Architecture:
    - CNN backbone (ResNet18) for image feature extraction
    - Optional clinical feature fusion via concatenation
    - Fully connected layers for classification
    """
    
    def __init__(self, num_clinical_features=7, use_clinical_features=True, pretrained=True):
        super(StrepThroatClassifier, self).__init__()
        
        self.use_clinical_features = use_clinical_features
        
        # Load pretrained ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get the feature dimension from ResNet (512 for ResNet18)
        image_feature_dim = 512
        
        # Calculate input dimension for classifier
        if use_clinical_features:
            classifier_input_dim = image_feature_dim + num_clinical_features
        else:
            classifier_input_dim = image_feature_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification
        )
        
    def forward(self, image, clinical_features=None):
        """
        Forward pass.
        
        Args:
            image: Batch of images [B, C, H, W]
            clinical_features: Optional batch of clinical features [B, num_features]
        """
        # Extract image features
        image_features = self.backbone(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        
        # Fuse with clinical features if available
        if self.use_clinical_features and clinical_features is not None:
            # Concatenate image and clinical features
            combined_features = torch.cat([image_features, clinical_features], dim=1)
        else:
            combined_features = image_features
        
        # Classify
        output = self.classifier(combined_features)
        
        return output
