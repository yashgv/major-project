import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Enhanced Loss Functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(self, inputs, targets):
        # Ensure inputs is float32 and targets is long
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.long)
        
        ce_loss = self.criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class EnhancedLossFunction(nn.Module):
    def __init__(self, n_classes, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super(EnhancedLossFunction, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        self.consistency_weight = 0.5
        
    def forward(self, predictions, targets, reconstructed=None, original=None):
        # Classification loss with focal loss
        cls_loss = self.focal_loss(predictions, targets)
        
        # Add reconstruction loss if provided
        if reconstructed is not None and original is not None:
            recon_loss = F.mse_loss(reconstructed, original)
            cls_loss = cls_loss + 0.1 * recon_loss
        
        # Consistency regularization
        if self.training:
            consistency_loss = F.mse_loss(predictions, predictions.detach())
            cls_loss = cls_loss + self.consistency_weight * consistency_loss
            
        return cls_loss

# Advanced Data Augmentation
class HSIAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        
    def spectral_noise(self, x, std_range=(0.01, 0.05)):
        if random.random() < self.p:
            std = random.uniform(*std_range)
            noise = torch.randn_like(x) * std
            return x + noise
        return x
    
    def band_dropout(self, x, drop_prob=0.1):
        if random.random() < self.p:
            mask = torch.rand(x.shape[1]) > drop_prob
            return x * mask.to(x.device).view(1, -1, 1, 1)
        return x
    
    def spectral_shift(self, x, shift_range=(-2, 2)):
        if random.random() < self.p:
            shift = random.randint(*shift_range)
            return torch.roll(x, shifts=shift, dims=1)
        return x
    
    def mixup(self, x, y, alpha=0.2):
        if random.random() < self.p:
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index]
            mixed_y = lam * y + (1 - lam) * y[index]
            
            return mixed_x, mixed_y
        return x, y
        
    def __call__(self, x, y):
        x = self.spectral_noise(x)
        x = self.band_dropout(x)
        x = self.spectral_shift(x)
        x, y = self.mixup(x, y)
        return x, y

# Attention-Based Fusion
class AttentionFusion(nn.Module):
    def __init__(self, abundance_dim, correlation_dim, pixel_dim):
        super(AttentionFusion, self).__init__()
        hidden_dim = 128
        
        self.abundance_proj = nn.Linear(abundance_dim, hidden_dim)
        self.correlation_proj = nn.Linear(correlation_dim, hidden_dim)
        self.pixel_proj = nn.Linear(pixel_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.fusion_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim)
        )
        
    def forward(self, abundance, correlation, pixel):
        # Project each feature to common dimension
        a = self.abundance_proj(abundance)
        c = self.correlation_proj(correlation)
        p = self.pixel_proj(pixel)
        
        # Self attention on concatenated features
        features = torch.cat([a, c, p], dim=1)
        attended_features, _ = self.attention(features, features, features)
        
        # Fusion
        fused = self.fusion_net(attended_features)
        return fused
