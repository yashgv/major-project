import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from retentive import VisRetNet
from einops import rearrange, repeat
import numpy as np
import math
from enhanced_modules import AttentionFusion

class SpectralResponseNetwork(nn.Module):
    def __init__(self, n_bands, n_endmembers):
        super(SpectralResponseNetwork, self).__init__()
        self.spectral_response = nn.Sequential(
            nn.Linear(n_bands, n_bands),
            nn.ReLU(),
            nn.Linear(n_bands, n_bands),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Handle the input shape: [B, H, W, C] -> [B, C, H, W]
        if x.dim() == 4 and x.shape[3] == self.band:
            x = x.permute(0, 3, 1, 2)
        
        # Reshape for pixel-wise processing: [B, C, H, W] -> [B*H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(-1, self.band)
        
        # Apply spectral unmixing
        x = self.unmix(x)  # [B*H*W, num_classes]
        
        # Global average pooling
        x = x.view(batch_size, -1, self.num_classes)  # [B, H*W, num_classes]
        x = x.mean(dim=1)  # [B, num_classes]
        
        return x
        
        # Reshape for per-pixel processing
        x = x.permute(0, 2, 3, 1).reshape(-1, c)  # [batch_size*height*width, n_bands]
        
        # Apply spectral response
        x = self.spectral_response(x)
        
        # Reshape back
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x

class AdaptivePerturbationModule(nn.Module):
    def __init__(self, n_bands):
        super(AdaptivePerturbationModule, self).__init__()
        self.perturbation_predictor = nn.Sequential(
            nn.Linear(n_bands, n_bands//2),
            nn.ReLU(),
            nn.Linear(n_bands//2, n_bands),
            nn.Tanh()  # Bounded perturbations
        )
        self.uncertainty_estimator = nn.Linear(n_bands, 1)
        
    def forward(self, x):
        perturbation = self.perturbation_predictor(x)
        uncertainty = self.uncertainty_estimator(x).sigmoid()
        return x + uncertainty * perturbation

class EnhancedS2VNet(nn.Module):
    def __init__(self, band, num_classes, patch_size):
        super(EnhancedS2VNet, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.band = band

        # Enhanced Unmixing Module
        hidden_dim = band // 2
        self.unmix = nn.Sequential(
            nn.Linear(band, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Convert input to float32 and handle shape
        x = x.to(torch.float32)
        if x.dim() == 4 and x.shape[3] == self.band:
            x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            
        # Reshape for pixel-wise processing
        x = x.permute(0, 2, 3, 1).reshape(-1, self.band)  # [B*H*W, C]
        
        # Apply spectral unmixing and ensure float32
        x = self.unmix(x).float()  # [B*H*W, num_classes]
        
        # Global average pooling
        x = x.view(batch_size, -1, self.num_classes)  # [B, H*W, num_classes]
        x = x.mean(dim=1)  # [B, num_classes]
        
        return x
