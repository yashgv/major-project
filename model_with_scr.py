import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from retentive import VisRetNet
from einops import rearrange, repeat
import numpy as np
import math

class S2VNet(nn.Module):
    """
    Subpixel spectral variability network for hyperspectral image classification
    with Spectral Consistency Regularization (SCR)
    """
    def __init__(self, band, num_classes, patch_size, lambda4=0.005):
        super(S2VNet, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.band = band
        self.lambda4 = lambda4  # SCR weight
        
        # Original modules
        self.unmix_encoder = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2),
            nn.ReLU(),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4),
            nn.ReLU(),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.unmix_decoder = nn.Sequential(
            nn.Conv2d(num_classes, band*2, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.unmix_decoder_nonlinear = nn.Sequential(
            nn.Conv2d(band*2, band, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(band, band, kernel_size=1, stride=1, bias=True),
        )

        # Rest of the original initialization code...
        [... Previous initialization code remains unchanged ...]

    def compute_spectral_consistency_loss(self, x_i, v_i, neighborhood_size=3):
        """
        Compute spectral consistency regularization loss
        
        Args:
            x_i: Input HSI patch [B x L x H x W]
            v_i: Abundance maps [B x P x H x W] 
            neighborhood_size: Size of spatial neighborhood to consider
        """
        batch_size = x_i.shape[0]
        consistency_loss = 0.0
        
        for b in range(batch_size):
            center = neighborhood_size // 2
            central_spectrum = x_i[b, :, center, center]  # [L]
            central_abundance = v_i[b, :, center, center]  # [P]
            
            neighbor_count = 0
            patch_loss = 0.0
            
            # Iterate through spatial neighborhood
            for i in range(neighborhood_size):
                for j in range(neighborhood_size):
                    if i == center and j == center:
                        continue
                        
                    neighbor_spectrum = x_i[b, :, i, j]  # [L]
                    neighbor_abundance = v_i[b, :, i, j]  # [P]
                    
                    # Compute spectral similarity using cosine similarity
                    spectral_similarity = F.cosine_similarity(
                        central_spectrum.unsqueeze(0), 
                        neighbor_spectrum.unsqueeze(0), 
                        dim=0
                    )
                    
                    # If spectrally similar (similarity > threshold), abundances should be consistent
                    similarity_threshold = 0.8
                    if spectral_similarity > similarity_threshold:
                        abundance_diff = torch.norm(central_abundance - neighbor_abundance, p=2)
                        # Weight by spectral similarity - more similar spectra should have more similar abundances
                        weighted_diff = abundance_diff * spectral_similarity
                        patch_loss += weighted_diff
                        neighbor_count += 1
            
            # Average over valid neighbors for this patch
            if neighbor_count > 0:
                patch_loss = patch_loss / neighbor_count
                consistency_loss += patch_loss
        
        # Average over batch
        consistency_loss = consistency_loss / batch_size
        return consistency_loss

    def forward(self, x, output_abu=False):
        # Original forward pass code
        abu = self.unmix_encoder(x)
        re_unmix = self.unmix_decoder(abu)
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)
        feature_cls = self.cls(x)  # cls token
        
        # Rest of the original forward pass remains the same
        [... Previous forward pass code remains unchanged ...]
        
        # Compute SCR loss if we're training
        if self.training:
            scr_loss = self.compute_spectral_consistency_loss(x, abu)
        else:
            scr_loss = torch.tensor(0.0).to(x.device)
        
        if output_abu:
            return re_unmix_nonlinear, re_unmix, output_cls, feature_abu
        else:
            return re_unmix_nonlinear, re_unmix, output_cls, edm_var_1, edm_var_2, feature_abu, edm_per, scr_loss
