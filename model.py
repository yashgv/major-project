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
        # unmixing module
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

        # pixel-level classifier
        self.cls = VisRetNet(in_chans=band, num_classes=num_classes, embed_dims=[32])
        # endmember variability modeling
        z_dim = 4
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
        self.var_encoder_share = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
        )
        self.var_encoder_share.apply(init_weights)
        self.var_encoder_sep1 = nn.Linear(num_classes, z_dim)
        self.var_encoder_sep2 = nn.Linear(num_classes, z_dim)
        print(f"Initializing var_decoder with z_dim={z_dim}, num_classes={num_classes}")
        intermediate_dim = 128
        output_dim = int(num_classes) * int(num_classes)
        self.var_decoder = nn.Sequential(
            nn.Linear(z_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, output_dim),
            nn.ReLU(),
        )
        self.var_decoder.apply(init_weights)

        self.perturb_encoder = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU(),
        )
        # fusion module
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )
        self.feature_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.feature_size, num_classes)

        self.fc_2 = nn.Linear(num_classes*2, num_classes)
        self.relu = nn.ReLU()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.num_classes, self.patch_size, self.patch_size))
            x = self.conv(x)
            _, c, w, h = x.size()
            return c * w * h + self.num_classes

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape).cuda()
        return mu + eps * std

    def compute_spectral_consistency_loss(self, x_i, v_i, neighborhood_size=3):
        """
        Compute spectral consistency regularization loss
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
                        
                    if i < x_i.shape[2] and j < x_i.shape[3]:  # Check bounds
                        neighbor_spectrum = x_i[b, :, i, j]  # [L]
                        neighbor_abundance = v_i[b, :, i, j]  # [P]
                        
                        # Compute spectral similarity using dot product and normalization
                        dot_product = torch.dot(central_spectrum, neighbor_spectrum)
                        central_norm = torch.norm(central_spectrum)
                        neighbor_norm = torch.norm(neighbor_spectrum)
                        spectral_similarity = dot_product / (central_norm * neighbor_norm + 1e-10)
                        
                        # If spectrally similar, abundances should be consistent
                        similarity_threshold = 0.8
                        sim_val = spectral_similarity.item()
                        if sim_val > similarity_threshold:
                            abundance_diff = torch.norm(central_abundance - neighbor_abundance, p=2)
                            weighted_diff = abundance_diff * sim_val
                            patch_loss += weighted_diff
                            neighbor_count += 1
            
            # Average over valid neighbors for this patch
            if neighbor_count > 0:
                patch_loss = patch_loss / neighbor_count
                consistency_loss += patch_loss
        
        # Average over batch
        consistency_loss = consistency_loss / batch_size if batch_size > 0 else consistency_loss
        return consistency_loss
        
    def forward(self, x, output_abu=False):
        # Ensure input is float32 for GPU efficiency
        x = x.float()
        # Unmixing
        abu = self.unmix_encoder(x)
        re_unmix = self.unmix_decoder(abu)
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)
        feature_cls = self.cls(x)  # cls token
        
        # Abu sum-to-one and nonnegative constraint
        abu = abu.abs()
        abu = abu / (abu.sum(1).unsqueeze(1) + 1e-10)
        
        # Endmember variability modeling
        edm_weight = self.unmix_decoder[0].weight.squeeze()
        edm_var = self.var_encoder_share(edm_weight)
        edm_var_1 = self.var_encoder_sep1(edm_var)  # mu
        edm_var_2 = self.var_encoder_sep2(edm_var)  # log_var
        edm_reparam = self.reparameterize(edm_var_1, edm_var_2)
        
        # Process through var_decoder
        edm_var_de = self.var_decoder(edm_reparam)
        batch_size = int(edm_weight.shape[0])
        num_classes = int(self.num_classes)
        edm_var_de = edm_var_de.view(batch_size, num_classes, num_classes)
        
        # Endmember perturbation modeling
        edm_per = self.perturb_encoder(edm_weight)
        edm_per_tensor = edm_per.view([batch_size, self.num_classes, 1])
        
        # Update endmember results
        edm_weight_tensor = edm_weight.view([batch_size, self.num_classes, 1])
        edm_weight_new = torch.sigmoid(edm_var_de @ edm_weight_tensor + edm_per_tensor)
        edm_weight_new = edm_weight_new.view([edm_weight.shape[0], self.num_classes, 1, 1])
        self.unmix_decoder[0].weight = nn.Parameter(edm_weight_new)
        
        # Feature extraction and fusion
        feature_abu = self.conv(abu)
        abu_v = feature_abu.reshape(x.shape[0], -1)
        
        # Compute endmember probability
        edm = edm_weight_new[0:self.band, :] + edm_weight_new[self.band:self.band*2, :]
        edm = edm.squeeze()
        output_linear = re_unmix[:,0:self.band] + re_unmix[:,self.band:self.band*2]
        re_unmix_out = re_unmix_nonlinear + output_linear
        
        re_unmix_out = re_unmix_out.view([re_unmix.shape[0], self.band, -1])
        center_pixel = torch.mean(re_unmix_out, dim=-1)
        
        cos_value = torch.matmul(center_pixel, edm)
        edm_norm = torch.norm(edm, dim=0, keepdim=True) + 1e-10
        center_pixel_norm = torch.norm(center_pixel, dim=1, keepdim=True) + 1e-10
        cos_value = cos_value / (edm_norm * center_pixel_norm)
        
        # Final classification
        feature_fuse = torch.cat([abu_v, feature_cls], dim=1)
        output_cls = self.fc(feature_fuse)
        output_cls = self.relu(output_cls)
        output_cls = torch.cat([output_cls, cos_value], dim=1)
        output_cls = self.fc_2(output_cls)
        
        # Compute SCR loss during training
        if self.training:
            scr_loss = self.compute_spectral_consistency_loss(x, abu)
        else:
            scr_loss = torch.tensor(0.0).to(x.device)
            
        if output_abu:
            return re_unmix_nonlinear, re_unmix, output_cls, feature_abu
        else:
            return re_unmix_nonlinear, re_unmix, output_cls, edm_var_1, edm_var_2, feature_abu, edm_per, scr_loss
