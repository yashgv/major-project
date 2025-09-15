import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retentive import VisRetNet
from asar import ASAR
from einops import rearrange, repeat


class S2VNet(nn.Module):
    """
    Subpixel spectral variability network for hyperspectral image classification
    """
    def __init__(self, band, num_classes, patch_size):
        super(S2VNet, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.band = band
        
        # ASAR module
        self.asar = ASAR(band, num_classes)
        
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
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1),  # Use padding=1 for better size handling
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

    def forward(self, x, output_abu=False):
        abu = self.unmix_encoder(x)
        re_unmix = self.unmix_decoder(abu)
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)
        feature_cls = self.cls(x) # cls token
        # abu sum-to-one and nonnegative constraint
        abu = abu.abs()
        abu = abu / abu.sum(1).unsqueeze(1)
        # endmember variability modeling
        edm_weight = self.unmix_decoder[0].weight.squeeze()
        # print("edm_weight shape:", edm_weight.shape)
        edm_var = self.var_encoder_share(edm_weight)  # [band*2, num_classes]
        # print("edm_var shape:", edm_var.shape)
        edm_var_1 = self.var_encoder_sep1(edm_var) # mu [band*2, z_dim]
        # print("edm_var_1 shape:", edm_var_1.shape)
        edm_var_2 = self.var_encoder_sep2(edm_var) # log_var [band*2, z_dim]
        # print("edm_var_2 shape:", edm_var_2.shape)
        edm_reparam = self.reparameterize(edm_var_1, edm_var_2)  # [band*2, z_dim]
        # print("edm_reparam shape:", edm_reparam.shape)
        # Process through var_decoder
        edm_var_de = self.var_decoder(edm_reparam)  # [band*2, num_classes*num_classes]
        # print("edm_var_de shape before reshape:", edm_var_de.shape)
        
        # Get the batch size from the input
        batch_size = int(edm_weight.shape[0])  # band*2
        num_classes = int(self.num_classes)
        
        # Reshape the output to [batch_size, num_classes, num_classes]
        try:
            edm_var_de = edm_var_de.view(batch_size, num_classes, num_classes)
        except RuntimeError as e:
            # print(f"Debug info - Current tensor shape: {edm_var_de.shape}")
            # print(f"Debug info - Attempting to reshape to: [{batch_size}, {num_classes}, {num_classes}]")
            # print(f"Debug info - Current tensor elements: {edm_var_de.numel()}")
            # print(f"Debug info - Required elements: {batch_size * num_classes * num_classes}")
            raise e
        # endmember perturbation modeling
        edm_per = self.perturb_encoder(edm_weight)
        edm_per_tensor = edm_per.view([batch_size, self.num_classes, 1])
        # update endmember results based on scaling term and perturbation term
        edm_weight_tensor = edm_weight.view([batch_size, self.num_classes, 1])
        edm_weight_new = torch.sigmoid(edm_var_de @ edm_weight_tensor + edm_per_tensor) # control endmember value into range [0,1]
        edm_weight_new = edm_weight_new.view([edm_weight.shape[0], self.num_classes, 1, 1])
        
        # Process through ASAR module
        with torch.no_grad():
            # Prepare inputs for ASAR
            batch_size = x.size(0)
            x_asar = x.permute(0, 2, 3, 1)  # [B, H, W, band]
            x_asar = x_asar.reshape(batch_size, -1, self.band)  # [B, H*W, band]
            
            # Reshape abundance maps: [B, num_classes, H, W] -> [B, H*W, num_classes]
            abu_reshaped = abu.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            
            # Process through ASAR module
            refined_features, refinement_weights, edm_refined = self.asar(
                x_asar.detach(), 
                abu_reshaped.detach(), 
                edm_weight_new.detach()
            )
            
            # Reshape refined features back to original format
            x_refined = refined_features.view(
                batch_size, 
                x.size(2),  # H
                x.size(3),  # W 
                self.band
            ).permute(0, 3, 1, 2).contiguous()  # [B, band, H, W]
            
            # Get dimensions
            band_dim = edm_weight_new.size(0) // 2  # Half of total bands
            
            # Process without interpolation if sizes match
            new_weight = edm_refined.squeeze(-1)  # [band*2, num_classes]
            if new_weight.size() == self.unmix_decoder[0].weight.size():
                self.unmix_decoder[0].weight.data.copy_(new_weight)
            else:
                # Create properly sized output tensor
                new_weight_resized = torch.zeros_like(self.unmix_decoder[0].weight)
                
                # Process weights with resizing
                src_bands = band_dim
                
                # Split weights into top and bottom halves
                weight_top = new_weight[:src_bands]  # [src_bands, num_classes]
                weight_bottom = new_weight[src_bands:]  # [src_bands, num_classes]
                
                # Split weights into top and bottom halves and reshape
                weight_top = weight_top.reshape(src_bands, -1)  # [src_bands, num_classes]
                weight_bottom = weight_bottom.reshape(src_bands, -1)  # [src_bands, num_classes]
                
                # print(f"Reshaped weight shapes - top: {weight_top.shape}, bottom: {weight_bottom.shape}")
                
                # Simple linear interpolation using dimension-preserved operations
                if src_bands != self.band:
                    # Create interpolation points
                    src_idx = torch.linspace(0, 1, src_bands).to(new_weight.device)
                    tgt_idx = torch.linspace(0, 1, self.band).to(new_weight.device)
                    
                    # Calculate interpolation weights and indices
                    tgt_idx_scaled = tgt_idx * (src_bands - 1)
                    idx_low = tgt_idx_scaled.floor().long()
                    idx_high = (idx_low + 1).clamp(max=src_bands - 1)
                    frac = tgt_idx_scaled - idx_low.float()
                    
                    # Interpolate top weights
                    weight_top_low = weight_top[idx_low]  # [band, num_classes]
                    weight_top_high = weight_top[idx_high]  # [band, num_classes]
                    top_resized = weight_top_low + frac.unsqueeze(1) * (weight_top_high - weight_top_low)
                    
                    # Interpolate bottom weights
                    weight_bottom_low = weight_bottom[idx_low]  # [band, num_classes]
                    weight_bottom_high = weight_bottom[idx_high]  # [band, num_classes]
                    bottom_resized = weight_bottom_low + frac.unsqueeze(1) * (weight_bottom_high - weight_bottom_low)
                    
                    # print(f"Final shapes - top: {top_resized.shape}, bottom: {bottom_resized.shape}")
                else:
                    # No resizing needed
                    top_resized = weight_top
                    bottom_resized = weight_bottom
                
                # Print weight shapes before concatenation
                # print(f"Before cat - top_resized: {top_resized.shape}, bottom_resized: {bottom_resized.shape}")
                
                # Combine results
                new_weight_resized = torch.cat([top_resized, bottom_resized], dim=0)  # [band*2, num_classes]
                # print(f"After cat - new_weight_resized: {new_weight_resized.shape}")
                
                # Get target shape from decoder
                expected_shape = self.unmix_decoder[0].weight.shape  # [band*2, num_classes, 1, 1]
                # print(f"Target shape: {expected_shape}")
                
                # Reshape for convolutional layers
                # First reshape to [band*2, num_classes] to match expected dimensions
                weight_reshaped = new_weight_resized.view(expected_shape[0], -1)  # [band*2, 256] -> [band*2, num_classes]
                if weight_reshaped.size(1) != expected_shape[1]:
                    # If dimensions don't match, apply dimensionality reduction
                    weight_reshaped = F.linear(
                        weight_reshaped,  # [band*2, 256]
                        torch.randn(expected_shape[1], weight_reshaped.size(1)).to(weight_reshaped.device)  # [num_classes, 256]
                    )  # Result: [band*2, num_classes]
                
                # Now reshape to 4D
                weight_reshaped = weight_reshaped.view(expected_shape[0], expected_shape[1], 1, 1)
                # print(f"Final shape: {weight_reshaped.shape}")
                
                # Debug info for weight shapes
                # print(f"Weight reshaped shape: {weight_reshaped.shape}")
                # print(f"Unmix decoder weight shape: {self.unmix_decoder[0].weight.shape}")
                # print(f"Nonlinear decoder weight shape: {self.unmix_decoder_nonlinear[0].weight.shape}")
                
                # Update unmix decoder weights
                self.unmix_decoder[0].weight.data.copy_(weight_reshaped)
                
                # Handle nonlinear decoder weights
                nonlinear_weight = weight_reshaped[:self.band]  # Take first half
                # print(f"Initial nonlinear weight shape: {nonlinear_weight.shape}")
                
                # Get the target shape for nonlinear decoder
                target_shape = self.unmix_decoder_nonlinear[0].weight.shape
                # print(f"Target nonlinear shape: {target_shape}")
                
                # First flatten the weight to 2D
                nonlinear_flat = nonlinear_weight.view(self.band, -1)  # [band, num_classes*1*1]
                # print(f"Flattened nonlinear shape: {nonlinear_flat.shape}")
                
                # Project to correct dimension if needed
                if nonlinear_flat.size(1) != target_shape[1]:
                    proj_weight = torch.randn(target_shape[1], nonlinear_flat.size(1)).to(nonlinear_flat.device)
                    nonlinear_proj = F.linear(nonlinear_flat, proj_weight)  # [band, target_dim]
                    # print(f"Projected nonlinear shape: {nonlinear_proj.shape}")
                    nonlinear_final = nonlinear_proj.view(target_shape)
                else:
                    nonlinear_final = nonlinear_flat.view(target_shape)
                
                # print(f"Final nonlinear shape: {nonlinear_final.shape}")
                
                # Update nonlinear decoder weights
                self.unmix_decoder_nonlinear[0].weight.data.copy_(nonlinear_final)
        # reshape abu
        feature_abu = self.conv(abu)
        abu_v = feature_abu.reshape(x.shape[0], -1)

        # use endmember probability by computing center_pixel and endmember
        edm = edm_weight_new[0:self.band, :] + edm_weight_new[self.band:self.band*2, :]
        edm = edm.squeeze() # band * num_classes
        output_linear = re_unmix[:,0:self.band] + re_unmix[:,self.band:self.band*2]
        re_unmix_out = re_unmix_nonlinear + output_linear

        re_unmix_out = re_unmix_out.view([re_unmix.shape[0], self.band, -1])
        center_pixel = torch.mean(re_unmix_out, dim=-1)
      
        cos_value = torch.matmul(center_pixel, edm) # batch_size * num_classes
        edm_norm = torch.norm(center_pixel)
        center_pixel_norm = torch.norm(center_pixel)
        cos_value = cos_value / (edm_norm * center_pixel_norm)
        # fuse abu features and cls token
        feature_fuse = torch.cat([abu_v, feature_cls], dim=1)
        output_cls = self.fc(feature_fuse)
        output_cls = self.relu(output_cls)

        output_cls = torch.cat([output_cls, cos_value], dim=1)
        output_cls = self.fc_2(output_cls)

        if output_abu:
            return re_unmix_nonlinear, re_unmix, output_cls, feature_abu
        else:
            return re_unmix_nonlinear, re_unmix, output_cls, edm_var_1, edm_var_2, feature_abu, edm_per
