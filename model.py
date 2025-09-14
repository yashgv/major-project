import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from retentive import VisRetNet
from einops import rearrange, repeat
import numpy as np
import math

from unmixing_dirichlet import DirichletUnmixing

class S2VNetU(nn.Module):
    """
    Uncertainty-Aware Subpixel Spectral Variability Network (S2VNet-U) for hyperspectral image classification
    """
    def __init__(self, band, num_classes, patch_size, dropout_p=0.2, mc_samples_train=8, mc_samples_eval=32):
        super(S2VNetU, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.band = band
        self.mc_samples_train = mc_samples_train
        self.mc_samples_eval = mc_samples_eval
        
        # Bayesian unmixing module
        self.unmixing = DirichletUnmixing(band, num_classes, dropout_p)
        
        # Decoder for reconstruction with batch normalization and dropout
        self.unmix_decoder = nn.Sequential(
            nn.Conv2d(num_classes, band*2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(band*2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p)
        )
        self.unmix_decoder_nonlinear = nn.Sequential(
            nn.Conv2d(band*2, band, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(band),
            nn.Dropout2d(p=dropout_p),
            nn.Sigmoid(),
            nn.Conv2d(band, band, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(band)
        )

        # Pixel-level classifier with dropout
        self.cls = VisRetNet(in_chans=band, num_classes=num_classes, embed_dims=[32])
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        # Risk-aware gating MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_classes + 1, 64),  # +1 for confidence score
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 1)
        )
        self.gate_mlp.apply(init_weights)
        
        # Feature fusion module
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p)
        )
        
        # Compute feature size for FC layers
        self.feature_size = self._get_final_flattened_size()
        
        # Compute feature dimensions
        with torch.no_grad():
            x = torch.zeros((1, num_classes, patch_size, patch_size))
            x = self.conv(x)
            _, c, h, w = x.size()
            self.feature_size = c * h * w
            
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(num_classes * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

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

    def forward(self, x, is_training=None):
        if is_training is None:
            is_training = self.training
            
        # Get unmixing results with uncertainty
        unmixing_out = self.unmixing(x, num_samples=self.mc_samples_train if is_training else self.mc_samples_eval)
        abu_mean = unmixing_out['abundance_mean']
        confidence = unmixing_out['confidence']
        
        # Reconstruction path
        re_unmix = self.unmix_decoder(abu_mean)
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)
        
        # Get pixel-level features
        feature_cls = self.cls(x)  # cls token
        
        # Prepare gating inputs
        v_sub_stats = torch.mean(abu_mean, dim=(2, 3))  # Global abundance statistics [B, C]
        confidence_flat = confidence.view(confidence.size(0), -1).mean(dim=1, keepdim=True)  # [B, 1]
        g_input = torch.cat([confidence_flat, v_sub_stats], dim=1)  # [B, C+1]
        
        # Compute risk-aware gate
        g = torch.sigmoid(self.gate_mlp(g_input))  # [B, 1]
        
        # Process abundance features
        feature_abu = self.conv(abu_mean)  # [B, C, H', W']
        abu_v = feature_abu.reshape(x.shape[0], -1)  # [B, C*H'*W']
        
        # Ensure feature_cls matches abu_v dimensions
        feature_cls = feature_cls.view(feature_cls.size(0), -1)  # [B, num_classes]
        
        # Make abu_v match feature_cls size if needed
        if abu_v.size(1) != feature_cls.size(1):
            # Add a projection layer if needed
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(abu_v.size(1), feature_cls.size(1)).cuda()
            abu_v = self.projection(abu_v)
            
        # Process features through spatial pooling
        feature_abu = self.conv(abu_mean)  # [B, C, H', W']
        abu_v = feature_abu.view(feature_abu.size(0), -1)  # [B, C*H'*W']
        
        # Project abundance features to match classifier output size
        abu_v = self.fc(abu_v)  # [B, num_classes]
        
        # Compute gated fusion
        g = g.view(-1, 1)  # Ensure correct broadcasting
        feature_gated = g * abu_v + (1 - g) * feature_cls
        
        # Final classification
        final_feature = torch.cat([feature_gated, abu_v], dim=1)  # [B, num_classes*2]
        logits = self.fc_2(final_feature)  # [B, num_classes]
        
        if is_training:
            return {
                'reconstruction': re_unmix_nonlinear,
                'reconstruction_full': re_unmix,
                'logits': logits,
                'abundances': unmixing_out,
                'gate_value': g,
                'feature_abu': feature_abu,
                'confidence': confidence
            }
        else:
            return {
                'reconstruction': re_unmix_nonlinear,
                'logits': logits,
                'abundances': unmixing_out,
                'confidence': confidence
            }
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
        self.unmix_decoder[0].weight = nn.Parameter(edm_weight_new)
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
