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
    """
    def __init__(self, band, num_classes, patch_size):
        super(S2VNet, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.band = band
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
