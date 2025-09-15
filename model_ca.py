import torch
import torch.nn as nn
import torch.nn.functional as F
from model import S2VNet
from cross_attention import SpectralPixelCrossAttention

class S2VNet_CA(S2VNet):
    """
    S2VNet with Cross Attention enhancement
    """
    def __init__(self, band, num_classes, patch_size):
        super(S2VNet_CA, self).__init__(band, num_classes, patch_size)
        
        # Add Cross Attention Module
        self.cross_attention = SpectralPixelCrossAttention(
            num_classes=num_classes,  # P (number of endmembers/classes)
            pixel_dim=32,           # C (pixel feature dimension from cls)
            hidden_dim=64,
            num_heads=4
        )
        
        # Add attention regularization weight
        self.lambda5 = 0.01  # Cross attention regularization weight
    
    def compute_attention_regularization_loss(self, enhanced_spectral, original_abundances):
        """
        Regularization to prevent cross attention from destroying abundance constraints
        """
        # Ensure enhanced spectral features maintain abundance-like properties
        enhanced_abundances = F.softmax(enhanced_spectral, dim=1)  # Apply softmax to maintain sum-to-one
        
        # L2 regularization to keep enhanced features close to original abundances
        reg_loss = F.mse_loss(enhanced_abundances, original_abundances.mean(dim=[2,3]))
        
        return reg_loss

    def forward(self, x, output_abu=False):
        # First get abundances from unmixing network
        abu = self.unmix_encoder(x)  # [B, num_classes, H, W]
        abu = abu.abs()  # Ensure non-negativity
        abu = abu / abu.sum(1).unsqueeze(1)  # Ensure sum-to-one
        
        # Unmixing decoder
        re_unmix = self.unmix_decoder(abu)  # [B, band*2, H, W]
        # Split the bands into two parts
        band = re_unmix.shape[1] // 2
        re_unmix_1 = re_unmix[:, :band]  # First half
        re_unmix_2 = re_unmix[:, band:]  # Second half
        re_unmix_sum = re_unmix_1 + re_unmix_2  # [B, band, H, W]
        
        # Apply nonlinear decoder
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)  # [B, band, H, W]
        
        # Final reconstruction is sum of linear and nonlinear parts
        re_unmix_full = re_unmix_sum + re_unmix_nonlinear  # [B, band, H, W]
        
        # Get pixel-level features
        feature_cls = self.cls(x)  # Get features from classifier
        
        # Process abundances for cross attention
        B = x.shape[0]
        v_flat = abu.view(B, self.num_classes, -1).mean(dim=-1)  # [B, num_classes]
        
        # Prepare pixel features for cross attention
        F_flat = feature_cls.unsqueeze(1).expand(-1, self.patch_size * self.patch_size, -1)  # [B, H*W, num_classes]
        
        # Apply cross attention
        enhanced_spectral, enhanced_pixel = self.cross_attention(v_flat, F_flat)
        
        # Reshape enhanced pixel features back to spatial dimensions
        enhanced_pixel_spatial = enhanced_pixel.view(B, self.patch_size, self.patch_size, self.num_classes).permute(0, 3, 1, 2)
        
        # Process abundances with enhancement
        abu_enhanced = abu + enhanced_spectral.view(B, self.num_classes, 1, 1).expand_as(abu)
        feature_abu = self.conv(abu_enhanced)
        abu_v = feature_abu.reshape(B, -1)
        
        # Apply endmember variability modeling
        edm_weight = self.unmix_decoder[0].weight.squeeze()
        edm_var = self.var_encoder_share(edm_weight)
        edm_var_1 = self.var_encoder_sep1(edm_var)
        edm_var_2 = self.var_encoder_sep2(edm_var)
        edm_reparam = self.reparameterize(edm_var_1, edm_var_2)
        edm_var_de = self.var_decoder(edm_reparam)
        
        # Get batch size and reshape endmember variance
        batch_size = int(edm_weight.shape[0])  # band*2
        edm_var_de = edm_var_de.view(batch_size, self.num_classes, self.num_classes)
        
        # Endmember perturbation modeling
        edm_per = self.perturb_encoder(edm_weight)
        edm_per_tensor = edm_per.view([batch_size, self.num_classes, 1])
        
        # Update endmember results based on scaling term and perturbation term
        edm_weight_tensor = edm_weight.view([batch_size, self.num_classes, 1])
        edm_weight_new = torch.sigmoid(edm_var_de @ edm_weight_tensor + edm_per_tensor)
        edm_weight_new = edm_weight_new.view([edm_weight.shape[0], self.num_classes, 1, 1])
        self.unmix_decoder[0].weight = nn.Parameter(edm_weight_new)

        # Fuse features
        feature_fuse = torch.cat([abu_v, enhanced_pixel_spatial.mean(dim=[2, 3])], dim=1)
        output_cls = self.fc(feature_fuse)
        output_cls = self.relu(output_cls)
        
        # Add spectral cosine similarity
        cos_value = self.compute_spectral_cosine(x, re_unmix, re_unmix_nonlinear)
        output_cls = torch.cat([output_cls, cos_value], dim=1)
        output_cls = self.fc_2(output_cls)

        if output_abu:
            return re_unmix_full, re_unmix, output_cls, feature_abu
        else:
            return (
                re_unmix_full,
                re_unmix,
                output_cls,
                edm_var_1,
                edm_var_2,
                feature_abu,
                edm_per,
                enhanced_spectral  # Return enhanced_spectral for regularization
            )

    def compute_spectral_cosine(self, x, re_unmix, re_unmix_nonlinear):
        # Extract relevant dimensions
        edm_weight = self.unmix_decoder[0].weight.squeeze()
        edm = edm_weight[0:self.band, :] + edm_weight[self.band:self.band*2, :]
        edm = edm.squeeze()  # band * num_classes

        output_linear = re_unmix[:,0:self.band] + re_unmix[:,self.band:self.band*2]
        re_unmix_out = re_unmix_nonlinear + output_linear
        re_unmix_out = re_unmix_out.view([re_unmix.shape[0], self.band, -1])
        center_pixel = torch.mean(re_unmix_out, dim=-1)
        
        cos_value = torch.matmul(center_pixel, edm)  # batch_size * num_classes
        edm_norm = torch.norm(edm, dim=0)
        center_pixel_norm = torch.norm(center_pixel, dim=1, keepdim=True)
        cos_value = cos_value / (edm_norm * center_pixel_norm + 1e-8)
        
        return cos_value
