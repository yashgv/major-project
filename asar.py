import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SpectralAttention(nn.Module):
    def __init__(self, band_dim):
        super(SpectralAttention, self).__init__()
        hidden_dim = band_dim // 4
        self.attention = nn.Sequential(
            nn.Linear(band_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, band_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [B, H*W, band]
        batch_size, hw, band = x.shape
        x_flat = x.reshape(-1, band)  # [B*H*W, band]
        
        # Compute attention weights
        att = self.attention(x_flat)  # [B*H*W, band]
        
        # Reshape back and apply attention
        att = att.view(batch_size, hw, band)  # [B, H*W, band]
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7], patch_size=None):
        super(SpatialAttention, self).__init__()
        self.branches = nn.ModuleList()
        self.hidden_dim = 32
        
        for k in kernel_sizes:
            branch = nn.Module()
            self.branches.append(branch)
            
    def forward(self, x):
        # x shape: [B, H*W, band]
        B, HW, C = x.shape
        patch_size = int(math.sqrt(HW))  # Dynamically calculate patch size
        
        # Create linear layers with correct dimensions for current patch size
        for branch in self.branches:
            if not hasattr(branch, 'input_linear'):
                branch.input_linear = nn.Linear(patch_size * patch_size, self.hidden_dim).to(x.device)
                branch.output_linear = nn.Linear(self.hidden_dim, patch_size * patch_size).to(x.device)
                branch.relu = nn.ReLU().to(x.device)
                branch.sigmoid = nn.Sigmoid().to(x.device)
        
        # Reshape input for spatial attention
        x_spatial = x.transpose(1, 2).view(B, C, patch_size, patch_size)  # [B, band, H, W]
        
        # Compute pooled features
        avg_pool = torch.mean(x_spatial, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool = torch.max(x_spatial, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        
        # Process through multiple scales
        att_weights = []
        for branch in self.branches:
            # Flatten pooled features
            avg_flat = avg_pool.squeeze(1).reshape(B, -1)  # [B, H*W]
            max_flat = max_pool.squeeze(1).reshape(B, -1)  # [B, H*W]
            
            # Process through linear layers and activation functions
            avg_hidden = branch.input_linear(avg_flat)  # [B, hidden_dim]
            max_hidden = branch.input_linear(max_flat)  # [B, hidden_dim]
            
            # Apply ReLU and Sigmoid
            avg_hidden = branch.relu(avg_hidden)  # [B, hidden_dim]
            max_hidden = branch.relu(max_hidden)  # [B, hidden_dim]
            
            # Project back to spatial dimensions
            avg_att = branch.output_linear(avg_hidden)  # [B, H*W]
            max_att = branch.output_linear(max_hidden)  # [B, H*W]
            
            # Apply final sigmoid
            avg_att = branch.sigmoid(avg_att)  # [B, H*W]
            max_att = branch.sigmoid(max_att)  # [B, H*W]
            
            # Combine attentions
            att = avg_att + max_att
            att_weights.append(att.view(B, -1, 1))  # [B, H*W, 1]
            
        # Combine multi-scale attention weights
        att = torch.mean(torch.stack(att_weights, dim=0), dim=0)  # [B, H*W, 1]
        att = torch.sigmoid(att)
        
        return x * att

class EndmemberRefinementModule(nn.Module):
    def __init__(self, num_classes, band_dim):
        super(EndmemberRefinementModule, self).__init__()
        
        self.band_dim = band_dim
        self.num_classes = num_classes
        
        # Process endmembers
        hidden_dim = 128
        self.endmember_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(band_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, band_dim)
            ) for _ in range(num_classes)
        ])
        
        # Process abundances
        self.abundance_net = nn.Sequential(
            nn.Linear(num_classes, band_dim),
            nn.ReLU(),
            nn.Linear(band_dim, band_dim),
            nn.Sigmoid()
        )
        
        # Refinement network
        self.refinement_net = nn.Sequential(
            nn.Linear(band_dim, band_dim),
            nn.ReLU(),
            nn.Linear(band_dim, band_dim)
        )
    
    def forward(self, endmembers, abundances):
        batch_size = endmembers.size(0)
        band_dim = endmembers.size(1)
        num_classes = abundances.size(1)
        
        # Process each endmember class independently
        endmembers_list = []
        for i in range(num_classes):
            # Extract endmembers for current class
            endm = endmembers[:, :, i, :].squeeze(-1)  # [B, band_dim]
            # Process through corresponding network
            refined = self.endmember_net[i](endm)  # [B, band_dim]
            endmembers_list.append(refined.unsqueeze(2))  # [B, band_dim, 1]
            
        # Concatenate refined endmembers
        endmembers_refined = torch.cat(endmembers_list, dim=2)  # [B, band_dim, num_classes]
        
        # Process abundances to generate attention weights
        attention_weights = self.abundance_net(abundances)  # [B, band_dim]
        attention_weights = attention_weights.unsqueeze(-1)  # [B, band_dim, 1]
        
        # Apply refinement
        refinement = self.refinement_net(attention_weights.squeeze(-1))  # [B, band_dim]
        refinement = refinement.unsqueeze(-1)  # [B, band_dim, 1]
        
        # Apply attention-based refinement
        output = endmembers_refined + refinement * endmembers_refined
        
        # Restore original shape
        output = output.unsqueeze(-1)  # [B, band_dim, num_classes, 1]
        return output

class ASAR(nn.Module):
    def __init__(self, band_dim, num_classes, patch_size=7):
        super(ASAR, self).__init__()
        
        # Spectral attention module
        self.spectral_attention = SpectralAttention(band_dim)
        
        # Spatial attention module with multiple scales
        self.spatial_attention = SpatialAttention(patch_size=patch_size)
        
        # Feature fusion
        fusion_dim = band_dim
        self.fusion = nn.Sequential(
            nn.Linear(band_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, band_dim)
        )
        
        # Endmember refinement module
        hidden_dim = 128
        self.endm_refine = nn.Sequential(
            nn.Linear(band_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x, abundances, endmembers):
        # x shape: [B, H*W, band]
        batch_size, hw, band_dim = x.shape
        
        # Apply spectral attention
        x_spectral = self.spectral_attention(x)
        
        # Apply spatial attention
        x_spatial = self.spatial_attention(x)
        
        # Fuse features
        x_fused = x_spectral + x_spatial
        x_refined = self.fusion(x_fused)
        
        # Generate attention weights for endmember refinement
        # Average features spatially and concatenate with abundance information
        mean_features = torch.mean(x_refined, dim=1)  # [B, band]
        mean_abundances = torch.mean(abundances, dim=1)  # [B, num_classes]
        
        # Concatenate features and abundances
        combined_features = torch.cat([mean_features, mean_abundances], dim=1)
        
        # Generate refinement weights
        refinement_weights = self.endm_refine(combined_features)  # [B, num_classes]
        
        # Reshape refinement weights for endmember update
        refinement_weights = refinement_weights.unsqueeze(1)  # [B, 1, num_classes]
        
        # Apply refinement to endmembers
        band_dim = endmembers.size(0) // 2
        num_classes = endmembers.size(1)
        
        # Reshape refinement weights for broadcasting
        refinement_weights = refinement_weights.view(batch_size, num_classes)  # [B, num_classes]
        
        # Split endmembers into top and bottom parts
        endm_top = endmembers[:band_dim]  # [band, num_classes, 1]
        endm_bottom = endmembers[band_dim:]  # [band, num_classes, 1]
        
        # Apply refinement to each part
        endm_top_refined = endm_top * refinement_weights.mean(0).unsqueeze(-1)  # [band, num_classes, 1]
        endm_bottom_refined = endm_bottom * refinement_weights.mean(0).unsqueeze(-1)  # [band, num_classes, 1]
        
        # Concatenate refined parts back together
        endmembers_refined = torch.cat([endm_top_refined, endm_bottom_refined], dim=0)  # [band*2, num_classes, 1]
        
        return x_refined, refinement_weights, endmembers_refined
