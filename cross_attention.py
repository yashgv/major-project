import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralPixelCrossAttention(nn.Module):
    """
    Cross attention between spectral endmember features and pixel-level features
    """
    def __init__(self, num_classes, pixel_dim=32, hidden_dim=64, num_heads=4):
        super(SpectralPixelCrossAttention, self).__init__()
        
        self.num_classes = num_classes  # P (number of endmembers/classes)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projection layers
        self.input_proj = nn.Linear(num_classes, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        
        # Output projections
        self.output_proj_spectral = nn.Linear(hidden_dim, num_classes)
        self.output_proj_pixel = nn.Linear(hidden_dim, num_classes)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, spectral_features, pixel_features):
        """
        Args:
            spectral_features: [batch_size, num_classes] - abundance/endmember features
            pixel_features: [batch_size, H*W, num_classes] - pixel-level features
        Returns:
            enhanced_spectral: [batch_size, num_classes]
            enhanced_pixel: [batch_size, H*W, num_classes]
        """
        # Project both inputs to hidden dimension
        spectral_proj = self.input_proj(spectral_features).unsqueeze(1)  # [B, 1, H]
        pixel_proj = self.input_proj(pixel_features)  # [B, HW, H]
        
        # Cross attention: spectral attends to pixel
        spectral_out, _ = self.attention(spectral_proj, pixel_proj, pixel_proj)
        spectral_out = spectral_out.squeeze(1)  # [B, H]
        
        # Cross attention: pixel attends to spectral
        pixel_out, _ = self.attention(pixel_proj, spectral_proj, spectral_proj)  # [B, HW, H]
        
        # Project back to original dimensions
        enhanced_spectral = self.output_proj_spectral(spectral_out)  # [B, num_classes]
        enhanced_pixel = self.output_proj_pixel(pixel_out)  # [B, HW, num_classes]
        
        # Residual connection and normalization
        enhanced_spectral = spectral_features + self.dropout(enhanced_spectral)
        enhanced_pixel = pixel_features + self.dropout(enhanced_pixel)
        
        return enhanced_spectral, enhanced_pixel
