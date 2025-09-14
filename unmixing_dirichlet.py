import torch
import torch.nn as nn
import torch.nn.functional as F

class DirichletUnmixing(nn.Module):
    """
    Bayesian unmixing module that outputs Dirichlet abundances
    """
    def __init__(self, band, num_classes, dropout_p=0.2):
        super(DirichletUnmixing, self).__init__()
        self.band = band
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        # Log-alpha predictor
        self.log_alpha_predictor = nn.Sequential(
            nn.Conv2d(num_classes, num_classes*2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(num_classes*2, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x, num_samples=1):
        # Initial encoding
        hidden = self.encoder(x)
        
        # Get log alpha parameters
        log_alpha = self.log_alpha_predictor(hidden)
        
        # Convert to alpha (ensure positivity)
        alpha = F.softplus(log_alpha) + 1e-6
        
        # Compute concentration and abundance mean
        concentration = alpha.sum(dim=1, keepdim=True)
        a_mean = alpha / alpha.sum(dim=1, keepdim=True)
        
        # Compute entropy for confidence
        entropy = -torch.sum(a_mean * torch.log(a_mean + 1e-8), dim=1)
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=x.device))
        
        # Compute confidence scores
        conf_score = torch.sigmoid((concentration - concentration.mean()) / (concentration.std() + 1e-8))
        conf_from_entropy = 1.0 - (entropy / max_entropy)
        confidence = 0.5 * conf_score + 0.5 * conf_from_entropy

        # For training mode, return multiple samples if requested
        if self.training and num_samples > 1:
            samples = []
            for _ in range(num_samples):
                # Re-run through dropout layers
                hidden = self.encoder(x)
                log_alpha_t = self.log_alpha_predictor(hidden)
                alpha_t = F.softplus(log_alpha_t) + 1e-6
                a_t = alpha_t / alpha_t.sum(dim=1, keepdim=True)
                samples.append(a_t)
            samples = torch.stack(samples, dim=0)
            return {
                'alpha': alpha,
                'abundance_mean': a_mean,
                'concentration': concentration,
                'confidence': confidence,
                'samples': samples
            }
        
        return {
            'alpha': alpha,
            'abundance_mean': a_mean, 
            'concentration': concentration,
            'confidence': confidence
        }
