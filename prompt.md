Follow the instruction as given below and update the code base accordingly and test it by running the python demo.py do not create the venv instead directly run the python file

Follow the instructions:
# S2VNet Accuracy Improvement Implementation Guide

## OBJECTIVE
Enhance the existing S2VNet (Subpixel Spectral Variability Network) codebase to achieve accuracy improvements of 2-5% over the baseline results. Focus on systematic architectural and training enhancements while maintaining the core subpixel variability modeling approach.

## BASELINE UNDERSTANDING
- **Current Architecture**: Nonlinear AE Unmixing + Pixel-level CNN-Transformer + Enhanced Subpixel Fusion
- **Core Strengths**: Handles mixed pixels and spectral heterogeneity via endmember-abundance modeling
- **Target**: Improve classification accuracy while preserving subpixel interpretability

## STEP-BY-STEP IMPROVEMENT STRATEGY

### Phase 1: Enhanced Autoencoder Unmixing Network

#### 1.1 Advanced Spectral Variability Modeling
```python
# IMPLEMENTATION: Replace simple scaling factors with learnable spectral response functions
class SpectralResponseNetwork(nn.Module):
    def __init__(self, n_bands, n_endmembers):
        # Add wavelength-dependent scaling using 1D convolutions
        self.spectral_response = nn.Conv1d(n_endmembers, n_endmembers, 
                                         kernel_size=5, padding=2, groups=n_endmembers)
        # Add atmospheric correction module
        self.atm_correction = nn.Sequential(
            nn.Linear(n_bands, n_bands),
            nn.ReLU(),
            nn.Linear(n_bands, n_bands)
        )
```

**ACTION**: Replace static scaling factors with learnable spectral response functions that model wavelength-dependent variations more accurately.

#### 1.2 Multi-Resolution Endmember Learning
```python
# IMPLEMENTATION: Learn endmembers at multiple spectral resolutions
class MultiResEndmemberExtractor(nn.Module):
    def __init__(self, n_bands, n_endmembers):
        # Full resolution endmembers
        self.full_res_endmembers = nn.Parameter(torch.randn(n_endmembers, n_bands))
        # Coarse resolution (every 4th band)
        self.coarse_endmembers = nn.Parameter(torch.randn(n_endmembers, n_bands//4))
        # Fusion network
        self.fusion_net = nn.Linear(n_bands + n_bands//4, n_bands)
```

**ACTION**: Implement multi-resolution endmember learning to capture both fine-grained and broad spectral characteristics.

#### 1.3 Adaptive Perturbation Modeling
```python
# IMPLEMENTATION: Make perturbations content-aware rather than random
class AdaptivePerturbationModule(nn.Module):
    def __init__(self, n_bands):
        self.perturbation_predictor = nn.Sequential(
            nn.Linear(n_bands, n_bands//2),
            nn.ReLU(),
            nn.Linear(n_bands//2, n_bands),
            nn.Tanh()  # Bounded perturbations
        )
        self.uncertainty_estimator = nn.Linear(n_bands, 1)
```

**ACTION**: Replace random perturbation terms with adaptive, content-aware perturbations predicted from the input spectrum.

### Phase 2: Enhanced Pixel-Level Classifier

#### 2.1 Advanced CNN-Transformer Hybrid
```python
# IMPLEMENTATION: Replace standard CNN with EfficientNet-style compound scaling
class AdvancedCNNBackbone(nn.Module):
    def __init__(self, in_channels):
        # Depthwise separable convolutions for efficiency
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding=1),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                SEBlock(out_channels)  # Squeeze-and-Excitation
            ) for out_channels in [32, 64, 128]
        ])
```

**ACTION**: Upgrade CNN backbone with depthwise separable convolutions, squeeze-and-excitation blocks, and compound scaling.

#### 2.2 Spectral-Spatial Transformer Enhancement
```python
# IMPLEMENTATION: Add spectral attention mechanism to transformer
class SpectralSpatialTransformer(nn.Module):
    def __init__(self, dim, n_bands):
        self.spatial_attention = MultiHeadAttention(dim, num_heads=8)
        self.spectral_attention = MultiHeadAttention(n_bands, num_heads=4)
        self.cross_attention = MultiHeadAttention(dim, num_heads=8)
        
    def forward(self, spatial_features, spectral_features):
        # Self-attention on spatial features
        spatial_attended = self.spatial_attention(spatial_features)
        # Self-attention on spectral features
        spectral_attended = self.spectral_attention(spectral_features)
        # Cross-attention between spatial and spectral
        fused = self.cross_attention(spatial_attended, spectral_attended)
        return fused
```

**ACTION**: Implement spectral-spatial cross-attention to better model relationships between spatial and spectral domains.

#### 2.3 Progressive Feature Refinement
```python
# IMPLEMENTATION: Add progressive feature refinement stages
class ProgressiveFeatureRefinement(nn.Module):
    def __init__(self, feature_dims):
        self.refinement_stages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for dim in feature_dims
        ])
        self.skip_connections = nn.ModuleList([
            nn.Linear(dim, dim) for dim in feature_dims
        ])
```

**ACTION**: Add progressive refinement with residual connections to gradually improve feature quality.

### Phase 3: Advanced Fusion Strategy

#### 3.1 Attention-Based Fusion
```python
# IMPLEMENTATION: Replace simple concatenation with attention-based fusion
class AttentionFusion(nn.Module):
    def __init__(self, abundance_dim, correlation_dim, pixel_dim):
        self.abundance_proj = nn.Linear(abundance_dim, 128)
        self.correlation_proj = nn.Linear(correlation_dim, 128)
        self.pixel_proj = nn.Linear(pixel_dim, 128)
        
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        self.fusion_net = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
```

**ACTION**: Implement attention-based fusion that learns optimal weighting of abundance, correlation, and pixel-level features.

#### 3.2 Uncertainty-Aware Fusion
```python
# IMPLEMENTATION: Add uncertainty estimation to fusion process
class UncertaintyAwareFusion(nn.Module):
    def __init__(self, feature_dims):
        self.uncertainty_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            ) for dim in feature_dims
        ])
        
    def forward(self, features_list):
        uncertainties = [est(feat) for est, feat in zip(self.uncertainty_estimators, features_list)]
        weights = F.softmax(torch.cat(uncertainties, dim=-1), dim=-1)
        # Weighted combination based on uncertainty
        return sum(w * f for w, f in zip(weights, features_list))
```

**ACTION**: Add uncertainty-aware weighting to the fusion module for more robust feature combination.

### Phase 4: Training Enhancements

#### 4.1 Advanced Loss Functions
```python
# IMPLEMENTATION: Add focal loss and label smoothing
class EnhancedLossFunction(nn.Module):
    def __init__(self, n_classes, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.label_smoothing = label_smoothing
        self.consistency_weight = 0.5
        
    def forward(self, predictions, targets, reconstructed, original):
        # Classification loss with focal loss and label smoothing
        cls_loss = self.focal_loss(predictions, targets, self.label_smoothing)
        
        # Spectral reconstruction loss
        recon_loss = spectral_angle_distance(reconstructed, original)
        
        # Consistency regularization
        consistency_loss = F.mse_loss(predictions, predictions.detach())
        
        return cls_loss + 0.1 * recon_loss + self.consistency_weight * consistency_loss
```

**ACTION**: Implement focal loss for handling class imbalance, add label smoothing, and include consistency regularization.

#### 4.2 Advanced Data Augmentation
```python
# IMPLEMENTATION: Hyperspectral-specific augmentations
class HSIAugmentation:
    def __init__(self):
        self.spectral_noise = SpectralNoise(std_range=(0.01, 0.05))
        self.band_dropout = BandDropout(drop_prob=0.1)
        self.spectral_shift = SpectralShift(shift_range=(-2, 2))
        self.mixup = HSIMixup(alpha=0.2)
        
    def __call__(self, x, y):
        # Apply random combination of augmentations
        if random.random() < 0.3:
            x = self.spectral_noise(x)
        if random.random() < 0.2:
            x = self.band_dropout(x)
        if random.random() < 0.3:
            x = self.spectral_shift(x)
        if random.random() < 0.4:
            x, y = self.mixup(x, y)
        return x, y
```

**ACTION**: Implement hyperspectral-specific augmentations including spectral noise, band dropout, spectral shifts, and HSI mixup.

#### 4.3 Adaptive Learning Rate Scheduling
```python
# IMPLEMENTATION: Advanced learning rate scheduling
class AdaptiveLRScheduler:
    def __init__(self, optimizer, patience=10, factor=0.5, warmup_epochs=5):
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        self.warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        self.cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
    def step(self, epoch, val_loss):
        if epoch < 5:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
            self.scheduler.step(val_loss)
```

**ACTION**: Implement adaptive learning rate scheduling with warmup, cosine annealing, and plateau-based reduction.

### Phase 5: Model Ensemble and Post-Processing

#### 5.1 Multi-Scale Training
```python
# IMPLEMENTATION: Train on multiple patch sizes
class MultiScaleTraining:
    def __init__(self, model, scales=[3, 5, 7, 9]):
        self.model = model
        self.scales = scales
        
    def forward(self, x):
        predictions = []
        for scale in self.scales:
            # Extract patches of different sizes
            patches = extract_patches(x, patch_size=scale)
            pred = self.model(patches)
            predictions.append(pred)
        
        # Weighted ensemble
        weights = [0.4, 0.3, 0.2, 0.1]  # Favor smaller patches
        return sum(w * p for w, p in zip(weights, predictions))
```

**ACTION**: Implement multi-scale training and inference to capture features at different spatial resolutions.

#### 5.2 Test-Time Augmentation
```python
# IMPLEMENTATION: TTA for improved inference accuracy
class TestTimeAugmentation:
    def __init__(self, model, n_augmentations=8):
        self.model = model
        self.n_augmentations = n_augmentations
        
    def predict(self, x):
        predictions = []
        for _ in range(self.n_augmentations):
            # Apply light augmentations
            aug_x = self.light_augment(x)
            pred = self.model(aug_x)
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
```

**ACTION**: Implement test-time augmentation for more robust predictions during inference.

## IMPLEMENTATION PRIORITY ORDER

### High Priority (Immediate Implementation)
1. **Enhanced Loss Functions** (Phase 4.1) - Quick wins with focal loss and label smoothing
2. **Advanced Data Augmentation** (Phase 4.2) - Significant accuracy boost with minimal code changes
3. **Attention-Based Fusion** (Phase 3.1) - Core improvement to fusion strategy

### Medium Priority (Next Phase)
4. **Spectral-Spatial Transformer Enhancement** (Phase 2.2) - Major architectural improvement
5. **Adaptive Perturbation Modeling** (Phase 1.3) - Better variability modeling
6. **Multi-Scale Training** (Phase 5.1) - Robustness improvement

### Lower Priority (Future Enhancement)
7. **Advanced CNN Backbone** (Phase 2.1) - Computational overhead consideration
8. **Multi-Resolution Endmember Learning** (Phase 1.2) - Research-level enhancement
9. **Test-Time Augmentation** (Phase 5.2) - Inference-time improvement

## EXPECTED ACCURACY IMPROVEMENTS

- **Phase 4 (Training Enhancements)**: +1.5-2.5% accuracy
- **Phase 3 (Fusion Strategy)**: +1.0-2.0% accuracy  
- **Phase 2 (Classifier Enhancement)**: +1.0-1.5% accuracy
- **Phase 1 (Autoencoder Enhancement)**: +0.5-1.0% accuracy
- **Phase 5 (Ensemble Methods)**: +0.5-1.0% accuracy

**Total Expected Improvement: 4.5-8.0%**

## IMPLEMENTATION CHECKLIST

- [ ] Backup original S2VNet codebase
- [ ] Implement enhanced loss functions with focal loss
- [ ] Add hyperspectral-specific data augmentations
- [ ] Replace simple fusion with attention-based mechanism
- [ ] Upgrade transformer with spectral-spatial cross-attention
- [ ] Add adaptive perturbation modeling to autoencoder
- [ ] Implement multi-scale training pipeline
- [ ] Add comprehensive evaluation metrics
- [ ] Perform ablation studies for each component
- [ ] Fine-tune hyperparameters using grid search

## VALIDATION STRATEGY

1. **Ablation Study**: Test each component individually
2. **Cross-Dataset Validation**: Ensure improvements generalize
3. **Computational Analysis**: Monitor training time and memory usage
4. **Statistical Significance**: Use McNemar's test for accuracy comparisons

Start with High Priority modifications first, validate improvements, then proceed systematically through Medium and Lower Priority enhancements.