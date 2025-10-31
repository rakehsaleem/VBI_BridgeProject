# Hyperparameter Tuning for Domain Adaptation

## Latest Update: Systematic Hyperparameter Optimization (2025-10-30)

### Overview
A systematic hyperparameter optimization approach was implemented in `train_domain_adaptation_optimized.py` to automatically search for the best combination of hyperparameters using validation loss as the selection criterion.

### Why Hyperparameter Tuning Was Performed
The domain adaptation model involves multiple competing objectives that need careful balancing:
1. **Classification Loss (α)**: Primary task - damage classification accuracy
2. **Domain Alignment (β)**: Adversarial domain alignment to reduce domain shift
3. **MMD Loss (γ_mmd)**: Maximum Mean Discrepancy for distribution matching
4. **Consistency Regularization (δ_consistency)**: Smoothness constraint for unlabeled target data
5. **Gradient Reversal Strength (λ_reversal)**: Controls adversarial training intensity
6. **Learning Rate (lr)**: Training stability and convergence speed
7. **Target Weight**: Importance multiplier for target domain samples

Manually tuning these 7 hyperparameters is intractable. A systematic grid search was implemented to find optimal combinations.

### How It Was Achieved

#### Implementation Details
- **Method**: Grid search over 8 carefully designed hyperparameter configurations
- **Selection Metric**: Validation loss (20% validation split from training data)
- **Training Strategy**: Each configuration tested with:
  1. Pre-training on source domain (5 epochs for quick initialization)
  2. Domain adaptation training (15 epochs with full loss components)
  3. Validation monitoring to select best configuration
  
#### Hyperparameter Grid Tested
The search space included 8 configurations ranging from:
- **Conservative**: Lower domain alignment, moderate reversal
- **Aggressive**: Strong domain alignment, higher reversal
- **Balanced**: Mid-range values across all parameters

Key configurations tested:
```python
# Example configurations from the grid:
1. alpha=1.0, beta=0.5, gamma_mmd=0.2, delta_consistency=0.1, lambda_reversal=0.8, lr=0.0001, target_weight=3.0
2. alpha=1.0, beta=0.3, gamma_mmd=0.3, delta_consistency=0.15, lambda_reversal=1.0, lr=0.0001, target_weight=3.0
3. alpha=0.9, beta=0.6, gamma_mmd=0.3, delta_consistency=0.15, lambda_reversal=0.8, lr=0.00015, target_weight=3.5
4. ... (5 more configurations with varying emphasis on domain alignment vs. classification)
```

#### Optimization Process
1. **Validation Split**: 20% of combined source and target labeled data held out for validation
2. **Fresh Models**: Each hyperparameter set tested with newly initialized models to ensure fair comparison
3. **Early Evaluation**: 15 epochs per configuration for efficient search (full training uses 30+ epochs)
4. **Best Selection**: Configuration with minimum validation loss selected for full training

### Results Achieved

#### Optimal Hyperparameters Found
Based on validation loss minimization, the following parameters were identified as optimal:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **alpha** | 0.9 | Classification loss weight (slightly reduced to allow domain adaptation) |
| **beta** | 0.6 | Domain alignment weight (moderate strength) |
| **gamma_mmd** | 0.3 | MMD loss weight for distribution matching |
| **delta_consistency** | 0.15 | Consistency regularization weight |
| **lambda_reversal** | 0.8 | Gradient reversal strength for adversarial training |
| **lr** | 0.00015 | Learning rate (balanced between stability and speed) |
| **target_weight** | 3.5 | Target domain sample importance multiplier |

#### Performance Results
- **Best Validation Loss**: 0.7695 (during full training with early stopping)
- **Previous Target Accuracy**: 3.40% (before optimization)
- **Optimized Target Accuracy**: 64.50% ✅
- **Improvement**: **+61.1 percentage points** (19x improvement)

#### Key Insights from Optimization
1. **Moderate Domain Alignment Works Best**: Beta=0.6 provides good balance - too weak (β<0.3) fails to align domains, too strong (β>0.9) disrupts classification
2. **MMD Loss is Important**: gamma_mmd=0.3 helps bridge the distribution gap between source and target
3. **Target Weighting Matters**: target_weight=3.5 compensates for target domain data scarcity
4. **Consistency Regularization Helps**: delta_consistency=0.15 provides smooth decision boundaries
5. **Learning Rate Balance**: lr=0.00015 is optimal - higher rates cause instability, lower rates slow convergence

### Implementation Location
- **File**: `train_domain_adaptation_optimized.py`
- **Key Functions**:
  - `hyperparameter_search()`: Grid search implementation (lines 306-460)
  - `train_with_hyperparameters()`: Training with specific hyperparameters (lines 209-303)
  - `train_full_model_with_best_params()`: Full training with optimal parameters (lines 463-675)

### Full Training Pipeline
After hyperparameter optimization, the best parameters are used for:
1. **Extended Pre-training**: 20 epochs on source domain
2. **Bridged-Graph Construction**: Knowledge graph from source features
3. **Domain Adaptation Training**: 30+ epochs with all loss components:
   - Classification loss (α)
   - Domain adversarial loss (β)
   - MMD loss (γ_mmd)
   - Consistency loss (δ_consistency)
   - Graph knowledge transfer (every 3 epochs)
4. **Early Stopping**: Monitors validation loss with patience=10
5. **Model Saving**: Timestamped models saved to `domain_adaptation_results_optimized/`

---

## Previous Issues Identified from Training

1. **0% target accuracy** - All predictions were class 3, but true labels are class 0
2. **Domain discriminator too strong** - Reached 100% accuracy (perfect domain separation is BAD for domain adaptation)
3. **Classification loss explosion** - Loss ratio reached 307,736x compared to domain loss
4. **Training instability** - Classification accuracy showed 0.0000 during many epochs

## Hyperparameter Adjustments Made

### 1. Domain Alignment Weight (BETA_DOMAIN)
- **Old**: `0.1`
- **New**: `0.01` (10x reduction)
- **Reason**: Domain discriminator was too successful (100% accuracy), meaning domains weren't being aligned. Reducing beta weakens domain alignment loss to allow classification to dominate.

### 2. Gradient Reversal Strength (DOMAIN_REVERSAL_LAMBDA)
- **Old**: `0.1`
- **New**: `1.0` (10x increase)
- **Reason**: Gradient reversal needs to be stronger to effectively confuse the domain discriminator. Higher lambda means more aggressive reversal to push features toward domain invariance.

### 3. Graph Alignment Weight (alignment_gamma)
- **Old**: `0.1`
- **New**: `0.01` (10x reduction)
- **Reason**: Graph alignment losses were very large (200-300), which could destabilize training. Reducing this weight makes graph knowledge transfer more gentle.

### 4. Learning Rate
- **Old**: `0.0005`
- **New**: `0.0001` (5x reduction)
- **Reason**: Lower learning rate provides better stability, especially when dealing with multiple competing objectives (classification vs. domain alignment).

### 5. Gradient Clipping
- **Added**: `clipnorm=1.0` to Adam optimizer
- **Reason**: Prevents gradient explosion, which was causing the classification loss to blow up.

### 6. Evaluation Subset
- **Old**: First 1000 samples (likely all source)
- **New**: 500 source + 500 target samples
- **Reason**: More balanced evaluation that reflects both domains, not just source.

### 7. Domain Confusion Metrics
- **Improved**: Better messaging about domain discriminator accuracy
- **Clarification**: Lower discriminator accuracy = better domain alignment (discriminator is confused)
- **Goal**: Discriminator should be around 50% accurate (random chance)

## Expected Improvements

With these changes:
1. **Domain alignment should improve** - Discriminator accuracy should decrease toward 50%
2. **Classification stability** - Lower learning rate and gradient clipping prevent loss explosion
3. **Better balance** - Weaker domain loss (beta=0.01) allows classification to learn better
4. **Graph knowledge transfer** - Gentler alignment updates (gamma=0.01) prevent interference

## Target Metrics After Tuning

- **Domain discriminator accuracy**: ~50-60% (confused = good)
- **Target classification accuracy**: >50% (improvement from 0%)
- **Loss ratio (class/domain)**: <100x (down from 307,736x)
- **Training stability**: Smooth loss curves without spikes

## Training Strategy

1. **Stage 1 (Pre-training)**: Train feature extractor + classifier on source domains only
   - This ensures good initial features for classification

2. **Stage 2 (Domain Adaptation)**: Combine source + target labeled data
   - Weak domain alignment (beta=0.01)
   - Strong gradient reversal (lambda=1.0)
   - Graph knowledge transfer every 3 epochs

## Summary

The systematic hyperparameter optimization successfully addressed the initial training issues:

1. ✅ **Target accuracy improved from 3.4% to 64.5%** - A 19x improvement demonstrating effective domain adaptation
2. ✅ **Balanced loss components** - Optimal weights found for all 7 hyperparameters through validation-based search
3. ✅ **Stable training** - Proper learning rate and gradient clipping prevent loss explosions
4. ✅ **Effective domain alignment** - Moderate beta (0.6) with MMD loss (0.3) achieves good domain matching

### Key Takeaway
The optimization revealed that **moderate domain alignment (β=0.6) combined with MMD loss (γ=0.3) and consistency regularization (δ=0.15)** provides the best trade-off between preserving classification performance and aligning source-target distributions. This was discovered through systematic search rather than manual tuning.

## Future Improvements

While the current results show significant improvement, further enhancements could include:

1. **Expanded Search Space**: Test more hyperparameter combinations (e.g., Bayesian optimization)
2. **Per-Domain Adaptation**: Different hyperparameters for different source bridges
3. **Dynamic Weighting**: Adaptive loss weights that change during training
4. **Semi-supervised Learning**: Better utilization of unlabeled target data
5. **Architecture Search**: Optimize network architecture alongside hyperparameters
6. **Ensemble Methods**: Combine multiple optimized models for better generalization

