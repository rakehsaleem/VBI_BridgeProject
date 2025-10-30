# Hyperparameter Tuning for Domain Adaptation

## Issues Identified from Training

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

## Next Steps if Issues Persist

If target accuracy is still low:
1. Further reduce BETA_DOMAIN to 0.001
2. Check if pre-training is effective (should achieve >90% on source)
3. Verify data loading - ensure target labels are correct
4. Consider class imbalance - target only has DC0 labels, which might affect learning

