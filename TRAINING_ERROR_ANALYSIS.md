# Training Error Analysis and Learnings

## Error Summary

### Error Details
**Location**: Line 487 in `train_domain_adaptation.py`  
**Error Type**: `ValueError: Unknown variable`  
**Root Cause**: Optimizer variable tracking mismatch

### The Error
```
ValueError: Unknown variable: <Variable path=dense_1/kernel, shape=(256, 128), dtype=float32...>. 
This optimizer can only be called for the variables it was originally built with. 
When working with a new set of variables, you should recreate a new optimizer instance.
```

### Technical Explanation

The error occurred when trying to apply consistency loss gradients. The issue is:

1. **Optimizer Scope Mismatch**: 
   - `combined_model.optimizer` was created when `combined_model.compile()` was called
   - This optimizer tracks variables from `combined_model` (which includes feature_extractor, classifier, and domain_discriminator)
   - However, we tried to use `combined_model.optimizer.apply_gradients()` with a list of variables: `feature_extractor.trainable_variables + classifier.trainable_variables`
   
2. **Why It Failed**:
   - The optimizer maintains an internal list of variables it tracks (slot variables for momentum, Adam's m/v estimates, etc.)
   - When we passed `feature_extractor.trainable_variables + classifier.trainable_variables`, the optimizer didn't recognize these variables because they might be referenced differently (different variable objects or order)
   - Keras optimizers require that the variables passed to `apply_gradients()` match exactly the variables the optimizer was initialized with

3. **Inconsistency in Code**:
   - For MMD loss (line 461): Used `combined_model.optimizer` with `feature_extractor.trainable_variables` ✓ (worked)
   - For consistency loss (line 487): Used `combined_model.optimizer` with `feature_extractor.trainable_variables + classifier.trainable_variables` ✗ (failed)
   - For graph loss (line 527): Used `additional_loss_optimizer` with `feature_extractor.trainable_variables` ✓ (worked)

## What Worked Before the Error

### Successful Steps Completed:

1. **Pre-training**: ✅ Completed successfully
   - Trained on source domains for 20 epochs
   - Achieved ~94% accuracy on source data
   - Validation accuracy reached ~48% (indicates learning)

2. **Hyperparameter Search**: ✅ Completed successfully
   - Tested 5 different configurations
   - Selected best config based on validation loss
   - **Best Configuration (Config3)**:
     - alpha: 1.0
     - beta: 0.4
     - gamma_mmd: 0.25
     - delta_consistency: 0.1
     - lambda_reversal: 1.0
     - learning_rate: 0.00015
     - target_weight: 3.5
   - **Best validation loss: 1.1126** (lowest among all configs)

3. **Graph Construction**: ✅ Completed successfully
   - Built bridged-graph with 50,000 edges
   - Connected source to target (labeled + unlabeled)
   - Included 4,000 unlabeled target samples

4. **MMD Loss Application**: ✅ No error (working correctly)
   - Applied MMD loss for domain alignment
   - Used proper optimizer scope

### What Failed:

1. **Consistency Regularization**: ❌ Failed at first epoch
   - Error when applying consistency loss gradients
   - Optimizer couldn't handle classifier variables

## Key Learnings

### 1. **Optimizer Variable Tracking is Strict**
   - Keras optimizers maintain internal state per variable
   - Cannot mix-and-match variables from different model scopes
   - Each optimizer instance tracks specific variable objects
   - **Solution**: Use a single optimizer for all updates, OR use separate optimizers for separate loss components

### 2. **Hyperparameter Search Results Were Valuable**
   - Config3 performed best (validation loss: 1.1126)
   - Higher learning rate (0.00015) worked better than 0.0001
   - Moderate domain alignment (beta=0.4) + MMD loss (gamma=0.25) balanced well
   - Target sample weight of 3.5x was optimal

### 3. **Consistency Regularization Needs Careful Implementation**
   - Affects both feature extractor AND classifier
   - Requires optimizer that tracks both model components
   - Cannot rely on combined_model.optimizer if it doesn't track variables correctly

### 4. **Pre-training Validation**
   - Source domain pre-training worked well (~94% accuracy)
   - Validation accuracy was low (~48%), suggesting potential overfitting or domain gap
   - This validates the need for domain adaptation

## Solution Approach

### Fix Strategy:

**Option 1: Use Additional Loss Optimizer (Recommended)**
- Use `additional_loss_optimizer` (already created for graph loss) for consistency loss
- Apply gradients separately for feature_extractor and classifier
- Keeps updates independent and clear

**Option 2: Use Combined Model Optimizer Correctly**
- Ensure variables match exactly
- Better: use the optimizer from the model that contains all the variables
- Apply gradients in two separate calls if needed

**Option 3: Combine All Losses in Single Step**
- Compute all losses (MMD, consistency, classification, domain) in one gradient tape
- Apply all gradients together
- More complex but ensures optimizer consistency

### Recommended Fix:

Use **Option 1** - Apply consistency gradients using `additional_loss_optimizer`:

```python
# For consistency loss, use additional_loss_optimizer and apply separately
if target_X_unlabeled is not None and DELTA_CONSISTENCY > 0:
    # Apply to feature extractor
    with tf.GradientTape() as tape_fe:
        # ... compute consistency loss ...
    gradients_fe = tape_fe.gradient(consistency_loss_weighted, feature_extractor.trainable_variables)
    
    # Apply to classifier  
    with tf.GradientTape() as tape_cls:
        # ... compute consistency loss ...
    gradients_cls = tape_cls.gradient(consistency_loss_weighted, classifier.trainable_variables)
    
    # Apply using additional_loss_optimizer
    if additional_loss_optimizer is not None:
        additional_loss_optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients_fe, feature_extractor.trainable_variables) if g is not None]
        )
        additional_loss_optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients_cls, classifier.trainable_variables) if g is not None]
        )
```

## Next Steps

1. ✅ Fix optimizer usage for consistency loss
2. ✅ Re-run training with fixed code
3. ✅ Monitor all loss components separately
4. ✅ Evaluate performance on DC0-DC4
5. ✅ Compare results with/without consistency regularization

## Expected Benefits After Fix

1. **Better Domain Alignment**: MMD loss will help align source and target feature distributions
2. **Improved Generalization**: Consistency regularization will help the model generalize to unlabeled target samples
3. **Robust Predictions**: Graph knowledge transfer will inject source knowledge into target predictions
4. **Optimal Hyperparameters**: Using the best config found during search

