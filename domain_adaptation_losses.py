"""
Additional loss functions for domain adaptation.
Includes MMD loss, consistency regularization, and other domain alignment techniques.
"""
import tensorflow as tf
import numpy as np


def mmd_loss(source_features, target_features, kernel='rbf', sigma=1.0):
    """
    Maximum Mean Discrepancy (MMD) loss for domain alignment.
    
    Args:
        source_features: (N_s, feature_dim) tensor of source domain features
        target_features: (N_t, feature_dim) tensor of target domain features
        kernel: 'rbf' or 'linear'
        sigma: RBF kernel parameter
    
    Returns:
        MMD loss scalar
    """
    if kernel == 'rbf':
        # RBF kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))
        def rbf_kernel(X, Y, s):
            # X: (N, D), Y: (M, D)
            # Compute ||x_i - y_j||^2 for all pairs
            XX = tf.reduce_sum(X * X, axis=1, keepdims=True)  # (N, 1)
            YY = tf.reduce_sum(Y * Y, axis=1, keepdims=True)  # (M, 1)
            XY = tf.matmul(X, tf.transpose(Y))  # (N, M)
            
            # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2*x_i^T*y_j
            distances_sq = XX + tf.transpose(YY) - 2 * XY
            return tf.exp(-distances_sq / (2 * s ** 2))
        
        # Support multi-kernel by allowing sigma to be a list/tuple
        if isinstance(sigma, (list, tuple)):
            K_ss = 0.0
            K_tt = 0.0
            K_st = 0.0
            for s in sigma:
                K_ss += rbf_kernel(source_features, source_features, s)
                K_tt += rbf_kernel(target_features, target_features, s)
                K_st += rbf_kernel(source_features, target_features, s)
            K_ss /= float(len(sigma))
            K_tt /= float(len(sigma))
            K_st /= float(len(sigma))
        else:
            # Compute single-kernel matrices
            K_ss = rbf_kernel(source_features, source_features, sigma)  # (N_s, N_s)
            K_tt = rbf_kernel(target_features, target_features, sigma)  # (N_t, N_t)
            K_st = rbf_kernel(source_features, target_features, sigma)  # (N_s, N_t)
        
        # MMD^2 = mean(K_ss) + mean(K_tt) - 2*mean(K_st)
        mmd_sq = (
            tf.reduce_mean(K_ss) + 
            tf.reduce_mean(K_tt) - 
            2 * tf.reduce_mean(K_st)
        )
        
        return mmd_sq
    
    elif kernel == 'linear':
        # Linear kernel: k(x, y) = x^T * y
        mean_source = tf.reduce_mean(source_features, axis=0)
        mean_target = tf.reduce_mean(target_features, axis=0)
        
        mmd_sq = tf.reduce_sum((mean_source - mean_target) ** 2)
        return mmd_sq
    
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def consistency_loss(predictions_1, predictions_2):
    """
    Consistency regularization loss for semi-supervised learning.
    Encourages consistent predictions under different augmentations/noise.
    
    Args:
        predictions_1: (N, num_classes) prediction tensor
        predictions_2: (N, num_classes) prediction tensor (same samples, different augmentation)
    
    Returns:
        Consistency loss (MSE between predictions)
    """
    return tf.reduce_mean(tf.reduce_sum((predictions_1 - predictions_2) ** 2, axis=1))


def entropy_loss(predictions):
    """
    Entropy regularization for unlabeled data.
    Encourages confident predictions (low entropy).
    
    Args:
        predictions: (N, num_classes) prediction probabilities
    
    Returns:
        Negative entropy (to maximize confidence)
    """
    # Entropy: -sum(p * log(p))
    epsilon = 1e-8  # Avoid log(0)
    entropy = -tf.reduce_sum(predictions * tf.math.log(predictions + epsilon), axis=1)
    # Return mean entropy (we want to minimize this = maximize confidence)
    return tf.reduce_mean(entropy)


def mixup_data(x, y, alpha=0.2):
    """
    Mixup data augmentation.
    
    Args:
        x: input features (N, ...)
        y: labels (N, num_classes)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x, mixed_y, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    
    return mixed_x, mixed_y, lam

