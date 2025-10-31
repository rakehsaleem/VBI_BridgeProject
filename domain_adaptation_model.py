import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization,
    Input, Lambda, Concatenate, Layer
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy.sparse as sp


def build_feature_extractor(input_shape=(250, 2), feature_dim=256):
    """
    Build shared CNN feature extractor (encoder).
    Used by all domains to extract domain-invariant features.
    
    Args:
        input_shape: (timesteps, channels) = (250, 2)
        feature_dim: Dimension of output feature vector
    
    Returns:
        Keras Model: Feature extraction model
    """
    inputs = Input(shape=input_shape)
    
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.35)(x)
    
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.35)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(feature_dim, activation='relu', name='features')(x)
    
    model = Model(inputs=inputs, outputs=x, name='feature_extractor')
    return model


def build_bridged_graph(source_features, target_features, k=10, metric='cosine'):
    """
    Build Bridged-Graph connecting target samples to similar source samples.
    
    Args:
        source_features: (N_source, feature_dim) source domain features
        target_features: (N_target, feature_dim) target domain features
        k: Number of nearest neighbors to connect
        metric: 'cosine' or 'euclidean'
    
    Returns:
        edges: (num_edges, 2) array of [source_idx, target_idx]
        edge_weights: (num_edges,) array of similarity weights
    """
    if metric == 'cosine':
        distances = cosine_similarity(target_features, source_features)
        # Cosine similarity: higher = more similar (negate for distances)
        similarities = distances
    else:  # euclidean
        distances = euclidean_distances(target_features, source_features)
        # Convert distance to similarity (inverse)
        similarities = 1.0 / (1.0 + distances)
    
    edges = []
    edge_weights = []
    
    for target_idx in range(len(target_features)):
        # Get top K most similar source samples
        source_indices = np.argsort(similarities[target_idx])[-k:][::-1]
        
        for source_idx in source_indices:
            edges.append([source_idx, target_idx])
            edge_weights.append(similarities[target_idx][source_idx])
    
    return np.array(edges), np.array(edge_weights)


class GraphKnowledgeAggregation(Layer):
    """
    TensorFlow layer for graph-based knowledge aggregation.
    Aggregates features from bridged neighbors during training.
    """
    def __init__(self, edges, edge_weights, aggregation_strength=0.3, **kwargs):
        super().__init__(**kwargs)
        # Convert edges and weights to tensors
        self.edges = tf.constant(edges, dtype=tf.int32)
        self.edge_weights = tf.constant(edge_weights, dtype=tf.float32)
        self.aggregation_strength = aggregation_strength
    
    def call(self, features):
        """
        Aggregate features from graph neighbors.
        
        Args:
            features: (batch_size, feature_dim) input features
        
        Returns:
            enhanced_features: Knowledge-enhanced features
        """
        # Get batch size and feature dim
        batch_size = tf.shape(features)[0]
        feature_dim = tf.shape(features)[1]
        
        # Initialize aggregated features
        enhanced_features = tf.zeros_like(features)
        neighbor_counts = tf.zeros((batch_size,), dtype=tf.float32)
        
        # Aggregate from neighbors (only for target samples that have edges)
        # Edge format: [source_idx, target_idx]
        # We only aggregate if target_idx is in current batch
        
        # For each edge, if target is in batch, aggregate source feature
        for i in range(len(self.edges)):
            source_idx = self.edges[i, 0]
            target_idx = self.edges[i, 1]
            weight = self.edge_weights[i]
            
            # Check if target is in batch (this is simplified - assumes full batch)
            # In practice, we'd need to map target indices to batch positions
            if target_idx < batch_size:
                source_feat = features[source_idx:source_idx+1]  # Keep dimension
                enhanced_features = tf.tensor_scatter_nd_add(
                    enhanced_features,
                    [[target_idx]],
                    source_feat * weight
                )
                neighbor_counts = tf.tensor_scatter_nd_add(
                    neighbor_counts,
                    [[target_idx]],
                    [weight]
                )
        
        # Normalize
        neighbor_counts = tf.maximum(neighbor_counts, 1e-8)  # Avoid division by zero
        enhanced_features = enhanced_features / tf.expand_dims(neighbor_counts, 1)
        
        # Combine original and aggregated (only for nodes with neighbors)
        has_neighbors = tf.greater(neighbor_counts, 1e-8)
        mask = tf.expand_dims(tf.cast(has_neighbors, tf.float32), 1)
        
        # Blend: original (1-α) + aggregated (α) only where neighbors exist
        final_features = (features * (1.0 - self.aggregation_strength * mask) + 
                         enhanced_features * (self.aggregation_strength * mask))
        
        return final_features
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'aggregation_strength': self.aggregation_strength
        })
        return config


def graph_attention_aggregation(features, edges, edge_weights, attention_heads=4):
    """
    NumPy-based graph attention aggregation (for non-training use).
    Aggregates features from connected nodes using attention mechanism.
    
    Args:
        features: (N, feature_dim) all node features (source + target)
        edges: (num_edges, 2) edge indices [source_idx, target_idx]
        edge_weights: (num_edges,) edge weights
        attention_heads: Number of attention heads
    
    Returns:
        enhanced_features: (N, feature_dim) knowledge-enhanced features
    """
    # For simplicity, using weighted average aggregation
    # Full GAT implementation would use learnable attention weights
    
    N, feature_dim = features.shape
    enhanced_features = np.zeros_like(features)
    
    # Count neighbors for each node
    neighbor_counts = np.zeros(N)
    
    for edge_idx, (source_idx, target_idx) in enumerate(edges):
        weight = edge_weights[edge_idx]
        enhanced_features[target_idx] += features[source_idx] * weight
        neighbor_counts[target_idx] += weight
    
    # Normalize by neighbor counts
    for i in range(N):
        if neighbor_counts[i] > 0:
            enhanced_features[i] /= neighbor_counts[i]
    
    # Combine original and aggregated features
    final_features = 0.7 * features + 0.3 * enhanced_features
    
    return final_features


class GradientReversalLayer(keras.layers.Layer):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    Reverses gradients during backpropagation to confuse domain discriminator.
    """
    def __init__(self, lambda_coeff=1.0, **kwargs):
        super().__init__(**kwargs)
        # Make lambda configurable at runtime
        self.lambda_coeff = tf.Variable(lambda_coeff, trainable=False, dtype=tf.float32)
    
    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)
        def custom_grad(dy):
            return -self.lambda_coeff * dy
        return y, custom_grad
    
    def call(self, inputs):
        return self.grad_reverse(inputs)
    
    def set_lambda(self, new_lambda: float):
        self.lambda_coeff.assign(tf.cast(new_lambda, tf.float32))
    
    def get_config(self):
        config = super().get_config()
        config.update({'lambda_coeff': float(self.lambda_coeff.numpy())})
        return config


def build_domain_discriminator(input_dim, num_domains=4):
    """
    Build domain discriminator to classify which bridge a sample comes from.
    Used adversarially to encourage domain-invariant features.
    
    Args:
        input_dim: Dimension of feature vector
        num_domains: Number of domains (4 bridges)
    
    Returns:
        Keras Model: Domain classification model
    """
    inputs = Input(shape=(input_dim,))
    
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_domains, activation='softmax', name='domain_pred')(x)
    
    model = Model(inputs=inputs, outputs=x, name='domain_discriminator')
    return model


def build_classification_head(input_dim, num_classes=5):
    """
    Build damage classification head.
    
    Args:
        input_dim: Dimension of feature vector
        num_classes: Number of damage classes (5)
    
    Returns:
        Keras Model: Classification model
    """
    inputs = Input(shape=(input_dim,))
    
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='damage_pred')(x)
    
    model = Model(inputs=inputs, outputs=x, name='classification_head')
    return model


def build_bridged_domain_adaptation_model(input_shape=(250, 2), feature_dim=256, num_classes=5, num_domains=4):
    """
    Build complete Bridged-GNN domain adaptation model.
    
    Returns:
        dict of models:
        - feature_extractor: Shared encoder
        - classifier: Damage classification head
        - domain_discriminator: Domain classification (adversarial)
    """
    # Build components
    feature_extractor = build_feature_extractor(input_shape, feature_dim)
    classifier = build_classification_head(feature_dim, num_classes)
    domain_discriminator = build_domain_discriminator(feature_dim, num_domains)
    
    return {
        'feature_extractor': feature_extractor,
        'classifier': classifier,
        'domain_discriminator': domain_discriminator
    }


if __name__ == '__main__':
    print("Testing domain adaptation model components...")
    
    # Test feature extractor
    feature_extractor = build_feature_extractor()
    print(f"\nFeature extractor:")
    feature_extractor.summary()
    
    # Test on sample data
    sample_input = np.random.randn(10, 250, 2).astype('float32')
    features = feature_extractor(sample_input)
    print(f"\nSample input shape: {sample_input.shape}")
    print(f"Output features shape: {features.shape}")
    
    # Test graph construction
    source_feat = np.random.randn(100, 256)
    target_feat = np.random.randn(20, 256)
    edges, weights = build_bridged_graph(source_feat, target_feat, k=5)
    print(f"\nBridged graph:")
    print(f"  Edges: {edges.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Sample edge: source={edges[0][0]}, target={edges[0][1]}, weight={weights[0]:.4f}")
    
    # Test domain discriminator
    domain_disc = build_domain_discriminator(256, 4)
    print(f"\nDomain discriminator:")
    domain_disc.summary()
    
    # Test classifier
    classifier = build_classification_head(256, 5)
    print(f"\nClassifier:")
    classifier.summary()
    
    print("\nAll components built successfully!")

