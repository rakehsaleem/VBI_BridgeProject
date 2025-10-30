"""
Optimized domain adaptation training with:
- MMD loss for domain alignment
- Consistency regularization for unlabeled data
- Improved graph knowledge transfer
- Hyperparameter optimization with validation monitoring
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from datetime import datetime
import sys
import copy

from domain_adaptation_data_loader import load_domain_adaptation_data, load_bridge_data, config
from domain_adaptation_model import (
    build_bridged_domain_adaptation_model,
    build_bridged_graph,
    graph_attention_aggregation,
    GradientReversalLayer,
    GraphKnowledgeAggregation
)
from domain_adaptation_losses import mmd_loss, consistency_loss, entropy_loss


class DomainAdaptationModel(tf.keras.Model):
    """
    Custom model that supports MMD loss and consistency regularization.
    """
    def __init__(self, feature_extractor, classifier, domain_discriminator, 
                 alpha=1.0, beta=0.1, gamma_mmd=0.1, delta_consistency=0.1, 
                 lambda_reversal=0.8):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.alpha = alpha
        self.beta = beta
        self.gamma_mmd = gamma_mmd
        self.delta_consistency = delta_consistency
        self.lambda_reversal = lambda_reversal
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(lambda_coeff=lambda_reversal)
    
    def call(self, inputs, training=False):
        features = self.feature_extractor(inputs, training=training)
        class_pred = self.classifier(features, training=training)
        reversed_features = self.gradient_reversal(features)
        domain_pred = self.domain_discriminator(reversed_features, training=training)
        return {
            'classification': class_pred,
            'domain': domain_pred,
            'features': features
        }
    
    def train_step(self, data):
        """Custom training step with multiple losses."""
        x, y = data
        
        # Unpack targets
        if isinstance(y, dict):
            y_class = y['classification']
            y_domain = y['domain']
        else:
            # Handle list format
            y_class = y[0]
            y_domain = y[1]
        
        # Separate source and target
        if 'source_indices' in data and 'target_indices' in data:
            source_idx = data['source_indices']
            target_idx = data['target_indices']
            x_source = tf.gather(x, source_idx)
            x_target = tf.gather(x, target_idx)
            
            # Get unlabeled target if available
            if 'target_unlabeled_indices' in data:
                target_unlabeled_idx = data['target_unlabeled_indices']
                x_target_unlabeled = tf.gather(x, target_unlabeled_idx)
            else:
                x_target_unlabeled = None
        else:
            # Fallback: assume first N are source, rest are target
            x_source = x[:len(x)//2] if len(x) > 1 else x
            x_target = x[len(x)//2:] if len(x) > 1 else x
            x_target_unlabeled = None
        
        with tf.GradientTape() as tape:
            # Forward pass
            source_features = self.feature_extractor(x_source, training=True)
            target_features = self.feature_extractor(x_target, training=True)
            
            # Classification predictions
            source_class_pred = self.classifier(source_features, training=True)
            target_class_pred = self.classifier(target_features, training=True)
            
            # Domain predictions (with gradient reversal)
            source_features_reversed = self.gradient_reversal(source_features)
            target_features_reversed = self.gradient_reversal(target_features)
            source_domain_pred = self.domain_discriminator(source_features_reversed, training=True)
            target_domain_pred = self.domain_discriminator(target_features_reversed, training=True)
            
            # Compute losses
            # 1. Classification loss
            source_class_loss = tf.keras.losses.categorical_crossentropy(
                y_class[:len(source_features)], source_class_pred)
            target_class_loss = tf.keras.losses.categorical_crossentropy(
                y_class[len(source_features):], target_class_pred)
            class_loss = tf.reduce_mean(source_class_loss) + tf.reduce_mean(target_class_loss)
            
            # 2. Domain adversarial loss
            source_domain_true = y_domain[:len(source_features)]
            target_domain_true = y_domain[len(source_features):]
            domain_loss = (
                tf.keras.losses.categorical_crossentropy(source_domain_true, source_domain_pred) +
                tf.keras.losses.categorical_crossentropy(target_domain_true, target_domain_pred)
            )
            domain_loss = tf.reduce_mean(domain_loss)
            
            # 3. MMD loss for domain alignment
            mmd = mmd_loss(source_features, target_features, kernel='rbf', sigma=1.0)
            
            # 4. Consistency regularization (if unlabeled data available)
            consistency = 0.0
            if x_target_unlabeled is not None:
                target_unlabeled_features = self.feature_extractor(x_target_unlabeled, training=True)
                target_unlabeled_pred = self.classifier(target_unlabeled_features, training=True)
                # Add small noise and get predictions again
                target_unlabeled_features_noisy = target_unlabeled_features + 0.01 * tf.random.normal(tf.shape(target_unlabeled_features))
                target_unlabeled_pred_noisy = self.classifier(target_unlabeled_features_noisy, training=True)
                consistency = consistency_loss(target_unlabeled_pred, target_unlabeled_pred_noisy)
            
            # Total loss
            total_loss = (
                self.alpha * class_loss +
                self.beta * domain_loss +
                self.gamma_mmd * mmd +
                self.delta_consistency * consistency
            )
        
        # Compute gradients
        trainable_vars = (
            self.feature_extractor.trainable_variables +
            self.classifier.trainable_variables +
            self.domain_discriminator.trainable_variables
        )
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Clip gradients
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in gradients]
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Metrics
        source_class_acc = tf.keras.metrics.categorical_accuracy(
            y_class[:len(source_features)], source_class_pred)
        target_class_acc = tf.keras.metrics.categorical_accuracy(
            y_class[len(source_features):], target_class_pred)
        
        return {
            'loss': total_loss,
            'class_loss': class_loss,
            'domain_loss': domain_loss,
            'mmd_loss': mmd,
            'consistency_loss': consistency,
            'source_class_acc': tf.reduce_mean(source_class_acc),
            'target_class_acc': tf.reduce_mean(target_class_acc),
        }


def create_validation_split(source_X, source_Y, target_X_labeled, target_Y_labeled, val_split=0.2):
    """Create validation split from training data."""
    # Split source data
    n_source_val = int(len(source_X) * val_split)
    indices_source = np.random.permutation(len(source_X))
    source_val_indices = indices_source[:n_source_val]
    source_train_indices = indices_source[n_source_val:]
    
    # Split target data
    n_target_val = int(len(target_X_labeled) * val_split)
    indices_target = np.random.permutation(len(target_X_labeled))
    target_val_indices = indices_target[:n_target_val]
    target_train_indices = indices_target[n_target_val:]
    
    # Create splits
    X_train = np.vstack([source_X[source_train_indices], target_X_labeled[target_train_indices]])
    Y_train = np.concatenate([source_Y[source_train_indices], target_Y_labeled[target_train_indices]])
    
    X_val = np.vstack([source_X[source_val_indices], target_X_labeled[target_val_indices]])
    Y_val = np.concatenate([source_Y[source_val_indices], target_Y_labeled[target_val_indices]])
    
    return (X_train, Y_train), (X_val, Y_val), {
        'source_train_idx': source_train_indices,
        'source_val_idx': source_val_indices,
        'target_train_idx': target_train_indices,
        'target_val_idx': target_val_indices
    }


def train_with_hyperparameters(
    feature_extractor, classifier, domain_discriminator,
    source_X, source_Y, target_X_labeled, target_Y_labeled, target_X_unlabeled,
    source_bridge_id, hyperparams, validation_data=None, epochs=30
):
    """
    Train model with given hyperparameters and return validation loss.
    
    Returns:
        best_val_loss, history
    """
    alpha = hyperparams['alpha']
    beta = hyperparams['beta']
    gamma_mmd = hyperparams['gamma_mmd']
    delta_consistency = hyperparams['delta_consistency']
    lambda_reversal = hyperparams['lambda_reversal']
    lr = hyperparams['lr']
    target_weight = hyperparams['target_weight']
    
    # Create model
    model = DomainAdaptationModel(
        feature_extractor, classifier, domain_discriminator,
        alpha=alpha, beta=beta, gamma_mmd=gamma_mmd,
        delta_consistency=delta_consistency, lambda_reversal=lambda_reversal
    )
    
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss={'classification': 'categorical_crossentropy', 'domain': 'categorical_crossentropy'},
        metrics=['accuracy']
    )
    
    # Prepare data
    combined_X = np.vstack([source_X, target_X_labeled])
    combined_Y_class = to_categorical(np.concatenate([source_Y, target_Y_labeled]), 5)
    combined_Y_domain = to_categorical(
        np.concatenate([source_bridge_id, np.full(len(target_Y_labeled), 3)]), 4
    )
    
    # Sample weights
    sample_weights = np.ones(len(combined_X))
    sample_weights[len(source_X):] = target_weight
    
    best_val_loss = float('inf')
    patience = 10
    wait = 0
    
    for epoch in range(epochs):
        # Simplified training for hyperparameter search
        # In real implementation, would use the custom train_step
        with tf.GradientTape() as tape:
            # Forward pass
            features = model.feature_extractor(combined_X, training=True)
            class_pred = model.classifier(features, training=True)
            
            # Compute basic losses
            class_loss = tf.keras.losses.categorical_crossentropy(
                combined_Y_class, class_pred)
            class_loss = tf.reduce_mean(class_loss)
            
            # MMD loss
            source_features = features[:len(source_X)]
            target_features = features[len(source_X):]
            mmd = mmd_loss(source_features, target_features)
            
            total_loss = alpha * class_loss + gamma_mmd * mmd
        
        # Train (simplified)
        trainable_vars = (model.feature_extractor.trainable_variables + 
                         model.classifier.trainable_variables)
        gradients = tape.gradient(total_loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in gradients]
        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Validation
        if validation_data:
            X_val, Y_val = validation_data
            val_features = model.feature_extractor(X_val, training=False)
            val_pred = model.classifier(val_features, training=False)
            val_loss = tf.keras.losses.categorical_crossentropy(
                to_categorical(Y_val, 5), val_pred)
            val_loss = tf.reduce_mean(val_loss).numpy()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
    
    return best_val_loss, {'val_loss': best_val_loss}


def hyperparameter_search(
    feature_extractor, classifier, domain_discriminator,
    source_X, source_Y, target_X_labeled, target_Y_labeled, target_X_unlabeled,
    source_bridge_id
):
    """
    Search for best hyperparameters using validation loss.
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)
    
    # Create validation split
    (X_train, Y_train), (X_val, Y_val), split_info = create_validation_split(
        source_X, source_Y, target_X_labeled, target_Y_labeled, val_split=0.2
    )
    
    # Hyperparameter grid to search
    param_grid = [
        {
            'alpha': 1.0,
            'beta': 0.5,
            'gamma_mmd': 0.2,
            'delta_consistency': 0.1,
            'lambda_reversal': 0.8,
            'lr': 0.0001,
            'target_weight': 3.0
        },
        {
            'alpha': 1.0,
            'beta': 0.3,
            'gamma_mmd': 0.3,
            'delta_consistency': 0.15,
            'lambda_reversal': 1.0,
            'lr': 0.0001,
            'target_weight': 3.0
        },
        {
            'alpha': 1.0,
            'beta': 0.4,
            'gamma_mmd': 0.25,
            'delta_consistency': 0.2,
            'lambda_reversal': 0.9,
            'lr': 0.0001,
            'target_weight': 4.0
        },
        {
            'alpha': 0.9,
            'beta': 0.6,
            'gamma_mmd': 0.3,
            'delta_consistency': 0.15,
            'lambda_reversal': 0.8,
            'lr': 0.00015,
            'target_weight': 3.5
        },
        {
            'alpha': 1.0,
            'beta': 0.5,
            'gamma_mmd': 0.4,
            'delta_consistency': 0.1,
            'lambda_reversal': 0.7,
            'lr': 0.0001,
            'target_weight': 4.0
        },
    ]
    
    results = []
    
    for i, params in enumerate(param_grid):
        print(f"\nTesting hyperparameter set {i+1}/{len(param_grid)}:")
        print(f"  alpha={params['alpha']}, beta={params['beta']}, "
              f"gamma_mmd={params['gamma_mmd']}, delta_consistency={params['delta_consistency']}")
        print(f"  lambda_reversal={params['lambda_reversal']}, lr={params['lr']}, "
              f"target_weight={params['target_weight']}")
        
        # Create fresh models for each test
        feat_ext = build_bridged_domain_adaptation_model(
            input_shape=(250, 2), feature_dim=256, num_classes=5, num_domains=4
        )['feature_extractor']
        cls = build_bridged_domain_adaptation_model(
            input_shape=(250, 2), feature_dim=256, num_classes=5, num_domains=4
        )['classifier']
        dom_disc = build_bridged_domain_adaptation_model(
            input_shape=(250, 2), feature_dim=256, num_classes=5, num_domains=4
        )['domain_discriminator']
        
        # Pre-train on source (quick)
        print("  Pre-training on source...")
        pre_model = tf.keras.Sequential([feat_ext, cls])
        pre_model.compile(optimizer=Adam(learning_rate=params['lr']), loss='categorical_crossentropy', metrics=['accuracy'])
        pre_model.fit(
            source_X, to_categorical(source_Y, 5),
            batch_size=32, epochs=5, verbose=0
        )
        
        # Train with domain adaptation
        val_loss, history = train_with_hyperparameters(
            feat_ext, cls, dom_disc,
            source_X, source_Y, target_X_labeled, target_Y_labeled, target_X_unlabeled,
            source_bridge_id, params, validation_data=(X_val, Y_val), epochs=15
        )
        
        results.append({
            'params': params,
            'val_loss': val_loss,
            'history': history
        })
        
        print(f"  Validation loss: {val_loss:.4f}")
    
    # Find best parameters
    best_result = min(results, key=lambda x: x['val_loss'])
    best_params = best_result['params']
    
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS:")
    print("="*70)
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Validation loss: {best_result['val_loss']:.4f}")
    print("="*70)
    
    return best_params, results


def train_full_model_with_best_params(best_params, source_X, source_Y, target_X_labeled, 
                                      target_Y_labeled, target_X_unlabeled, source_bridge_id):
    """
    Train full model with best hyperparameters and full training loop.
    """
    print("\n" + "="*70)
    print("FULL TRAINING WITH BEST HYPERPARAMETERS")
    print("="*70)
    
    # Build models
    models = build_bridged_domain_adaptation_model(
        input_shape=(250, 2), feature_dim=256, num_classes=5, num_domains=4
    )
    feature_extractor = models['feature_extractor']
    classifier = models['classifier']
    domain_discriminator = models['domain_discriminator']
    
    # Pre-train on source
    print("\n[1/6] Pre-training on source domains...")
    pre_model = tf.keras.Sequential([feature_extractor, classifier])
    pre_model.compile(
        optimizer=Adam(learning_rate=best_params['lr'] * 10),  # Higher LR for pre-training
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    pre_model.fit(
        source_X, to_categorical(source_Y, 5),
        batch_size=32, epochs=20, shuffle=True, validation_split=0.2, verbose=1
    )
    
    print("\n[2/6] Building Bridged-Graph...")
    # Extract features and build graph
    source_features = feature_extractor.predict(source_X, verbose=0)
    target_labeled_features = feature_extractor.predict(target_X_labeled, verbose=0)
    
    if target_X_unlabeled is not None:
        target_unlabeled_features = feature_extractor.predict(target_X_unlabeled, verbose=0)
        all_target_features = np.vstack([target_labeled_features, target_unlabeled_features])
    else:
        all_target_features = target_labeled_features
    
    edges, edge_weights = build_bridged_graph(
        source_features, all_target_features, k=10, metric='cosine'
    )
    print(f"  Built {len(edges)} edges connecting source to target")
    
    # Map edges to combined indices
    edges_combined = []
    edge_weights_combined = []
    for edge_idx, (source_global, target_global) in enumerate(edges):
        if target_global < len(target_labeled_features):
            source_combined = source_global
            target_combined = len(source_X) + target_global
            edges_combined.append([source_combined, target_combined])
            edge_weights_combined.append(edge_weights[edge_idx])
    
    edges_combined = np.array(edges_combined) if edges_combined else np.array([]).reshape(0, 2)
    edge_weights_combined = np.array(edge_weights_combined) if edge_weights_combined else np.array([])
    
    print("\n[3/6] Starting domain adaptation training...")
    
    # Prepare data
    combined_X = np.vstack([source_X, target_X_labeled])
    combined_Y = np.concatenate([source_Y, target_Y_labeled])
    combined_Y_cat = to_categorical(combined_Y, 5)
    combined_bridge_ids = np.concatenate([source_bridge_id, np.full(len(target_Y_labeled), 3)])
    combined_bridge_ids_cat = to_categorical(combined_bridge_ids, 4)
    
    sample_weights = np.ones(len(combined_X))
    sample_weights[len(source_X):] = best_params['target_weight']
    
    # Build model
    model = DomainAdaptationModel(
        feature_extractor, classifier, domain_discriminator,
        alpha=best_params['alpha'],
        beta=best_params['beta'],
        gamma_mmd=best_params['gamma_mmd'],
        delta_consistency=best_params['delta_consistency'],
        lambda_reversal=best_params['lambda_reversal']
    )
    
    # For full training, we'll use a simplified approach with custom loss
    # Build standard combined model
    inputs = Input(shape=(250, 2))
    features = feature_extractor(inputs)
    class_output = classifier(features)
    reversed_features = GradientReversalLayer(lambda_coeff=best_params['lambda_reversal'])(features)
    domain_output = domain_discriminator(reversed_features)
    
    combined_model = Model(inputs=inputs, outputs=[class_output, domain_output])
    
    # Custom loss function that includes MMD
    def combined_loss(y_true, y_pred):
        # This will be computed per output
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    combined_model.compile(
        optimizer=Adam(learning_rate=best_params['lr'], clipnorm=1.0),
        loss=['categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[best_params['alpha'], best_params['beta']],
        metrics=['accuracy', 'accuracy']
    )
    
    # Training loop with MMD and consistency losses
    epochs = 30
    batch_size = 32
    best_val_loss = float('inf')
    patience = 10
    wait = 0
    
    # Create validation set
    val_split = 0.2
    n_val = int(len(combined_X) * val_split)
    val_indices = np.random.choice(len(combined_X), n_val, replace=False)
    train_indices = np.setdiff1d(np.arange(len(combined_X)), val_indices)
    
    X_train = combined_X[train_indices]
    Y_train_class = combined_Y_cat[train_indices]
    Y_train_domain = combined_bridge_ids_cat[train_indices]
    X_val = combined_X[val_indices]
    Y_val_class = combined_Y_cat[val_indices]
    Y_val_domain = combined_bridge_ids_cat[val_indices]
    
    for epoch in range(epochs):
        # Standard training step
        history = combined_model.fit(
            X_train, [Y_train_class, Y_train_domain],
            sample_weight=sample_weights[train_indices],
            batch_size=batch_size, epochs=1, verbose=0
        )
        
        # Additional MMD and consistency losses via gradient update
        with tf.GradientTape() as tape:
            source_features = feature_extractor(source_X, training=True)
            target_labeled_features = feature_extractor(target_X_labeled, training=True)
            
            # MMD loss
            mmd = mmd_loss(source_features, target_labeled_features)
            
            # Consistency loss on unlabeled target
            consistency = 0.0
            if target_X_unlabeled is not None:
                target_unlabeled_features = feature_extractor(target_X_unlabeled, training=True)
                target_unlabeled_pred = classifier(target_unlabeled_features, training=True)
                # Add noise
                target_unlabeled_features_noisy = target_unlabeled_features + 0.01 * tf.random.normal(tf.shape(target_unlabeled_features))
                target_unlabeled_pred_noisy = classifier(target_unlabeled_features_noisy, training=True)
                consistency = consistency_loss(target_unlabeled_pred, target_unlabeled_pred_noisy)
            
            # Graph knowledge transfer (every 3 epochs)
            graph_loss = 0.0
            if (epoch + 1) % 3 == 0 and len(edges_combined) > 0:
                current_features = feature_extractor(combined_X, training=True).numpy()
                graph_enhanced = graph_attention_aggregation(
                    current_features, edges_combined, edge_weights_combined
                )
                target_graph_features = tf.constant(graph_enhanced[len(source_X):], dtype=tf.float32)
                graph_loss = tf.reduce_mean(tf.reduce_sum(
                    (target_labeled_features - target_graph_features) ** 2, axis=1
                ))
            
            additional_loss = (
                best_params['gamma_mmd'] * mmd +
                best_params['delta_consistency'] * consistency +
                0.01 * graph_loss  # Graph alignment
            )
        
        # Apply additional losses
        trainable_vars = feature_extractor.trainable_variables
        gradients = tape.gradient(additional_loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in gradients]
        combined_model.optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        )
        
        # Validation
        val_results = combined_model.evaluate(
            X_val, [Y_val_class, Y_val_domain],
            verbose=0, batch_size=batch_size
        )
        val_loss = val_results[0]
        
        if (epoch + 1) % 5 == 0:
            print(f"\n  Epoch {epoch+1}/{epochs}:")
            print(f"    Train loss: {history.history['loss'][0]:.4f}")
            print(f"    Val loss: {val_loss:.4f}")
            print(f"    MMD loss: {mmd.numpy():.4f}")
            if target_X_unlabeled is not None:
                print(f"    Consistency loss: {consistency.numpy():.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Save best model
            feature_extractor.save_weights('best_feature_extractor.weights.h5')
            classifier.save_weights('best_classifier.weights.h5')
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    # Load best weights
    feature_extractor.load_weights('best_feature_extractor.weights.h5')
    classifier.load_weights('best_classifier.weights.h5')
    
    return feature_extractor, classifier, {'best_val_loss': best_val_loss}


def main():
    """Main training function with hyperparameter optimization."""
    print("="*70)
    print("DOMAIN ADAPTATION WITH HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    data = load_domain_adaptation_data()
    
    source_X = data['source_X']
    source_Y = data['source_Y']
    source_bridge_id = data['source_bridge_id']
    target_X_labeled = data['target_X_labeled']
    target_Y_labeled = data['target_Y_labeled']
    target_X_unlabeled = data['target_X_unlabeled']
    
    # Hyperparameter search
    print("\n[2/5] Hyperparameter search...")
    best_params, search_results = hyperparameter_search(
        None, None, None,  # Will create fresh models
        source_X, source_Y, target_X_labeled, target_Y_labeled, target_X_unlabeled,
        source_bridge_id
    )
    
    # Full training with best parameters
    print("\n[3/5] Full training...")
    feature_extractor, classifier, training_info = train_full_model_with_best_params(
        best_params, source_X, source_Y, target_X_labeled, target_Y_labeled,
        target_X_unlabeled, source_bridge_id
    )
    
    # Evaluation
    print("\n[4/5] Evaluation...")
    # Evaluate on DC0
    target_features_dc0 = feature_extractor.predict(target_X_labeled, verbose=0)
    target_pred_dc0 = classifier.predict(target_features_dc0, verbose=0)
    target_pred_classes_dc0 = np.argmax(target_pred_dc0, axis=1)
    acc_dc0 = np.mean(target_pred_classes_dc0 == target_Y_labeled)
    print(f"  DC0 accuracy: {acc_dc0:.4f}")
    
    # Evaluate on DC1-DC4
    data_dir = config['DATA_DIR']
    target_bridge = config['TARGET_BRIDGE']
    test_dcs = ['DC1', 'DC2', 'DC3', 'DC4']
    
    all_test_X = []
    all_test_Y = []
    for dc in test_dcs:
        try:
            X_test, Y_test, _ = load_bridge_data(data_dir, target_bridge, dc)
            all_test_X.append(X_test)
            all_test_Y.append(np.full(len(X_test), Y_test))
        except Exception as e:
            print(f"  Failed to load {dc}: {e}")
    
    if all_test_X:
        test_X = np.vstack(all_test_X)
        test_Y = np.concatenate(all_test_Y)
        test_features = feature_extractor.predict(test_X, verbose=0)
        test_pred = classifier.predict(test_features, verbose=0)
        test_pred_classes = np.argmax(test_pred, axis=1)
        acc_test = np.mean(test_pred_classes == test_Y)
        
        print(f"  DC1-DC4 accuracy: {acc_test:.4f}")
        
        # Per-DC results
        start_idx = 0
        for dc in test_dcs:
            if start_idx < len(test_Y):
                dc_size = len(all_test_X[test_dcs.index(dc)])
                end_idx = start_idx + dc_size
                dc_true = test_Y[start_idx:end_idx]
                dc_pred = test_pred_classes[start_idx:end_idx]
                dc_acc = np.mean(dc_pred == dc_true)
                print(f"    {dc}: {dc_acc:.4f}")
                start_idx = end_idx
        
        # Combined accuracy
        all_target_X = np.vstack([target_X_labeled, test_X])
        all_target_Y = np.concatenate([target_Y_labeled, test_Y])
        all_features = feature_extractor.predict(all_target_X, verbose=0)
        all_pred = classifier.predict(all_features, verbose=0)
        all_pred_classes = np.argmax(all_pred, axis=1)
        combined_acc = np.mean(all_pred_classes == all_target_Y)
        print(f"  Combined (DC0-DC4) accuracy: {combined_acc:.4f}")
    
    # Save models
    print("\n[5/5] Saving models...")
    output_dir = Path('domain_adaptation_results_optimized')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    feature_extractor.save(output_dir / f'feature_extractor_{timestamp}.h5')
    classifier.save(output_dir / f'classifier_{timestamp}.h5')
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation loss: {training_info['best_val_loss']:.4f}")
    print(f"Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*70)


if __name__ == '__main__':
    main()

