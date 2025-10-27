import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
from cnn_data_loader import load_data_sets

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.metrics import TopKCategoricalAccuracy
    from sklearn.metrics import classification_report, confusion_matrix
    print("TensorFlow/Keras imported successfully")
except ImportError as e:
    print(f"ERROR: Error importing TensorFlow/Keras: {e}")
    print("Please install: pip install tensorflow")
    sys.exit(1)


def build_cnn(input_shape, num_classes: int) -> Sequential:
    """Build a 1D CNN model for damage classification."""
    model = Sequential([
        # First convolutional block
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # Second convolutional block
        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # Third convolutional block
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
    )
    
    return model


def build_dense_mlp(input_shape, num_classes: int) -> Sequential:
    """Build a Dense/MLP model for comparison."""
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
    )
    
    return model


def plot_training_history(history, model_name='Model'):
    """Plot training and validation loss/accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('training_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(output_dir / f'{model_name}_{timestamp}_history.png', dpi=300, bbox_inches='tight')
    print(f"\nTraining history saved to {output_dir / f'{model_name}_{timestamp}_history.png'}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name='Model'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=f'{model_name} - Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('training_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(output_dir / f'{model_name}_{timestamp}_confusion.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_dir / f'{model_name}_{timestamp}_confusion.png'}")
    
    plt.show()


def evaluate_model(model, X_test, y_test, class_names, model_name='Model'):
    """Evaluate model and print classification report."""
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print(f"\n{'='*60}")
    print(f"{model_name} Classification Report")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, class_names, model_name)
    
    # Calculate additional metrics
    test_loss, test_acc, top3_acc = model.evaluate(X_test, to_categorical(y_test, len(class_names)), verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")


def main():
    """Main training function."""
    print("="*70)
    print("Bridge Damage Classification - CNN Training")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    try:
        X_train, Y_train, X_test, Y_test = load_data_sets()
        print("Data loaded successfully")
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        sys.exit(1)
    
    # Prepare inputs for CNN
    print("\n[2/6] Preprocessing data...")
    # Data is already in correct shape: (samples, 250, 2)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # One-hot encode labels
    num_classes = int(len(np.unique(Y_train)))
    class_names = [f'DC{i}' for i in range(num_classes)]
    Y_train_cat = to_categorical(Y_train, num_classes)
    Y_test_cat = to_categorical(Y_test, num_classes)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Number of classes: {num_classes} {class_names}")
    
    # Create output directory
    output_dir = Path('training_results')
    output_dir.mkdir(exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(output_dir / 'best_cnn_model.h5'), 
                       monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    # Train CNN
    print("\n[3/6] Building CNN model...")
    cnn_model = build_cnn(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    print("Model built successfully")
    print("\nModel Architecture:")
    cnn_model.summary()
    
    print("\n[4/6] Training CNN model...")
    history_cnn = cnn_model.fit(
        X_train, Y_train_cat,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, Y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cnn_model.save(output_dir / f'cnn_model_{timestamp}.h5')
    print(f"\nModel saved to {output_dir / f'cnn_model_{timestamp}.h5'}")
    
    # Evaluate CNN
    print("\n[5/6] Evaluating CNN model...")
    evaluate_model(cnn_model, X_test, Y_test, class_names, 'CNN')
    
    # Plot training history
    print("\n[6/6] Plotting training history...")
    plot_training_history(history_cnn, 'CNN')
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

