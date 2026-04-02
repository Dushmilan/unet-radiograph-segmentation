"""
Training script for U-Net segmentation model.
GPU-ONLY: This script requires a working GPU setup.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

from unet_model import get_compiled_unet
from data_loader import create_data_generators


def check_gpu_requirement():
    """
    Check if GPU is available. Exit if not.
    This script requires GPU for training.
    """
    print("\n" + "=" * 80)
    print("GPU REQUIREMENT CHECK")
    print("=" * 80)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("\n❌ ERROR: No GPU detected!")
        print("\nThis training script requires a GPU. Please fix your GPU setup before proceeding.")
        print("\nTroubleshooting steps:")
        print("  1. Run: python test_gpu_comprehensive.py")
        print("  2. Follow the recommendations from the diagnostic")
        print("  3. Ensure CUDA paths are in system PATH")
        print("  4. Restart your computer after PATH changes")
        print("  5. Run this script again")
        print("\nFor detailed GPU setup instructions, see GPU_SETUP.md")
        sys.exit(1)
    
    print(f"\n✓ GPU(s) detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Enable memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n✓ Memory growth enabled for all GPUs")
    except RuntimeError as e:
        print(f"\n⚠ Warning: Could not enable memory growth: {e}")
    
    print("\n" + "=" * 80)
    print("GPU CHECK PASSED - Starting training...")
    print("=" * 80 + "\n")
    
    return gpus


def train_unet(
    data_dir,
    model_dir='models',
    logs_dir='logs',
    batch_size=8,
    epochs=100,
    input_height=512,
    input_width=512,
    learning_rate=1e-4,
    augment=True
):
    """
    Train U-Net model for image segmentation.
    
    Args:
        data_dir: Directory containing the dataset
        model_dir: Directory to save model checkpoints
        logs_dir: Directory to save training logs
        batch_size: Batch size for training
        epochs: Number of training epochs
        input_height: Input image height
        input_width: Input image width
        learning_rate: Initial learning rate
        augment: Whether to use data augmentation
    """
    # Check GPU requirement FIRST
    gpus = check_gpu_requirement()
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("=" * 60)
    print("U-Net Training for Image Segmentation")
    print("=" * 60)
    
    # Create data generators
    print("\n[1/4] Creating data generators...")
    train_gen, val_gen = create_data_generators(
        data_dir,
        batch_size=batch_size,
        target_height=input_height,
        target_width=input_width,
        augment=augment
    )
    
    print(f"  Training samples: {len(train_gen) * batch_size}")
    print(f"  Validation samples: {len(val_gen) * batch_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input size: {input_height}x{input_width}")
    
    # Create model
    print("\n[2/4] Building U-Net model...")
    model = get_compiled_unet(
        learning_rate=learning_rate,
        input_height=input_height,
        input_width=input_width,
        num_classes=1
    )
    model.summary()
    
    # Define callbacks
    print("\n[3/4] Setting up callbacks...")
    
    # Model checkpoint - save best model
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'unet_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    
    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        min_delta=1e-4,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )
    
    # Reduce learning rate on plateau
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        min_delta=1e-4,
        mode='min',
        verbose=1
    )
    
    # TensorBoard
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(logs_dir, 'training'),
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    
    # CSV logger
    csv_logger_callback = CSVLogger(
        os.path.join(logs_dir, 'training_history.csv'),
        append=False,
        separator=','
    )
    
    callbacks = [
        checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback,
        tensorboard_callback,
        csv_logger_callback
    ]
    
    # Train model
    print("\n[4/4] Starting training...")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_dir, 'unet_final.h5'))
    print(f"\nModel saved to {os.path.join(model_dir, 'unet_final.h5')}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model, history


def evaluate_model(model, data_dir, batch_size=8, input_height=512, input_width=512):
    """
    Evaluate trained model on test data.
    
    Args:
        model: Trained U-Net model
        data_dir: Directory containing the dataset
        batch_size: Batch size
        input_height: Input image height
        input_width: Input image width
    
    Returns:
        Evaluation metrics
    """
    print("\nEvaluating model...")
    
    _, val_gen = create_data_generators(
        data_dir,
        batch_size=batch_size,
        target_height=input_height,
        target_width=input_width,
        augment=False
    )
    
    # Evaluate
    results = model.evaluate(val_gen, verbose=1)
    
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    for metric, value in zip(model.metrics_names, results):
        print(f"  {metric}: {value:.4f}")
    
    return results


def predict_and_visualize(model, image_path, target_height=512, target_width=512):
    """
    Make prediction on a single image and visualize results.
    
    Args:
        model: Trained U-Net model
        image_path: Path to the image
        target_height: Target height for resizing
        target_width: Target width for resizing
    
    Returns:
        Original image, predicted mask
    """
    import cv2
    import matplotlib.pyplot as plt
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    
    # Resize for model input
    image_resized = cv2.resize(image, (target_width, target_height))
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Predict
    input_tensor = np.expand_dims(image_normalized, axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    
    # Resize prediction back to original size
    prediction_resized = cv2.resize(
        prediction[0, :, :, 0], 
        (orig_width, orig_height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(prediction_resized, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[prediction_resized > 0.5] = [255, 0, 0]
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red = Predicted)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image, prediction_resized


if __name__ == "__main__":
    # Configuration
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'TUFTS')
    MODEL_DIR = 'models'
    LOGS_DIR = 'logs'
    
    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 100
    INPUT_HEIGHT = 512
    INPUT_WIDTH = 512
    LEARNING_RATE = 1e-4
    
    # Train the model
    model, history = train_unet(
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        logs_dir=LOGS_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        learning_rate=LEARNING_RATE,
        augment=True
    )
    
    # Evaluate
    evaluate_model(
        model,
        DATA_DIR,
        batch_size=BATCH_SIZE,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH
    )
    
    print("\nTraining complete! You can now use the model for predictions.")
    print(f"Best model saved at: {os.path.join(MODEL_DIR, 'unet_best.h5')}")
