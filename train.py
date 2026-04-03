"""
Training script for U-Net segmentation model.
GPU-ONLY: This script requires a working GPU setup.
"""

import os
import sys

# Configure CUDA DLL paths BEFORE importing TensorFlow
import configure_cuda

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
    augment=True,
    loss_type='combined',
    use_pretrained=False,
    pretrained_backbone='resnet50',
    dropout_rate=0.2,
    use_residual=True,
    use_attention=True,
    use_cosine_decay=False,
    use_patch_based=False,
    patch_size=256,
    unfreeze_epoch=50
):
    """
    Train enhanced U-Net model for image segmentation.

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
        loss_type: Type of loss function ('combined', 'dice', 'iou', 'iou_dice', 'tversky')
        use_pretrained: Whether to use pretrained backbone
        pretrained_backbone: Backbone type ('resnet50', 'vgg16')
        dropout_rate: Dropout rate for regularization
        use_residual: Whether to use residual connections
        use_attention: Whether to use attention gates
        use_cosine_decay: Whether to use cosine learning rate decay
        use_patch_based: Whether to use patch-based training
        patch_size: Size of patches for patch-based training
        unfreeze_epoch: Epoch to unfreeze pretrained backbone (if use_pretrained=True)
    """
    # Check GPU requirement FIRST
    gpus = check_gpu_requirement()

    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 60)
    print("Enhanced U-Net Training for Image Segmentation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Loss function: {loss_type}")
    print(f"  Pretrained backbone: {pretrained_backbone if use_pretrained else 'None (training from scratch)'}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Residual connections: {use_residual}")
    print(f"  Attention gates: {use_attention}")
    print(f"  Cosine LR decay: {use_cosine_decay}")
    print(f"  Patch-based training: {use_patch_based}")
    print(f"  Data augmentation: {augment}")

    # Create data generators
    print("\n[1/4] Creating data generators...")
    train_gen, val_gen = create_data_generators(
        data_dir,
        batch_size=batch_size,
        target_height=input_height,
        target_width=input_width,
        augment=augment,
        use_patch_based=use_patch_based,
        patch_size=patch_size
    )

    print(f"  Training samples: {len(train_gen) * batch_size}")
    print(f"  Validation samples: {len(val_gen) * batch_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input size: {input_height}x{input_width}")

    # Create model
    print("\n[2/4] Building enhanced U-Net model...")
    
    # Calculate decay steps for cosine annealing
    decay_steps = len(train_gen) * epochs if use_cosine_decay else 10000
    
    model = get_compiled_unet(
        learning_rate=learning_rate,
        input_height=input_height,
        input_width=input_width,
        num_classes=1,
        loss_type=loss_type,
        use_pretrained=use_pretrained,
        pretrained_backbone=pretrained_backbone,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        use_attention=use_attention,
        use_cosine_decay=use_cosine_decay,
        decay_steps=decay_steps,
        alpha=0.01  # Minimum LR = 1% of initial
    )
    model.summary()

    # Define callbacks
    print("\n[3/4] Setting up callbacks...")

    # Model checkpoint - save best model based on IoU
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'unet_best.h5'),
        monitor='val_mean_io_u',  # Monitor IoU instead of loss
        save_best_only=True,
        save_weights_only=False,
        mode='max',  # We want to maximize IoU
        verbose=1
    )

    # Early stopping - based on IoU
    early_stopping_callback = EarlyStopping(
        monitor='val_mean_io_u',
        patience=25,  # Increased patience
        min_delta=0.005,  # Minimum IoU improvement
        mode='max',
        verbose=1,
        restore_best_weights=True
    )

    # Reduce learning rate on plateau - monitor IoU
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_mean_io_u',
        factor=0.5,
        patience=12,
        min_lr=1e-7,
        min_delta=0.003,
        mode='max',
        verbose=1
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
        csv_logger_callback
    ]
    
    # Optional: Unfreeze pretrained backbone after initial training
    if use_pretrained and unfreeze_epoch is not None:
        def unfreeze_backbone(epoch):
            if epoch == unfreeze_epoch:
                print(f"\n>>> Unfreezing pretrained backbone at epoch {epoch} <<<")
                for layer in model.layers:
                    if hasattr(layer, 'trainable'):
                        layer.trainable = True
                model.compile(
                    optimizer=tf.keras.optimizers.AdamW(
                        learning_rate=learning_rate * 0.1,  # Lower LR for fine-tuning
                        weight_decay=1e-4,
                        clipnorm=1.0
                    ),
                    loss=model.loss,
                    metrics=model.metrics
                )
        
        from tensorflow.keras.callbacks import LambdaCallback
        unfreeze_callback = LambdaCallback(on_epoch_begin=unfreeze_backbone)
        callbacks.append(unfreeze_callback)

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
    BATCH_SIZE = 4  # Reduced for better gradient updates
    EPOCHS = 150  # Increased for better convergence
    INPUT_HEIGHT = 512
    INPUT_WIDTH = 512
    LEARNING_RATE = 1e-4

    # ============================================================
    # IMPROVEMENTS CONFIGURATION
    # ============================================================
    
    # Loss function selection:
    # 'combined' - BCE + Dice + Focal (recommended, best performance)
    # 'iou_dice' - IoU + Dice combination (direct IoU optimization)
    # 'dice' - Dice loss only (better for class imbalance)
    # 'iou' - Direct IoU optimization
    # 'tversky' - Tversky loss (adjustable FP/FN tradeoff)
    LOSS_TYPE = 'combined'
    
    # Pretrained backbone (set to True to use transfer learning):
    # False - Train from scratch (current default)
    # True - Use pretrained ResNet50/VGG16 backbone
    USE_PRETRAINED = False
    PRETRAINED_BACKBONE = 'resnet50'  # Options: 'resnet50', 'vgg16'
    
    # Regularization:
    DROPOUT_RATE = 0.2  # Increased from 0.1 for better regularization
    USE_RESIDUAL = True  # Residual connections for better gradient flow
    USE_ATTENTION = True  # Attention gates in decoder
    
    # Learning rate scheduling:
    USE_COSINE_DECAY = True  # Cosine annealing for better convergence
    
    # Data augmentation:
    AUGMENT = True  # Enhanced augmentation (elastic, noise, blur, etc.)
    
    # Patch-based training (optional):
    USE_PATCH_BASED = False  # Set True for patch-based training
    PATCH_SIZE = 256  # Patch size for patch-based training
    
    # ============================================================

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
        augment=AUGMENT,
        loss_type=LOSS_TYPE,
        use_pretrained=USE_PRETRAINED,
        pretrained_backbone=PRETRAINED_BACKBONE,
        dropout_rate=DROPOUT_RATE,
        use_residual=USE_RESIDUAL,
        use_attention=USE_ATTENTION,
        use_cosine_decay=USE_COSINE_DECAY,
        use_patch_based=USE_PATCH_BASED,
        patch_size=PATCH_SIZE
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
    print(f"Final model saved at: {os.path.join(MODEL_DIR, 'unet_final.h5')}")
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    print(f"  Loss type: {LOSS_TYPE}")
    print(f"  Pretrained: {USE_PRETRAINED}")
    print(f"  Dropout: {DROPOUT_RATE}")
    print(f"  Residual: {USE_RESIDUAL}")
    print(f"  Attention: {USE_ATTENTION}")
    print(f"  Cosine LR decay: {USE_COSINE_DECAY}")
    print(f"  Enhanced augmentation: {AUGMENT}")
    
    if hasattr(history, 'history'):
        best_iou = max(history.history.get('val_mean_io_u', [0]))
        best_dice = max(history.history.get('val_dice_coefficient', [0]))
        print(f"\n  Best Validation IoU: {best_iou:.4f}")
        print(f"  Best Validation Dice: {best_dice:.4f}")
