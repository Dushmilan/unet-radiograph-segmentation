"""
U-Net Architecture for Image Segmentation
Built from scratch using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ========== Custom Loss Functions for Better IoU ==========

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Calculate Dice Coefficient (similar to IoU but emphasizes overlap).
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice Loss = 1 - Dice Coefficient.
    Better than binary crossentropy for imbalanced segmentation.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def iou_loss(y_true, y_pred, smooth=1.0):
    """
    Direct IoU Loss optimization.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return 1.0 - (intersection + smooth) / (union + smooth)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1.0):
    """
    Tversky Loss - Generalization of Dice that handles false positives/negatives.
    alpha=beta=0.5 gives Dice, alpha=beta=0.5 gives IoU.
    Higher alpha penalizes false negatives more (good for small objects).
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_f * y_pred_f)
    false_neg = tf.keras.backend.sum(y_true_f * (1.0 - y_pred_f))
    false_pos = tf.keras.backend.sum((1.0 - y_true_f) * y_pred_f)
    return 1.0 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)


def combined_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5):
    """
    Combined Binary Crossentropy + Dice Loss.
    Often works better than either alone.
    """
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + dice_weight * dice


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=10.0):
    """
    Weighted Binary Crossentropy to handle class imbalance.
    pos_weight: How much to weight positive class (foreground)
    Higher values = more focus on foreground objects.
    """
    # Standard BCE
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    
    # Create weight mask
    weight_mask = tf.ones_like(y_true) + (pos_weight - 1.0) * y_true
    
    # Apply weights
    weighted_bce = tf.keras.backend.mean(bce * weight_mask)
    return weighted_bce


def convolution_block(input_tensor, num_filters):
    """
    Double convolution block for U-Net.
    Two 3x3 convolutions with ReLU activation and BatchNormalization.
    """
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x


def encoder_block(input_tensor, num_filters):
    """
    Encoder block: Convolution block followed by 2x2 max pooling.
    Returns both the output and the skip connection (before pooling).
    """
    x = convolution_block(input_tensor, num_filters)
    pool = layers.MaxPooling2D((2, 2))(x)
    return pool, x


def decoder_block(input_tensor, skip_connection, num_filters):
    """
    Decoder block: Transpose convolution (upsampling) + concatenation with skip connection + convolution block.
    """
    # Upsampling
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    
    # Concatenate with skip connection
    x = layers.concatenate([x, skip_connection])
    
    # Apply convolution block
    x = convolution_block(x, num_filters)
    
    return x


def build_unet(input_height=512, input_width=512, num_classes=1):
    """
    Build U-Net model.
    
    Args:
        input_height: Height of input images
        input_width: Width of input images
        num_classes: Number of output classes (1 for binary segmentation)
    
    Returns:
        U-Net Keras Model
    """
    # Input layer
    inputs = layers.Input(shape=(input_height, input_width, 3))
    
    # Scale inputs to [0, 1]
    x = layers.Lambda(lambda img: img / 255.0)(inputs)
    
    # ========== Encoder Path ==========
    # Block 1: 64 filters
    pool1, conv1 = encoder_block(x, 64)
    
    # Block 2: 128 filters
    pool2, conv2 = encoder_block(pool1, 128)
    
    # Block 3: 256 filters
    pool3, conv3 = encoder_block(pool2, 256)
    
    # Block 4: 512 filters
    pool4, conv4 = encoder_block(pool3, 512)
    
    # ========== Bottleneck ==========
    bottleneck = convolution_block(pool4, 1024)
    
    # ========== Decoder Path ==========
    # Block 1: 512 filters
    up1 = decoder_block(bottleneck, conv4, 512)
    
    # Block 2: 256 filters
    up2 = decoder_block(up1, conv3, 256)
    
    # Block 3: 128 filters
    up3 = decoder_block(up2, conv2, 128)
    
    # Block 4: 64 filters
    up4 = decoder_block(up3, conv1, 64)
    
    # ========== Output Layer ==========
    if num_classes == 1:
        # Binary segmentation
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(up4)
    else:
        # Multi-class segmentation
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(up4)
    
    model = Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    
    return model


def get_compiled_unet(learning_rate=1e-4, input_height=512, input_width=512, num_classes=1,
                      loss_type='combined', pos_weight=10.0):
    """
    Build and compile U-Net model.

    Args:
        learning_rate: Learning rate for optimizer
        input_height: Height of input images
        input_width: Width of input images
        num_classes: Number of output classes
        loss_type: Type of loss function:
                   - 'binary_crossentropy': Standard BCE (baseline)
                   - 'dice': Dice loss (better for imbalance)
                   - 'iou': Direct IoU optimization
                   - 'tversky': Tversky loss (adjustable FP/FN tradeoff)
                   - 'combined': BCE + Dice (recommended)
                   - 'weighted_bce': Weighted BCE for imbalance
        pos_weight: Weight for positive class (used with weighted_bce)

    Returns:
        Compiled U-Net Keras Model
    """
    model = build_unet(input_height, input_width, num_classes)

    # Use Adam optimizer with weight decay for better generalization
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Select loss function based on loss_type
    if num_classes == 1:
        # Binary segmentation - choose loss based on parameter
        if loss_type == 'dice':
            loss = dice_loss
        elif loss_type == 'iou':
            loss = iou_loss
        elif loss_type == 'tversky':
            loss = tversky_loss
        elif loss_type == 'combined':
            loss = combined_loss
        elif loss_type == 'weighted_bce':
            from functools import partial
            loss = partial(weighted_binary_crossentropy, pos_weight=pos_weight)
        else:
            loss = 'binary_crossentropy'
        
        # Metrics for binary segmentation
        metrics = [
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=2),
            dice_coefficient
        ]
    else:
        # Multi-class segmentation
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


if __name__ == "__main__":
    # Test the model architecture
    model = get_compiled_unet(input_height=512, input_width=512, num_classes=1)
    model.summary()
    
    # Print total parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
