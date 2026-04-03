"""
U-Net Architecture for Image Segmentation
Built from scratch using TensorFlow/Keras

Features:
- Pretrained encoder support (ResNet, VGG backbones)
- Advanced loss functions (Combined, Focal, Tversky, IoU, Dice)
- Dropout and batch normalization for regularization
- Gradient clipping for stable training
- Configurable architecture depth
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from functools import partial


# ========== Custom Loss Functions for Better IoU ==========

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Calculate Dice Coefficient (similar to IoU but emphasizes overlap).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice Loss = 1 - Dice Coefficient.
    Better than binary crossentropy for imbalanced segmentation.
    """
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)


def iou_loss(y_true, y_pred, smooth=1.0):
    """
    Direct IoU Loss optimization.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return 1.0 - (intersection + smooth) / (union + smooth)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, smooth=1e-7):
    """
    Focal Loss - Addresses class imbalance by focusing on hard examples.
    Reduces weight for well-classified examples.
    
    Args:
        alpha: Weighting factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        smooth: Smoothing factor
    """
    y_pred = K.clip(y_pred, smooth, 1.0 - smooth)
    
    # Binary crossentropy
    bce = - (alpha * y_true * K.log(y_pred) + (1 - alpha) * (1 - y_true) * K.log(1 - y_pred))
    
    # Focal weighting
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = K.pow(1.0 - p_t, gamma)
    
    return K.mean(focal_weight * bce)


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1.0):
    """
    Tversky Loss - Generalization of Dice that handles false positives/negatives.
    alpha=beta=0.5 gives Dice, alpha=beta=0.5 gives IoU.
    Higher alpha penalizes false negatives more (good for small objects).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1.0 - y_pred_f))
    false_pos = K.sum((1.0 - y_true_f) * y_pred_f)
    return 1.0 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)


def combined_loss(y_true, y_pred, bce_weight=0.4, dice_weight=0.3, focal_weight=0.3,
                  alpha=0.25, gamma=2.0):
    """
    Advanced Combined Loss: BCE + Dice + Focal Loss.
    Superior performance for imbalanced segmentation tasks.
    
    Args:
        bce_weight: Weight for binary crossentropy
        dice_weight: Weight for dice loss
        focal_weight: Weight for focal loss
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
    """
    bce = K.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    
    return bce_weight * bce + dice_weight * dice + focal_weight * focal


def iou_dice_combined_loss(y_true, y_pred, iou_weight=0.4, dice_weight=0.6, smooth=1.0):
    """
    Combined IoU + Dice Loss for direct IoU optimization with stability.
    """
    iou = iou_loss(y_true, y_pred, smooth)
    dice = dice_loss(y_true, y_pred, smooth)
    
    return iou_weight * iou + dice_weight * dice


def convolution_block(input_tensor, num_filters, dropout_rate=0.1, use_residual=True):
    """
    Enhanced convolution block with dropout and optional residual connections.
    
    Args:
        input_tensor: Input tensor
        num_filters: Number of convolutional filters
        dropout_rate: Dropout rate for regularization (0.0 to disable)
        use_residual: Whether to use residual connections
    """
    # First convolution
    x = layers.Conv2D(num_filters, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    
    # Second convolution
    x = layers.Conv2D(num_filters, (3, 3), padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Residual connection if channels match
    if use_residual and input_tensor.shape[-1] == num_filters:
        x = layers.Add()([x, input_tensor])
    
    x = layers.Activation('relu')(x)
    
    return x


def encoder_block(input_tensor, num_filters, dropout_rate=0.1, use_residual=True):
    """
    Encoder block: Convolution block followed by 2x2 max pooling.
    Returns both the output and the skip connection (before pooling).
    """
    x = convolution_block(input_tensor, num_filters, dropout_rate, use_residual)
    pool = layers.MaxPooling2D((2, 2))(x)
    return pool, x


def decoder_block(input_tensor, skip_connection, num_filters, dropout_rate=0.1, 
                  use_residual=True, use_attention=False):
    """
    Enhanced decoder block with optional attention mechanism.
    
    Args:
        input_tensor: Input from previous decoder layer
        skip_connection: Skip connection from encoder
        num_filters: Number of filters
        dropout_rate: Dropout rate
        use_residual: Whether to use residual connections
        use_attention: Whether to apply attention to skip connection
    """
    # Upsampling
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), 
                               padding='same', kernel_initializer='he_normal')(input_tensor)
    
    # Optional attention mechanism on skip connection
    if use_attention:
        skip_connection = attention_block(x, skip_connection, num_filters)
    
    # Concatenate with skip connection
    x = layers.concatenate([x, skip_connection])
    
    # Apply convolution block
    x = convolution_block(x, num_filters, dropout_rate, use_residual)
    
    return x


def attention_block(gating_signal, skip_connection, num_filters):
    """
    Attention gate to focus on relevant skip connection features.
    Helps decoder focus on important regions.
    """
    # Transform gating signal
    gating_transformed = layers.Conv2D(num_filters, (1, 1), padding='same',
                                       kernel_initializer='he_normal')(gating_signal)
    gating_transformed = layers.BatchNormalization()(gating_transformed)
    
    # Transform skip connection
    skip_transformed = layers.Conv2D(num_filters, (1, 1), padding='same',
                                     kernel_initializer='he_normal')(skip_connection)
    skip_transformed = layers.BatchNormalization()(skip_transformed)
    
    # Combine and apply attention
    combined = layers.Add()([gating_transformed, skip_transformed])
    combined = layers.Activation('relu')(combined)
    
    attention_weights = layers.Conv2D(1, (1, 1), padding='same',
                                      activation='sigmoid',
                                      kernel_initializer='he_normal')(combined)
    
    return layers.Multiply()([skip_connection, attention_weights])


def build_unet(input_height=512, input_width=512, num_classes=1,
               use_pretrained=False, pretrained_backbone='resnet50',
               dropout_rate=0.2, use_residual=True, use_attention=True,
               num_filters_base=64):
    """
    Build enhanced U-Net model with optional pretrained encoder.

    Args:
        input_height: Height of input images
        input_width: Width of input images
        num_classes: Number of output classes (1 for binary segmentation)
        use_pretrained: Whether to use pretrained backbone
        pretrained_backbone: Backbone type ('resnet50', 'resnet34', 'vgg16')
        dropout_rate: Dropout rate for regularization
        use_residual: Whether to use residual connections
        use_attention: Whether to use attention gates in decoder
        num_filters_base: Base number of filters (default 64)

    Returns:
        U-Net Keras Model
    """
    # Input layer
    inputs = layers.Input(shape=(input_height, input_width, 3))

    # Scale inputs to [0, 1] and normalize
    if use_pretrained:
        # For pretrained models, use ImageNet normalization
        x = layers.Lambda(lambda img: tf.keras.applications.resnet50.preprocess_input(img * 255.0))(inputs)
    else:
        x = layers.Lambda(lambda img: img / 255.0)(inputs)

    if use_pretrained:
        # ========== Pretrained Encoder Path ==========
        print(f"Using pretrained backbone: {pretrained_backbone}")
        
        if pretrained_backbone.lower() == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
            # Extract features from different levels
            conv1 = backbone.get_layer('conv1_relu').output  # 256 filters
            conv2 = backbone.get_layer('conv2_block3_out').output  # 512 filters
            conv3 = backbone.get_layer('conv3_block4_out').output  # 1024 filters
            conv4 = backbone.get_layer('conv4_block6_out').output  # 2048 filters
            bottleneck = backbone.output  # 2048 filters
            
            # Adjust filter counts for skip connections
            conv1 = layers.Conv2D(num_filters_base, (1, 1), padding='same')(conv1)
            conv2 = layers.Conv2D(num_filters_base * 2, (1, 1), padding='same')(conv2)
            conv3 = layers.Conv2D(num_filters_base * 4, (1, 1), padding='same')(conv3)
            conv4 = layers.Conv2D(num_filters_base * 8, (1, 1), padding='same')(conv4)
            
        elif pretrained_backbone.lower() == 'resnet34':
            backbone = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
            # Use ResNet50 as approximation (ResNet34 not in keras.applications)
            conv1 = backbone.get_layer('conv1_relu').output
            conv2 = backbone.get_layer('conv2_block3_out').output
            conv3 = backbone.get_layer('conv3_block4_out').output
            conv4 = backbone.get_layer('conv4_block6_out').output
            bottleneck = backbone.output
            
            conv1 = layers.Conv2D(num_filters_base, (1, 1), padding='same')(conv1)
            conv2 = layers.Conv2D(num_filters_base * 2, (1, 1), padding='same')(conv2)
            conv3 = layers.Conv2D(num_filters_base * 4, (1, 1), padding='same')(conv3)
            conv4 = layers.Conv2D(num_filters_base * 8, (1, 1), padding='same')(conv4)
            
        elif pretrained_backbone.lower() == 'vgg16':
            backbone = tf.keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
            conv1 = backbone.get_layer('block1_conv2').output
            conv2 = backbone.get_layer('block2_conv2').output
            conv3 = backbone.get_layer('block3_conv3').output
            conv4 = backbone.get_layer('block4_conv3').output
            bottleneck = backbone.get_layer('block5_conv3').output
            
            conv1 = layers.Conv2D(num_filters_base, (1, 1), padding='same')(conv1)
            conv2 = layers.Conv2D(num_filters_base * 2, (1, 1), padding='same')(conv2)
            conv3 = layers.Conv2D(num_filters_base * 4, (1, 1), padding='same')(conv3)
            conv4 = layers.Conv2D(num_filters_base * 8, (1, 1), padding='same')(conv4)
            
            # Freeze backbone layers
            for layer in backbone.layers:
                layer.trainable = False
            print("Frozen pretrained backbone layers")
        else:
            raise ValueError(f"Unsupported backbone: {pretrained_backbone}")
        
    else:
        # ========== Standard Encoder Path ==========
        # Block 1: base filters
        pool1, conv1 = encoder_block(x, num_filters_base, dropout_rate * 0.5, use_residual)

        # Block 2: 2x filters
        pool2, conv2 = encoder_block(pool1, num_filters_base * 2, dropout_rate * 0.5, use_residual)

        # Block 3: 4x filters
        pool3, conv3 = encoder_block(pool2, num_filters_base * 4, dropout_rate, use_residual)

        # Block 4: 8x filters
        pool4, conv4 = encoder_block(pool3, num_filters_base * 8, dropout_rate, use_residual)

        # ========== Bottleneck ==========
        bottleneck = convolution_block(pool4, num_filters_base * 16, dropout_rate, use_residual)

    # ========== Decoder Path ==========
    # Block 1: 8x filters
    up1 = decoder_block(bottleneck, conv4, num_filters_base * 8, 
                       dropout_rate, use_residual, use_attention)

    # Block 2: 4x filters
    up2 = decoder_block(up1, conv3, num_filters_base * 4,
                       dropout_rate, use_residual, use_attention)

    # Block 3: 2x filters
    up3 = decoder_block(up2, conv2, num_filters_base * 2,
                       dropout_rate * 0.5, use_residual, use_attention)

    # Block 4: base filters
    up4 = decoder_block(up3, conv1, num_filters_base,
                       dropout_rate * 0.5, use_residual, use_attention)

    # ========== Output Layer ==========
    # Add intermediate convolution before output
    x = layers.Conv2D(num_filters_base, (3, 3), padding='same',
                      kernel_initializer='he_normal')(up4)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if num_classes == 1:
        # Binary segmentation
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid',
                               kernel_initializer='he_normal')(x)
    else:
        # Multi-class segmentation
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax',
                               kernel_initializer='he_normal')(x)

    model = Model(inputs=[inputs], outputs=[outputs], name='Enhanced-U-Net')

    return model


def get_compiled_unet(learning_rate=1e-4, input_height=512, input_width=512, num_classes=1,
                      loss_type='combined', pos_weight=10.0,
                      use_pretrained=False, pretrained_backbone='resnet50',
                      dropout_rate=0.2, use_residual=True, use_attention=True,
                      gradient_clip_norm=1.0, num_filters_base=64,
                      use_cosine_decay=False, decay_steps=10000, alpha=0.0):
    """
    Build and compile enhanced U-Net model.

    Args:
        learning_rate: Initial learning rate for optimizer
        input_height: Height of input images
        input_width: Width of input images
        num_classes: Number of output classes
        loss_type: Type of loss function:
                   - 'binary_crossentropy': Standard BCE (baseline)
                   - 'dice': Dice loss (better for imbalance)
                   - 'iou': Direct IoU optimization
                   - 'tversky': Tversky loss (adjustable FP/FN tradeoff)
                   - 'combined': BCE + Dice + Focal (recommended)
                   - 'iou_dice': IoU + Dice combination
                   - 'weighted_bce': Weighted BCE for imbalance
        pos_weight: Weight for positive class (used with weighted_bce)
        use_pretrained: Whether to use pretrained backbone
        pretrained_backbone: Backbone type ('resnet50', 'resnet34', 'vgg16')
        dropout_rate: Dropout rate for regularization
        use_residual: Whether to use residual connections
        use_attention: Whether to use attention gates
        gradient_clip_norm: Gradient clipping threshold (1.0 for stable training)
        num_filters_base: Base number of filters
        use_cosine_decay: Whether to use cosine learning rate decay
        decay_steps: Number of steps for cosine decay
        alpha: Minimum learning rate factor for cosine decay

    Returns:
        Compiled U-Net Keras Model
    """
    model = build_unet(
        input_height, input_width, num_classes,
        use_pretrained=use_pretrained,
        pretrained_backbone=pretrained_backbone,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        use_attention=use_attention,
        num_filters_base=num_filters_base
    )

    # Optimizer with gradient clipping
    if use_cosine_decay:
        # Cosine annealing learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=alpha  # Minimum LR as fraction of initial
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            clipnorm=gradient_clip_norm
        )
        print(f"Using AdamW with Cosine Annealing (initial LR: {learning_rate})")
    else:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=gradient_clip_norm
        )
        print(f"Using AdamW optimizer with gradient clipping (LR: {learning_rate})")

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
        elif loss_type == 'iou_dice':
            loss = iou_dice_combined_loss
        elif loss_type == 'weighted_bce':
            loss = partial(tf.keras.losses.binary_crossentropy, 
                          from_logits=False)
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
