"""
U-Net Architecture for Image Segmentation
Built from scratch using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


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


def get_compiled_unet(learning_rate=1e-4, input_height=512, input_width=512, num_classes=1):
    """
    Build and compile U-Net model.
    
    Args:
        learning_rate: Learning rate for optimizer
        input_height: Height of input images
        input_width: Width of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled U-Net Keras Model
    """
    model = build_unet(input_height, input_width, num_classes)
    
    # Use Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Choose loss based on number of classes
    if num_classes == 1:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    else:
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
