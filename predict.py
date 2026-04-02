"""
Prediction script for U-Net segmentation model.
Use this to make predictions on new images after training.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def load_trained_model(model_path):
    """
    Load a trained U-Net model.
    
    Args:
        model_path: Path to the saved model (.h5 file)
    
    Returns:
        Loaded Keras model
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, target_height=512, target_width=512):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path: Path to the image
        target_height: Target height for resizing
        target_width: Target width for resizing
    
    Returns:
        Preprocessed image tensor, original image, original dimensions
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    
    # Resize for model input
    image_resized = cv2.resize(image, (target_width, target_height))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_tensor = np.expand_dims(image_normalized, axis=0)
    
    return input_tensor, image, (orig_height, orig_width)


def predict(model, image_path, target_height=512, target_width=512, threshold=0.5):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained U-Net model
        image_path: Path to the image
        target_height: Target height for resizing
        target_width: Target width for resizing
        threshold: Threshold for binary mask
    
    Returns:
        Predicted mask (original size), confidence map
    """
    # Preprocess
    input_tensor, image, orig_dims = preprocess_image(
        image_path, target_height, target_width
    )
    
    # Predict
    print("Making prediction...")
    prediction = model.predict(input_tensor, verbose=0)
    
    # Get confidence map
    confidence_map = prediction[0, :, :, 0]
    
    # Resize prediction back to original size
    prediction_resized = cv2.resize(
        confidence_map,
        (orig_dims[1], orig_dims[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Apply threshold
    binary_mask = (prediction_resized > threshold).astype(np.uint8)
    
    return binary_mask, prediction_resized, image


def visualize_prediction(image, binary_mask, confidence_map, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        image: Original image (RGB)
        binary_mask: Binary segmentation mask
        confidence_map: Confidence map from model
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Confidence map
    im1 = axes[1].imshow(confidence_map, cmap='viridis')
    axes[1].set_title('Confidence Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(f'Binary Mask (threshold=0.5)', fontsize=12)
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[binary_mask > 0] = [255, 0, 0]  # Red overlay
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red = Segmented Region)', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def predict_batch(model, image_dir, output_dir='predictions', 
                  target_height=512, target_width=512, threshold=0.5):
    """
    Make predictions on all images in a directory.
    
    Args:
        model: Trained U-Net model
        image_dir: Directory containing images
        output_dir: Directory to save predictions
        target_height: Target height for resizing
        target_width: Target width for resizing
        threshold: Threshold for binary mask
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for i, image_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {image_file}...")
        
        image_path = os.path.join(image_dir, image_file)
        binary_mask, confidence_map, image = predict(
            model, image_path, target_height, target_width, threshold
        )
        
        # Save binary mask
        mask_path = os.path.join(output_dir, f"mask_{image_file}")
        cv2.imwrite(mask_path, binary_mask * 255)
        
        # Save overlay
        overlay = image.copy()
        overlay[binary_mask > 0] = [255, 0, 0]
        overlay_path = os.path.join(output_dir, f"overlay_{image_file}")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='U-Net Prediction Script')
    parser.add_argument('--model', type=str, default='models/unet_best.h5',
                        help='Path to trained model')
    parser.add_argument('--image', type=str, 
                        default='TUFTS/Radiographs/testing_images/1031.JPG',
                        help='Path to image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of images for batch prediction')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--height', type=int, default=512,
                        help='Target image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Target image width')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.model)
    
    if args.image_dir:
        # Batch prediction
        predict_batch(
            model, 
            args.image_dir, 
            args.output_dir,
            args.height, 
            args.width, 
            args.threshold
        )
    else:
        # Single image prediction
        binary_mask, confidence_map, image = predict(
            model, 
            args.image, 
            args.height, 
            args.width, 
            args.threshold
        )
        
        # Visualize
        visualize_prediction(
            image, 
            binary_mask, 
            confidence_map,
            save_path=os.path.join(args.output_dir, 'prediction_visualization.png')
        )
