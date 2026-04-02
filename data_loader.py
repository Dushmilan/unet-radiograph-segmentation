"""
Data loading and preprocessing for U-Net segmentation.
Creates segmentation masks from bounding box annotations.
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence


def load_bbox_annotations(csv_path):
    """
    Load bounding box annotations from CSV file.
    
    Args:
        csv_path: Path to the CSV file with bounding box annotations
    
    Returns:
        Dictionary mapping image IDs to list of bounding boxes
    """
    df = pd.read_csv(csv_path)
    
    # Group by image ID
    annotations = {}
    for _, row in df.iterrows():
        image_id = row['imageID']
        bbox = {
            'class': int(row['class']),
            'x_min': int(row['x-min']),
            'y_min': int(row['y-min']),
            'width': int(row['width']),
            'height': int(row['height'])
        }
        
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(bbox)
    
    return annotations


def create_segmentation_mask(image_height, image_width, bboxes, mode='binary'):
    """
    Create segmentation mask from bounding boxes.
    
    Args:
        image_height: Height of the image
        image_width: Width of the image
        bboxes: List of bounding box dictionaries
        mode: 'binary' for single class, 'multi' for class-specific masks
    
    Returns:
        Segmentation mask as numpy array
    """
    if mode == 'binary':
        # Single channel binary mask
        mask = np.zeros((image_height, image_width, 1), dtype=np.float32)
        for bbox in bboxes:
            x_min = bbox['x_min']
            y_min = bbox['y_min']
            x_max = x_min + bbox['width']
            y_max = y_min + bbox['height']
            
            # Clip to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_width, x_max)
            y_max = min(image_height, y_max)
            
            mask[y_min:y_max, x_min:x_max] = 1.0
        
        return mask
    
    elif mode == 'multi':
        # Multi-class mask (one channel per class + background)
        num_classes = max([bbox['class'] for bbox in bboxes]) + 1 if bboxes else 1
        mask = np.zeros((image_height, image_width, num_classes + 1), dtype=np.float32)
        
        # Set background
        mask[:, :, 0] = 1.0
        
        for bbox in bboxes:
            x_min = bbox['x_min']
            y_min = bbox['y_min']
            x_max = x_min + bbox['width']
            y_max = y_min + bbox['height']
            
            # Clip to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_width, x_max)
            y_max = min(image_height, y_max)
            
            # Set class mask and remove from background
            mask[y_min:y_max, x_min:x_max, bbox['class']] = 1.0
            mask[y_min:y_max, x_min:x_max, 0] = 0.0
        
        return mask
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


class SegmentationDataGenerator(Sequence):
    """
    TensorFlow Sequence for efficient data loading and augmentation.
    """
    
    def __init__(self, 
                 image_dir, 
                 annotations, 
                 batch_size=8, 
                 target_height=512, 
                 target_width=512,
                 augment=False,
                 shuffle=True):
        """
        Initialize the data generator.
        
        Args:
            image_dir: Directory containing images
            annotations: Dictionary of image annotations
            batch_size: Batch size
            target_height: Target height for resizing
            target_width: Target width for resizing
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment
        self.shuffle = shuffle
        
        # Get list of image IDs
        self.image_ids = list(annotations.keys())
        
        if self.shuffle:
            np.random.shuffle(self.image_ids)
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.image_ids) / self.batch_size))
    
    def __getitem__(self, idx):
        """Return a batch of images and masks."""
        # Get image IDs for this batch
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_ids))
        batch_image_ids = self.image_ids[start_idx:end_idx]
        
        images = []
        masks = []
        
        for image_id in batch_image_ids:
            # Load image
            image_path = os.path.join(self.image_dir, image_id)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            orig_height, orig_width = image.shape[:2]
            
            # Resize image
            image_resized = cv2.resize(image, (self.target_width, self.target_height))
            
            # Get bounding boxes for this image
            bboxes = self.annotations.get(image_id, [])
            
            # Create mask
            mask = create_segmentation_mask(
                orig_height, orig_width, bboxes, mode='binary'
            )
            mask_resized = cv2.resize(
                mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST
            )
            
            # Apply augmentation if enabled
            if self.augment:
                image_resized, mask_resized = self._augment(image_resized, mask_resized)
            
            # Normalize image
            image_resized = image_resized.astype(np.float32) / 255.0
            
            images.append(image_resized)
            masks.append(mask_resized)
        
        return np.array(images), np.array(masks)
    
    def _augment(self, image, mask):
        """Apply random augmentation to image and mask."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Random rotation (small angles)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h), 
                                  flags=cv2.INTER_NEAREST)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.image_ids)


def create_data_generators(data_dir, batch_size=8, target_height=512, target_width=512, augment=True):
    """
    Create training and validation data generators.
    
    Args:
        data_dir: Base directory containing the dataset
        batch_size: Batch size
        target_height: Target height for resizing
        target_width: Target width for resizing
        augment: Whether to apply data augmentation to training set
    
    Returns:
        train_generator, val_generator
    """
    # Paths
    train_images_dir = os.path.join(data_dir, 'Radiographs', 'training_images')
    test_images_dir = os.path.join(data_dir, 'Radiographs', 'testing_images')
    train_annotations_path = os.path.join(data_dir, 'bboxes', 'trainBoundryBoxes.csv')
    test_annotations_path = os.path.join(data_dir, 'bboxes', 'testBoundryBoxes.csv')
    
    # Load annotations
    print("Loading training annotations...")
    train_annotations = load_bbox_annotations(train_annotations_path)
    print(f"Loaded annotations for {len(train_annotations)} training images")
    
    print("Loading test annotations...")
    test_annotations = load_bbox_annotations(test_annotations_path)
    print(f"Loaded annotations for {len(test_annotations)} test images")
    
    # Filter out images without annotations
    train_annotations = {k: v for k, v in train_annotations.items() 
                         if os.path.exists(os.path.join(train_images_dir, k))}
    test_annotations = {k: v for k, v in test_annotations.items() 
                        if os.path.exists(os.path.join(test_images_dir, k))}
    
    # Create generators
    print("Creating data generators...")
    train_generator = SegmentationDataGenerator(
        train_images_dir,
        train_annotations,
        batch_size=batch_size,
        target_height=target_height,
        target_width=target_width,
        augment=augment,
        shuffle=True
    )
    
    val_generator = SegmentationDataGenerator(
        test_images_dir,
        test_annotations,
        batch_size=batch_size,
        target_height=target_height,
        target_width=target_width,
        augment=False,
        shuffle=False
    )
    
    return train_generator, val_generator


if __name__ == "__main__":
    # Test the data generator
    data_dir = os.path.join(os.path.dirname(__file__), 'TUFTS')
    
    train_gen, val_gen = create_data_generators(
        data_dir, 
        batch_size=4, 
        target_height=512, 
        target_width=512,
        augment=True
    )
    
    print(f"\nTraining batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    
    # Get a sample batch
    images, masks = train_gen[0]
    print(f"\nSample batch shape:")
    print(f"  Images: {images.shape}")
    print(f"  Masks: {masks.shape}")
    
    # Check value ranges
    print(f"\nImage value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
