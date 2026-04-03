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
        """Apply enhanced random augmentation to image and mask."""
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
            angle = np.random.uniform(-20, 20)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h),
                                  flags=cv2.INTER_NEAREST)

        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Random zoom (scale augmentation)
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(0.8, 1.2)
            h, w = image.shape[:2]

            # Calculate new dimensions
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

            # Resize
            image_zoomed = cv2.resize(image, (new_w, new_h))
            mask_zoomed = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Crop or pad to original size
            if zoom_factor > 1.0:
                # Crop center
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                image = image_zoomed[start_y:start_y+h, start_x:start_x+w]
                mask = mask_zoomed[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad with zeros
                if mask.ndim == 2:
                    mask_padded = np.zeros((h, w), dtype=np.float32)
                else:
                    mask_padded = np.zeros((h, w, mask.shape[2]), dtype=np.float32)
                
                image_padded = np.zeros((h, w, 3), dtype=np.uint8)
                start_y = (h - new_h) // 2
                start_x = (w - new_w) // 2
                image_padded[start_y:start_y+new_h, start_x:start_x+new_w] = image_zoomed
                mask_padded[start_y:start_y+new_h, start_x:start_x+new_w] = mask_zoomed
                image = image_padded
                mask = mask_padded

        # Random translation (shift augmentation)
        if np.random.random() > 0.5:
            h, w = image.shape[:2]
            shift_x = np.random.randint(-30, 30)
            shift_y = np.random.randint(-30, 30)

            # Create translation matrix
            translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

            # Apply translation
            image = cv2.warpAffine(image, translation_matrix, (w, h))
            mask = cv2.warpAffine(mask, translation_matrix, (w, h),
                                  flags=cv2.INTER_NEAREST)

        # Random contrast adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean = np.mean(gray)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, np.random.uniform(5, 15), image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)

        # Random Gaussian blur
        if np.random.random() > 0.5:
            kernel_size = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Elastic transformation (deformable augmentation)
        if np.random.random() > 0.6:
            image, mask = self._elastic_transform(image, mask, 
                                                   alpha=np.random.uniform(200, 400),
                                                   sigma=np.random.uniform(20, 40))

        # Random color jitter (RGB channel adjustments)
        if np.random.random() > 0.5:
            # Randomly adjust individual channels
            for channel in range(3):
                if np.random.random() > 0.5:
                    factor = np.random.uniform(0.8, 1.2)
                    image[:, :, channel] = np.clip(
                        image[:, :, channel].astype(np.float32) * factor, 
                        0, 255
                    ).astype(np.uint8)

        # Random shearing
        if np.random.random() > 0.5:
            h, w = image.shape[:2]
            shear_x = np.random.uniform(-0.2, 0.2)
            shear_y = np.random.uniform(-0.2, 0.2)
            
            shear_matrix = np.float32([
                [1, shear_x, 0],
                [shear_y, 1, 0]
            ])
            image = cv2.warpAffine(image, shear_matrix, (w, h))
            mask = cv2.warpAffine(mask, shear_matrix, (w, h),
                                  flags=cv2.INTER_NEAREST)

        return image, mask

    def _elastic_transform(self, image, mask, alpha, sigma):
        """
        Apply elastic transformation for realistic deformations.
        
        Args:
            image: Input image
            mask: Input mask
            alpha: Intensity of transformation
            sigma: Smoothness of deformation
        """
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.rand(h, w).astype(np.float32) * 2 - 1
        dy = np.random.rand(h, w).astype(np.float32) * 2 - 1
        
        # Apply Gaussian filtering for smooth deformations
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacements
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap image and mask
        image_transformed = cv2.remap(image, map_x, map_y, 
                                      interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
        mask_transformed = cv2.remap(mask, map_x, map_y,
                                     interpolation=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_REFLECT_101)
        
        return image_transformed, mask_transformed
    
    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.image_ids)


def compute_class_weights(data_generator, target_bg_weight=1.0, target_fg_weight=10.0):
    """
    Compute class weights based on data distribution.
    
    Args:
        data_generator: SegmentationDataGenerator instance
        target_bg_weight: Base weight for background class
        target_fg_weight: Base weight for foreground class
        
    Returns:
        Dictionary with class weights
    """
    print("Computing class weights from training data...")
    
    total_pixels = 0
    total_fg_pixels = 0
    num_batches = min(len(data_generator), 50)  # Sample first 50 batches
    
    for i in range(num_batches):
        _, masks = data_generator[i]
        total_pixels += np.prod(masks.shape)
        total_fg_pixels += np.sum(masks > 0.5)
    
    fg_ratio = total_fg_pixels / total_pixels
    bg_ratio = 1.0 - fg_ratio
    
    # Compute balanced weights
    if fg_ratio > 0:
        fg_weight = min(target_fg_weight, 1.0 / (fg_ratio + 1e-7))
        bg_weight = target_bg_weight
    else:
        fg_weight = target_fg_weight
        bg_weight = target_bg_weight
    
    print(f"  Foreground pixel ratio: {fg_ratio:.4f}")
    print(f"  Background pixel ratio: {bg_ratio:.4f}")
    print(f"  Using foreground weight: {fg_weight:.2f}")
    print(f"  Using background weight: {bg_weight:.2f}")
    
    return {0: bg_weight, 1: fg_weight}


def create_data_generators(data_dir, batch_size=8, target_height=512, target_width=512, 
                           augment=True, use_patch_based=False, patch_size=256):
    """
    Create training and validation data generators.

    Args:
        data_dir: Base directory containing the dataset
        batch_size: Batch size
        target_height: Target height for resizing
        target_width: Target width for resizing
        augment: Whether to apply data augmentation to training set
        use_patch_based: Whether to use patch-based training
        patch_size: Size of patches for patch-based training

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
    
    if use_patch_based:
        print(f"Using patch-based training with patch size: {patch_size}x{patch_size}")
        train_generator = PatchBasedDataGenerator(
            train_images_dir,
            train_annotations,
            batch_size=batch_size,
            target_height=target_height,
            target_width=target_width,
            patch_size=patch_size,
            augment=augment,
            shuffle=True
        )
    else:
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


class PatchBasedDataGenerator(Sequence):
    """
    Patch-based data generator for training on smaller regions.
    Helps with memory efficiency and better local feature learning.
    """

    def __init__(self,
                 image_dir,
                 annotations,
                 batch_size=8,
                 target_height=512,
                 target_width=512,
                 patch_size=256,
                 augment=False,
                 shuffle=True):
        """
        Initialize the patch-based data generator.

        Args:
            image_dir: Directory containing images
            annotations: Dictionary of image annotations
            batch_size: Batch size
            target_height: Target height for resizing
            target_width: Target width for resizing
            patch_size: Size of patches to extract
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.patch_size = patch_size
        self.augment = augment
        self.shuffle = shuffle

        # Get list of image IDs
        self.image_ids = list(annotations.keys())

        if self.shuffle:
            np.random.shuffle(self.image_ids)

        # Calculate patches per image
        self.patches_per_image = max(1, (target_height // patch_size) * (target_width // patch_size))

    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.image_ids) * self.patches_per_image / self.batch_size))

    def __getitem__(self, idx):
        """Return a batch of patches and masks."""
        images = []
        masks = []

        for _ in range(self.batch_size):
            # Select random image
            img_idx = np.random.randint(0, len(self.image_ids))
            image_id = self.image_ids[img_idx]

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

            # Extract random patch
            patch_img, patch_mask = self._extract_random_patch(image_resized, mask_resized)

            # Apply augmentation if enabled
            if self.augment:
                # Temporarily modify to work with single-image augmentation
                aug_img, aug_mask = self._augment(patch_img, patch_mask)
                patch_img = aug_img
                patch_mask = aug_mask

            # Normalize image
            patch_img = patch_img.astype(np.float32) / 255.0

            images.append(patch_img)
            masks.append(patch_mask)

        return np.array(images), np.array(masks)

    def _extract_random_patch(self, image, mask):
        """Extract a random patch from image and mask."""
        h, w = image.shape[:2]
        
        # Random top-left corner
        max_y = h - self.patch_size
        max_x = w - self.patch_size
        
        if max_y <= 0 or max_x <= 0:
            # If image is smaller than patch, return full image
            return image, mask
        
        start_y = np.random.randint(0, max_y)
        start_x = np.random.randint(0, max_x)
        
        patch_img = image[start_y:start_y+self.patch_size, 
                         start_x:start_x+self.patch_size]
        patch_mask = mask[start_y:start_y+self.patch_size, 
                         start_x:start_x+self.patch_size]
        
        return patch_img, patch_mask

    def _augment(self, image, mask):
        """Apply random augmentation (same as SegmentationDataGenerator)."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random vertical flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h),
                                  flags=cv2.INTER_NEAREST)

        # Random brightness
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return image, mask

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.image_ids)


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
