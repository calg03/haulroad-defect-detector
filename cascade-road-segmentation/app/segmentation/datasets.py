#!/usr/bin/env python
"""
Dataset classes for road defect segmentation training.
Handles multiple datasets: PotholeMix, RTK, R2S100K, and Automine.
EXACT COPY from the original working script.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Import the color mapping functions
from config import CLASSES

# Define color mappings from original script
SHREC_COLORS = {
    "crack_red": (255, 0, 0),   # Maps to class 'crack' (2)
    "crack_green": (0, 255, 0), # Maps to class 'crack' (2)
    "pothole": (0, 0, 255)      # Maps to class 'pothole' (1)
}

R2S_COLORS = {
    "water_puddle": (140, 160, 222),  # Maps to class 'puddle' (3)
    "distressed_patch": (119, 61, 128), # Maps to class 'distressed_patch' (4)
    "mud": (112, 84, 62)              # Maps to class 'mud' (5)
}

AUTOMINE_MAPPING = {
    0: 0,  # background -> background
    1: 4,  # defect -> distressed_patch
    2: 1,  # pothole -> pothole
    3: 3,  # puddle -> puddle
    4: 0,  # road -> background
}

RTK_TO_MODEL = {
    11: 1,  # pothole → pothole
    12: 2,  # craks → crack
    10: 3,  # waterPuddle → puddle
    9: 4,   # patchs → distressed_patch (opcional)
    # todo lo demás será 0 (background)
}

def remap_rtk_mask(mask):
    """Remapea índices de máscara RTK a los índices esperados por el modelo"""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for orig_idx, model_idx in RTK_TO_MODEL.items():
        new_mask[mask == orig_idx] = model_idx
    return new_mask

def rgb_to_class_indices(mask_rgb):
    """Convert RGB mask to class indices based on color mappings"""
    height, width = mask_rgb.shape[:2]
    mask_indices = np.zeros((height, width), dtype=np.uint8)
    
    # Process SHREC colors
    for label, color in SHREC_COLORS.items():
        r, g, b = color
        matching_pixels = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        if label in ["crack_red", "crack_green"]:
            mask_indices[matching_pixels] = 2  # crack
        elif label == "pothole":
            mask_indices[matching_pixels] = 1  # pothole
    
    # Process R2S colors
    for label, color in R2S_COLORS.items():
        r, g, b = color
        matching_pixels = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        if label == "water_puddle":
            mask_indices[matching_pixels] = 3  # puddle
        elif label == "distressed_patch":
            mask_indices[matching_pixels] = 4  # distressed_patch
        elif label == "mud":
            mask_indices[matching_pixels] = 5  # mud
    
    return mask_indices

def gray_to_class_indices(mask_gray):
    """Convert grayscale mask to class indices based on pixel value mappings"""
    mask_indices = np.zeros_like(mask_gray, dtype=np.uint8)
    for original_value, new_class in AUTOMINE_MAPPING.items():
        mask_indices[mask_gray == original_value] = new_class
    # Any unlisted class is mapped to background (0)
    return mask_indices


class PotholeMixDataset(Dataset):
    """Dataset for pothole_mix data with RGB masks - EXACT COPY from original script"""
    def __init__(self, root_dir, transform=None, preprocessing=None, mode='training', debug_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.mode = mode
        self.debug_limit = debug_limit
        
        self.samples = []
        
        # ADDED: Check if root directory exists
        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist!")
            return
        
        # Different dataset structures in pothole_mix
        subdirs = ['cnr-road-dataset', 'cracks-and-potholes-in-road', 'crack500', 'pothole600', 'gaps384', 'edmcrack600']
        for subdir in subdirs:
            image_dir = os.path.join(root_dir, subdir, 'images')
            mask_dir = os.path.join(root_dir, subdir, 'masks')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    # Find corresponding mask with potentially different extension
                    mask_file = os.path.splitext(img_file)[0] + '.png'  # Most masks are PNG
                    mask_path = os.path.join(mask_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
                        
                        # ADDED: Limit samples in debug mode
                        if self.debug_limit and len(self.samples) >= self.debug_limit:
                            print(f"  [DEBUG] Limited {subdir} to {len(self.samples)} samples")
                            break
            
            if self.debug_limit and len(self.samples) >= self.debug_limit:
                break

    def __getitem__(self, idx):
        # FIXED: Better error handling for out-of-bounds access
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
            
        img_path, mask_path = self.samples[idx]
        
        # FIXED: Restore original OpenCV loading for consistent color handling
        try:
            # Load image and mask using OpenCV like in original script
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            if image is None or mask is None:
                print(f"[ERROR] Failed to load image or mask: {img_path}, {mask_path}")
                # Create dummy data with correct shape
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                mask = np.zeros((256, 256, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"[ERROR] Failed to load PotholeMix sample {idx}: {e}")
            # Create dummy data with correct shape
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # FIXED: Use the EXACT same re-tagging logic from the original notebook
        # This matches the approach in PaC_unetpp_scse_effb5_320_320.ipynb
        mask = mask // 255  # Normalize to 0 or 1
        target = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        
        # Re-etiquetado: BW-0(000)/C-1(010)/P-2(100) 
        # But we need to map to our global classes: 0:background, 1:pothole, 2:crack
        mask_indices = np.zeros_like(target, dtype=np.uint8)
        mask_indices[target == 1] = 0  # background -> background (0)
        mask_indices[target == 2] = 2  # crack (010) -> crack (2) 
        mask_indices[target == 3] = 0  # background -> background (0)
        mask_indices[target == 4] = 1  # pothole (100) -> pothole (1)
        mask_indices[target == 5] = 0  # background -> background (0) 
        mask_indices[target == 6] = 0  # background -> background (0)
        mask_indices[target == 7] = 0  # background -> background (0)
        
        # Debug: Print mask statistics occasionally to verify correct mapping
        if idx % 1000 == 0:
            unique_target, counts_target = np.unique(target, return_counts=True)
            unique_final, counts_final = np.unique(mask_indices, return_counts=True)
            print(f"[DEBUG] Sample {idx}:")
            print(f"  Raw target values: {unique_target} with counts {counts_target}")
            print(f"  Final classes: {unique_final} with counts {counts_final}")
            
            # Check specifically for pothole class
            pothole_pixels = np.sum(mask_indices == 1)
            crack_pixels = np.sum(mask_indices == 2)
            if pothole_pixels > 0 or crack_pixels > 0:
                print(f"  -> Found {pothole_pixels} pothole pixels, {crack_pixels} crack pixels")
        
        # Ensure mask and image have the same dimensions
        if image.shape[:2] != mask_indices.shape[:2]:
            mask_indices = cv2.resize(mask_indices, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations to processed indices
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask_indices)
            image = augmented['image']
            mask_indices = augmented['mask']
        
        # Apply preprocessing
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask_indices)
            image = sample['image']
            mask_indices = sample['mask']
        
        # FIXED: Ensure mask is always Long type
        if torch.is_tensor(mask_indices):
            mask_indices = mask_indices.long()
        else:
            mask_indices = torch.from_numpy(mask_indices).long()
        
        return image, mask_indices

    def __len__(self):
        return len(self.samples)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index."""
        return self.samples[idx][1]


class RTKDataset(Dataset):
    """Dataset for RTK data - EXACT COPY from original script"""
    def __init__(self, root_dir, transform=None, preprocessing=None, debug_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.debug_limit = debug_limit
        
        self.images_dir = os.path.join(root_dir, 'images') if os.path.exists(os.path.join(root_dir, 'images')) else root_dir
        self.masks_dir = os.path.join(root_dir, 'masks') if os.path.exists(os.path.join(root_dir, 'masks')) else root_dir
        
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png', '.jpeg')) and not f.endswith('_mask.png')]
        self.mask_files = {}
        
        # Map image files to their corresponding mask files
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            # Try different mask naming conventions
            possible_masks = [
                f"{base_name}_mask.png",
                f"{base_name}.png",
                f"{base_name}_mask.jpg",
                f"{base_name}.jpg"
            ]
            
            for mask_file in possible_masks:
                if os.path.exists(os.path.join(self.masks_dir, mask_file)):
                    self.mask_files[img_file] = mask_file
                    break
        
        # Keep only images that have masks
        self.image_files = [img for img in self.image_files if img in self.mask_files]
        
        # ADDED: Limit samples in debug mode
        if self.debug_limit:
            self.image_files = self.image_files[:self.debug_limit]

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.mask_files[img_file]
        
        # FIXED: Restore original OpenCV loading for consistency
        try:
            image = cv2.imread(os.path.join(self.images_dir, img_file))
            mask = cv2.imread(os.path.join(self.masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"[ERROR] Failed to load RTK sample {idx}: {img_file}, {mask_file}")
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                mask = np.zeros((256, 256), dtype=np.uint8)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply RTK-specific remapping
            mask = remap_rtk_mask(mask)
            
        except Exception as e:
            print(f"[ERROR] Failed to load RTK sample {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Ensure mask and image have the same dimensions
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        # FIXED: Ensure mask is always Long type
        if torch.is_tensor(mask):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask

    def __len__(self):
        return len(self.image_files)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index."""
        img_file = self.image_files[idx]
        return os.path.join(self.masks_dir, self.mask_files[img_file])


class R2S100KDataset(Dataset):
    """Dataset for R2S100K data - EXACT COPY from original script"""
    def __init__(self, images_dir, masks_dir=None, transform=None, preprocessing=None, debug_limit=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.debug_limit = debug_limit
        
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # If masks directory is provided, find matching pairs
        if masks_dir:
            self.mask_files = {}
            for img_file in self.image_files:
                base_name = os.path.splitext(img_file)[0]
                mask_file = f"{base_name}.png"  # R2S masks are typically PNG
                if os.path.exists(os.path.join(masks_dir, mask_file)):
                    self.mask_files[img_file] = mask_file
            
            # Keep only images that have masks
            self.image_files = [img for img in self.image_files if img in self.mask_files]
        
        # ADDED: Limit samples in debug mode
        if self.debug_limit:
            self.image_files = self.image_files[:self.debug_limit]

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # FIXED: Restore original OpenCV loading for proper color mapping
        try:
            image = cv2.imread(os.path.join(self.images_dir, img_file))
            if image is None:
                print(f"[ERROR] Failed to load R2S image: {img_file}")
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"[ERROR] Failed to load R2S sample {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Load mask if available
        if hasattr(self, 'mask_files') and img_file in self.mask_files:
            try:
                mask = cv2.imread(os.path.join(self.masks_dir, self.mask_files[img_file]))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = rgb_to_class_indices(mask)
            except Exception as e:
                print(f"[ERROR] Failed to load R2S mask for {img_file}: {e}")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # Create dummy mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        # FIXED: Ensure mask is always Long type
        if torch.is_tensor(mask):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def __len__(self):
        return len(self.image_files)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index, or None if not available."""
        if hasattr(self, 'mask_files'):
            img_file = self.image_files[idx]
            return os.path.join(self.masks_dir, self.mask_files[img_file]) if img_file in self.mask_files else None
        else:
            return None


class AutomineDataset(Dataset):
    """Dataset for Automine data (train_train) - EXACT COPY from original script"""
    def __init__(self, root_dir, transform=None, preprocessing=None, debug_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.debug_limit = debug_limit
        
        # Get all image files (not containing _mask.png)
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg')) and '_mask' not in f]
        self.mask_files = {}
        
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_mask.png"
            if os.path.exists(os.path.join(root_dir, mask_file)):
                self.mask_files[img_file] = mask_file
        
        # Keep only images with masks
        self.image_files = [img for img in self.image_files if img in self.mask_files]
        
        # ADDED: Limit samples in debug mode
        if self.debug_limit:
            self.image_files = self.image_files[:self.debug_limit]

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.mask_files[img_file]
        
        # FIXED: Restore original OpenCV loading for consistency
        try:
            image = cv2.imread(os.path.join(self.root_dir, img_file))
            mask = cv2.imread(os.path.join(self.root_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"[ERROR] Failed to load Automine sample {idx}: {img_file}, {mask_file}")
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                mask = np.zeros((256, 256), dtype=np.uint8)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = gray_to_class_indices(mask)
            
        except Exception as e:
            print(f"[ERROR] Failed to load Automine sample {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Ensure mask and image have the same dimensions
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        # FIXED: Ensure mask is always Long type
        if torch.is_tensor(mask):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def __len__(self):
        return len(self.image_files)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index."""
        img_file = self.image_files[idx]
        return os.path.join(self.root_dir, self.mask_files[img_file])


class PotholeMixDataset(Dataset):
    """Dataset for pothole_mix data with RGB masks"""
    
    def __init__(self, root_dir, transform=None, preprocessing=None, mode='training', debug_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.mode = mode
        self.debug_limit = debug_limit
        
        self.samples = []
        
        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist!")
            return
        
        # Different dataset structures in pothole_mix
        subdirs = ['cnr-road-dataset', 'cracks-and-potholes-in-road', 'crack500', 
                  'pothole600', 'gaps384', 'edmcrack600']
        
        for subdir in subdirs:
            image_dir = os.path.join(root_dir, subdir, 'images')
            mask_dir = os.path.join(root_dir, subdir, 'masks')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                image_files = [f for f in os.listdir(image_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    mask_file = os.path.splitext(img_file)[0] + '.png'
                    mask_path = os.path.join(mask_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
                        
                        if self.debug_limit and len(self.samples) >= self.debug_limit:
                            print(f"  [DEBUG] Limited {subdir} to {len(self.samples)} samples")
                            break
            
            if self.debug_limit and len(self.samples) >= self.debug_limit:
                break

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
            
        img_path, mask_path = self.samples[idx]
        
        try:
            # Load image and mask using OpenCV
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            if image is None or mask is None:
                print(f"[WARN] Corrupt pair -> {img_path} / {mask_path}")
                return self.__getitem__((idx + 1) % len(self))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"[ERROR] Failed to load PotholeMix sample {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Validate image and mask dimensions
        if len(image.shape) != 3 or len(mask.shape) != 3:
            print(f"[ERROR] Invalid image/mask dimensions at {idx}: image {image.shape}, mask {mask.shape}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Ensure image and mask have the same spatial dimensions before processing
        if image.shape[:2] != mask.shape[:2]:
            # Check if dimensions are just transposed
            if image.shape[:2] == mask.shape[:2][::-1]:
                print(f"[INFO] Transposed dimensions at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - transposing mask")
                # Transpose the mask to match image dimensions
                mask = mask.transpose(1, 0, 2)
            else:
                print(f"[WARN] Size mismatch at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - resizing mask")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Process mask - exact same logic from original
        mask = mask // 255  # Normalize to 0 or 1
        target = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        
        # Re-etiquetado: BW-0(000)/C-1(010)/P-2(100) 
        mask_indices = np.zeros_like(target, dtype=np.uint8)
        mask_indices[target == 1] = 0  # background -> background (0)
        mask_indices[target == 2] = 2  # crack (010) -> crack (2) 
        mask_indices[target == 3] = 0  # background -> background (0)
        mask_indices[target == 4] = 1  # pothole (100) -> pothole (1)
        mask_indices[target == 5] = 0  # background -> background (0) 
        mask_indices[target == 6] = 0  # background -> background (0)
        mask_indices[target == 7] = 0  # background -> background (0)
        
        # Debug logging
        if idx % 1000 == 0:
            unique_target, counts_target = np.unique(target, return_counts=True)
            unique_final, counts_final = np.unique(mask_indices, return_counts=True)
            print(f"[DEBUG] Sample {idx}:")
            print(f"  Raw target values: {unique_target} with counts {counts_target}")
            print(f"  Final classes: {unique_final} with counts {counts_final}")
            
            pothole_pixels = np.sum(mask_indices == 1)
            crack_pixels = np.sum(mask_indices == 2)
            if pothole_pixels > 0 or crack_pixels > 0:
                print(f"  -> Found {pothole_pixels} pothole pixels, {crack_pixels} crack pixels")
        
        # Final validation before transforms
        if image.shape[:2] != mask_indices.shape[:2]:
            # Check if dimensions are transposed
            if image.shape[:2] == mask_indices.shape[:2][::-1]:
                print(f"[INFO] Final transposed dimensions at {idx}: image {image.shape[:2]}, mask {mask_indices.shape[:2]} - transposing mask")
                mask_indices = mask_indices.T
            else:
                print(f"[ERROR] Final size mismatch at {idx}: image {image.shape[:2]}, mask {mask_indices.shape[:2]} - resizing mask")
                mask_indices = cv2.resize(mask_indices, (image.shape[1], image.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations with error handling
        try:
            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask_indices)
                image = augmented['image']
                mask_indices = augmented['mask']
        except Exception as e:
            print(f"[ERROR] Transform failed for sample {idx}: {e}")
            print(f"  Image shape: {image.shape}, Mask shape: {mask_indices.shape}")
            # Fallback: resize both to standard size
            from config import IMG_SIZE
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            mask_indices = cv2.resize(mask_indices, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Apply preprocessing with error handling
        try:
            if self.preprocessing is not None:
                sample = self.preprocessing(image=image, mask=mask_indices)
                image = sample['image']
                mask_indices = sample['mask']
        except Exception as e:
            print(f"[ERROR] Preprocessing failed for sample {idx}: {e}")
            # Skip preprocessing if it fails
            pass
        
        # Ensure mask is always Long type
        if torch.is_tensor(mask_indices):
            mask_indices = mask_indices.long()
        else:
            mask_indices = torch.from_numpy(mask_indices).long()
        
        return image, mask_indices

    def __len__(self):
        return len(self.samples)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index."""
        return self.samples[idx][1]


class RTKDataset(Dataset):
    """Dataset for RTK data"""
    
    def __init__(self, root_dir, transform=None, preprocessing=None, debug_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.debug_limit = debug_limit
        
        self.images_dir = (os.path.join(root_dir, 'images') 
                          if os.path.exists(os.path.join(root_dir, 'images')) 
                          else root_dir)
        self.masks_dir = (os.path.join(root_dir, 'masks') 
                         if os.path.exists(os.path.join(root_dir, 'masks')) 
                         else root_dir)
        
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg')) and not f.endswith('_mask.png')]
        self.mask_files = {}
        
        # Map image files to their corresponding mask files
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}.png"
            if os.path.exists(os.path.join(self.masks_dir, mask_file)):
                self.mask_files[img_file] = mask_file
        
        # Keep only images that have masks
        self.image_files = [img for img in self.image_files if img in self.mask_files]
        
        if self.debug_limit:
            self.image_files = self.image_files[:self.debug_limit]
            print(f"  [DEBUG] Limited RTK to {len(self.image_files)} samples")

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = self.mask_files[img_file]
        
        try:
            image = cv2.imread(os.path.join(self.images_dir, img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(self.masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
            mask = remap_rtk_mask(mask)
        except Exception as e:
            print(f"[ERROR] Failed to load RTK sample {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Ensure mask and image have the same dimensions
        if image.shape[:2] != mask.shape[:2]:
            # Check if dimensions are just transposed
            if image.shape[:2] == mask.shape[:2][::-1]:
                print(f"[INFO] RTK transposed dimensions at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - transposing mask")
                mask = mask.T
            else:
                print(f"[WARN] RTK size mismatch at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - resizing mask")
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']
        
        # Ensure mask is always Long type
        if torch.is_tensor(mask):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask

    def __len__(self):
        return len(self.image_files)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index."""
        img_file = self.image_files[idx]
        return os.path.join(self.masks_dir, self.mask_files[img_file])


class R2S100KDataset(Dataset):
    """Dataset for R2S100K data"""
    
    def __init__(self, images_dir, masks_dir=None, transform=None, preprocessing=None, debug_limit=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.debug_limit = debug_limit
        
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # If masks directory is provided, find matching pairs
        if masks_dir:
            self.mask_files = {}
            for img_file in self.image_files:
                base_name = os.path.splitext(img_file)[0]
                mask_candidates = [
                    f"{base_name}.png",
                    f"{base_name}_mask.png",
                    f"{base_name}_segmentation.png",
                    f"{base_name}.jpg___fuse.png",
                ]
                
                for mask_name in mask_candidates:
                    if os.path.exists(os.path.join(masks_dir, mask_name)):
                        self.mask_files[img_file] = mask_name
                        break
            
            # Keep only images with masks
            self.image_files = [img for img in self.image_files if img in self.mask_files]
        
        if self.debug_limit:
            self.image_files = self.image_files[:self.debug_limit]
            print(f"  [DEBUG] Limited R2S100K to {len(self.image_files)} samples")

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        try:
            image = cv2.imread(os.path.join(self.images_dir, img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] Failed to load R2S100K image {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Load mask if available
        if hasattr(self, 'mask_files') and img_file in self.mask_files:
            mask_file = self.mask_files[img_file]
            try:
                mask = cv2.imread(os.path.join(self.masks_dir, mask_file))
                if mask is not None:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    mask = rgb_to_class_indices(mask)
                    
                    if image.shape[:2] != mask.shape[:2]:
                        # Check if dimensions are just transposed
                        if image.shape[:2] == mask.shape[:2][::-1]:
                            print(f"[INFO] R2S100K transposed dimensions at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - transposing mask")
                            mask = mask.T
                        else:
                            print(f"[WARN] R2S100K size mismatch at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - resizing mask")
                            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            except Exception as e:
                print(f"[ERROR] Failed to load R2S100K mask {idx}: {e}")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask'].long()
        
        # Ensure mask is always Long type
        if torch.is_tensor(mask):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def __len__(self):
        return len(self.image_files)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index, or None if not available."""
        if hasattr(self, 'mask_files'):
            img_file = self.image_files[idx]
            return os.path.join(self.masks_dir, self.mask_files[img_file])
        else:
            return None


class AutomineDataset(Dataset):
    """Dataset for Automine data (train_train)"""
    
    def __init__(self, root_dir, transform=None, preprocessing=None, debug_limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.debug_limit = debug_limit
        
        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist!")
            self.image_files = []
            return
        
        # Look for images and masks
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg')) and '_mask' not in f]
        
        if self.debug_limit:
            self.image_files = self.image_files[:self.debug_limit]
            print(f"  [DEBUG] Limited Automine to {len(self.image_files)} samples")

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        try:
            image = cv2.imread(os.path.join(self.root_dir, img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Look for corresponding mask
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_mask.png"
            mask_path = os.path.join(self.root_dir, mask_file)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = gray_to_class_indices(mask)
                
                # Ensure mask and image have the same dimensions
                if image.shape[:2] != mask.shape[:2]:
                    # Check if dimensions are just transposed
                    if image.shape[:2] == mask.shape[:2][::-1]:
                        print(f"[INFO] Automine transposed dimensions at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - transposing mask")
                        mask = mask.T
                    else:
                        print(f"[WARN] Automine size mismatch at {idx}: image {image.shape[:2]}, mask {mask.shape[:2]} - resizing mask")
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
        except Exception as e:
            print(f"[ERROR] Failed to load Automine sample {idx}: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']
        
        # Ensure mask is always Long type
        if torch.is_tensor(mask):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask

    def __len__(self):
        return len(self.image_files)
    
    def get_mask_path(self, idx):
        """Return the mask file path for a given index."""
        img_file = self.image_files[idx]
        base_name = os.path.splitext(img_file)[0]
        return os.path.join(self.root_dir, f"{base_name}_mask.png")
