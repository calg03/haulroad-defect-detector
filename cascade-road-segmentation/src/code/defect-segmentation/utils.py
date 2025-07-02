#!/usr/bin/env python
"""
Utility functions for road defect segmentation training.
"""

import os
import numpy as np
import cv2


def find_dataset_path(primary_path, fallback_key=None):
    """Find existing dataset path with fallbacks - matches original script exactly"""
    from config import FALLBACK_PATHS
    
    if os.path.exists(primary_path):
        return primary_path
    
    if fallback_key and fallback_key in FALLBACK_PATHS:
        for fallback in FALLBACK_PATHS[fallback_key]:
            if os.path.exists(fallback):
                print(f"Using fallback path: {fallback}")
                return fallback
    
    return primary_path  # Return original even if not found


def remap_rtk_mask(mask):
    """Remapea índices de máscara RTK a los índices esperados por el modelo"""
    from config import RTK_TO_MODEL
    
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for orig_idx, model_idx in RTK_TO_MODEL.items():
        new_mask[mask == orig_idx] = model_idx
    return new_mask


def rgb_to_class_indices(mask_rgb):
    """Convert RGB mask to class indices based on color mappings"""
    from config import SHREC_COLORS, R2S_COLORS
    
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
    from config import AUTOMINE_MAPPING
    
    mask_indices = np.zeros_like(mask_gray, dtype=np.uint8)
    for original_value, new_class in AUTOMINE_MAPPING.items():
        mask_indices[mask_gray == original_value] = new_class
    # Any unlisted class is mapped to background (0)
    return mask_indices


def verify_class_distribution(dataset, name, num_classes=6):
    """Verify class distribution in a dataset"""
    from config import CLASSES
    
    class_hist = np.zeros(num_classes, dtype=int)
    for idx in range(min(len(dataset), 300)):  # limit to 300 for speed
        try:
            _, mask = dataset[idx]
            np_mask = mask.numpy() if hasattr(mask, 'numpy') else mask
            unique, counts = np.unique(np_mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                if cls < num_classes:
                    class_hist[cls] += cnt
        except Exception as e:
            print(f"[ERROR] Dataset {name}, sample {idx}: {e}")
    
    print(f"\n[{name}] Class Distribution:")
    for cls_id, count in enumerate(class_hist):
        print(f"  {CLASSES[cls_id]:>16s}: {count:,} pixels")
