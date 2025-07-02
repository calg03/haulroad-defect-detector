#!/usr/bin/env python3
"""
Modular Testing Script for Road Defect Segmentation

Unified testing script that supports all architectures from the modular training suite:
- UNet, UNet++, DeepLabV3, DeepLabV3+, FPN, PSPNet, TransUNet, Swin-UNet
- Automatic architecture detection from model files
- Comprehensive evaluation metrics
- Visual output generation (overlays, masks, probability maps)
- Batch processing with progress tracking
- Integration with the same architecture system used in training

Usage:
    python test_modular_segmentation.py --config test_config.py
    
    All settings are loaded from the config file. Optional flags:
    --verbose              Enable verbose output
    --direct_evaluation    Run evaluation only (no output files)
    --list_architectures   Show available architectures
"""

import sys
import os
import json
import time
import argparse
import glob
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our modular architecture system
try:
    from architectures import (
        create_model, get_available_architectures, get_architecture_info,
        ARCHITECTURE_CONFIGS, list_architectures,
        get_supported_encoders, get_encoder_variants, 
        create_model_with_encoder, detect_encoder_from_checkpoint
    )
    from augmentation import get_val_transform  # Import exact validation transform used during training
    print("✅ Modular architecture system and validation transforms loaded")
except ImportError as e:
    print(f"❌ Error importing architectures module: {e}")
    print("Make sure architectures.py is in the same directory")
    sys.exit(1)

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    SMP_AVAILABLE = True
except ImportError:
    print("⚠️ segmentation_models_pytorch not available")
    SMP_AVAILABLE = False

# Configuration
DEFAULT_CONFIG = {
    'input_dir': '../data/test/images',
    'output_dir': './results/test_output',
    'gt_dir': None,  # Set for evaluation
    'batch_size': 8,
    'num_workers': 4,
    'img_size': 512,
    'device': 'auto',  # 'auto', 'cuda', 'cpu', or specific like 'cuda:0'
    'save_overlays': True,
    'save_masks': True,
    'save_probability_maps': False,
    'evaluate': False,
    'mixed_precision': True,
    'verbose': True
}

# Class definitions - matches training configuration
CLASSES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
NUM_CLASSES = len(CLASSES)

# Color mapping for visualization - matches training
COLOR_MAP = {
    0: (0, 0, 0),         # background: black
    1: (0, 0, 255),       # pothole: blue
    2: (0, 255, 0),       # crack: green
    3: (140, 160, 222),   # puddle: light blue
    4: (119, 61, 128),    # distressed_patch: purple
    5: (112, 84, 62)      # mud: brown
}

# Training color mappings for ground truth processing
SHREC_COLORS = {
    "crack_red": (255, 0, 0),   # Maps to class 'crack' (2)
    "crack_green": (0, 255, 0), # Maps to class 'crack' (2)
    "pothole": (0, 0, 255)      # Maps to class 'pothole' (1)
}

R2S_COLORS = {
    "water_puddle": (140, 160, 222),     # Maps to class 'puddle' (3)
    "distressed_patch": (119, 61, 128),  # Maps to class 'distressed_patch' (4)
    "mud": (112, 84, 62)                 # Maps to class 'mud' (5)
}

# AutoMine mapping - corrected based on actual mask analysis
# From mask analysis: 0=background, 4=defects (road damage)
# Map 4 to distressed_patch (class 4) since it represents road defects
AUTOMINE_MAPPING = {
    0: 0,  # background -> background
    1: 4,  # defect -> distressed_patch (matches training config.py)
    2: 1,  # pothole -> pothole
    3: 3,  # puddle -> puddle
    4: 0,  # road -> background (road surface is not a defect!)
    255: 0 # Unknown/invalid -> background
}

# For debugging - inverse mapping to understand what we're comparing
AUTOMINE_INVERSE_MAPPING = {v: k for k, v in AUTOMINE_MAPPING.items() if k != 255}
# Note: original values 0 and 4 both map to class 0 (background), value 1 maps to class 4 (distressed_patch)
AUTOMINE_INVERSE_MAPPING[4] = [1, 4]  # distressed_patch can come from original 1 or 4


class TestDataset(Dataset):
    """Dataset class for testing/inference"""
    
    def __init__(self, image_paths: List[str], architecture: str, img_size: int = 512):
        self.image_paths = image_paths
        self.architecture = architecture
        self.img_size = img_size
        
        # Get architecture-specific preprocessing
        self.transform = self._get_transform()
        self.preprocessing = self._get_preprocessing()
    
    def _get_transform(self) -> A.Compose:
        """Get architecture-appropriate transforms - USE EXACT SAME AS VALIDATION"""
        # CRITICAL FIX: Use the exact same validation transform as training
        # This ensures square images (512x512) instead of maintaining aspect ratio
        return get_val_transform()  # This matches training validation exactly
    
    def _get_preprocessing(self) -> Optional[A.Compose]:
        """Get architecture-specific preprocessing"""
        arch_info = get_architecture_info(self.architecture)
        
        if not arch_info or not arch_info.get('requires_smp', False):
            # For custom architectures (TransUNet, Swin-UNet), use standard normalization
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # For SMP architectures, use encoder-specific preprocessing
        encoder = arch_info.get('encoder', 'resnet50')
        encoder_weights = arch_info.get('encoder_weights', 'imagenet')
        
        preprocessing_fn = get_preprocessing_fn(encoder, encoder_weights)
        
        def _preprocess(img, **kwargs):
            img = img.astype(np.float32) / 255.0
            img = preprocessing_fn(img)
            return img.astype(np.float32)
        
        return A.Compose([
            A.Lambda(name="preproc", image=_preprocess),
            ToTensorV2()
        ])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size
        original_size = torch.tensor([image.shape[0], image.shape[1]])  # (height, width)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image)
            image = preprocessed['image']
        
        return image, image_path, original_size
    
    def __len__(self) -> int:
        return len(self.image_paths)


class ModelTester:
    """Main testing class that handles all architectures"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.results = {}
        
        # Validate AutoMine mapping consistency
        if config.get('verbose', False):
            self.validate_automine_mapping()
        
        print(f"🔧 Initialized ModelTester on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                print("💻 Using CPU")
        else:
            device = torch.device(device_config)
            print(f"🎯 Using specified device: {device}")
        
        return device
    
    def load_model(self, model_path: str, architecture: str, encoder_override: str = None, 
                   auto_detect_encoder: bool = True) -> torch.nn.Module:
        """Load model with architecture-specific configuration and optional encoder override"""
        print(f"🏗️ Loading {architecture} model from {model_path}")
        
        # Validate architecture
        if architecture not in ARCHITECTURE_CONFIGS:
            available = get_available_architectures()
            raise ValueError(f"Architecture '{architecture}' not available. Choose from: {available}")
        
        # Handle encoder override and auto-detection for architectures that support it
        final_encoder = None
        supported_encoders = get_supported_encoders(architecture)
        
        if len(supported_encoders) > 1:  # Architecture supports multiple encoders
            if encoder_override:
                if encoder_override not in supported_encoders:
                    raise ValueError(f"Encoder '{encoder_override}' not supported for {architecture}. "
                                   f"Supported encoders: {supported_encoders}")
                final_encoder = encoder_override
                print(f"🔧 Using encoder override: {encoder_override}")
                
            elif auto_detect_encoder:
                detected_encoder = detect_encoder_from_checkpoint(model_path)
                if detected_encoder and detected_encoder in supported_encoders:
                    final_encoder = detected_encoder
                    print(f"🔍 Auto-detected encoder: {detected_encoder}")
                else:
                    print(f"⚠️ Could not auto-detect encoder, using default: {ARCHITECTURE_CONFIGS[architecture]['encoder']}")
            
            # Create model with encoder override if needed
            if final_encoder:
                model = create_model_with_encoder(
                    architecture=architecture,
                    num_classes=NUM_CLASSES,
                    encoder_override=final_encoder,
                    device=self.device
                )
            else:
                model = create_model(
                    architecture=architecture,
                    num_classes=NUM_CLASSES,
                    img_size=self.config['img_size']
                )
        else:
            # Standard model creation for single-encoder architectures
            model = create_model(
                architecture=architecture,
                num_classes=NUM_CLASSES,
                img_size=self.config['img_size']
            )
        
        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Handle PyTorch 2.6+ weights_only default change
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e) or "GLOBAL numpy" in str(e):
                print("⚠️ Loading with weights_only=False for compatibility with older checkpoints")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            else:
                raise e
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"📚 Loaded checkpoint from epoch {checkpoint['epoch']}")
            if 'best_metric' in checkpoint:
                print(f"🏆 Best metric: {checkpoint['best_metric']:.4f}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        return model
    
    def _find_automine_mask(self, image_path: str, gt_dir: str) -> Optional[str]:
        """
        Find corresponding AutoMine mask for an image.
        AutoMine format: 000064_png.rf.5ea97b21a46f19d9ceb9867f255cd0ef.jpg -> 000064_png.rf.5ea97b21a46f19d9ceb9867f255cd0ef_mask.png
        """
        image_name = os.path.basename(image_path)
        
        # Handle AutoMine format specifically
        if image_name.endswith('.jpg'):
            # For AutoMine: remove .jpg and add _mask.png
            base_name = image_name[:-4]  # Remove .jpg extension
            mask_name = f"{base_name}_mask.png"
            mask_path = os.path.join(gt_dir, mask_name)
            
            if os.path.exists(mask_path):
                if self.config.get('verbose', False):
                    print(f"✅ Found AutoMine mask: {mask_name} for image: {image_name}")
                return mask_name
        
        # Fallback: try standard naming conventions
        base_name = os.path.splitext(image_name)[0]  # Remove extension
        
        # Standard mask naming conventions
        mask_candidates = [
            f"{base_name}_mask.png",
            f"{base_name}.png",
            f"{base_name}_gt.png", 
            f"{base_name}_label.png",
            f"{base_name}.tif",
            f"{base_name}.tiff"
        ]
        
        for mask_name in mask_candidates:
            mask_path = os.path.join(gt_dir, mask_name)
            if os.path.exists(mask_path):
                if self.config.get('verbose', False):
                    print(f"✅ Found fallback mask: {mask_name} for image: {image_name}")
                return mask_name
        
        # Final attempt: search all mask files for similar names
        all_mask_files = glob.glob(os.path.join(gt_dir, "*_mask.png"))
        for mask_file in all_mask_files:
            mask_name = os.path.basename(mask_file)
            mask_base = mask_name.replace("_mask.png", "")
            if mask_base == base_name:
                if self.config.get('verbose', False):
                    print(f"✅ Found glob match mask: {mask_name} for image: {image_name}")
                return mask_name
        
        return None

    def _get_image_mask_pairs(self, input_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
        """
        Get paired image and ground truth mask paths.
        Specifically handles AutoMine format where images are .jpg and masks are *_mask.png
        """
        image_paths = self._get_image_paths(input_dir)
        pairs = []
        
        if self.config.get('verbose', False):
            print(f"\n🔍 Looking for image-mask pairs:")
            print(f"   Input directory: {input_dir}")
            print(f"   Ground truth directory: {gt_dir}")
            print(f"   Found {len(image_paths)} images to process")
            
            # In verbose mode, show the first few images to help debug
            print(f"\n   Sample images found:")
            for i, path in enumerate(image_paths[:5]):
                print(f"   {i+1}. {os.path.basename(path)}")
            if len(image_paths) > 5:
                print(f"   ... and {len(image_paths)-5} more")
                
            # List some masks to help debug
            gt_masks = glob.glob(os.path.join(gt_dir, "*_mask.png"))
            print(f"\n   Sample masks found in GT dir:")
            for i, path in enumerate(gt_masks[:5]):
                print(f"   {i+1}. {os.path.basename(path)}")
            if len(gt_masks) > 5:
                print(f"   ... and {len(gt_masks)-5} more")
        
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            mask_name = self._find_automine_mask(image_path, gt_dir)
            
            if mask_name:
                mask_path = os.path.join(gt_dir, mask_name)
                pairs.append((image_path, mask_path))
                if self.config.get('verbose', False):
                    print(f"✅ Matched: {image_name} → {mask_name}")
            else:
                if self.config.get('verbose', False):
                    print(f"⚠️ No ground truth mask found for {image_name}")
        
        print(f"\n📋 Found {len(pairs)} image-mask pairs for evaluation")
        
        # In verbose mode, show details of the first few pairs
        if self.config.get('verbose', False) and pairs:
            print(f"\n   Sample pairs to be evaluated:")
            for i, (img_path, mask_path) in enumerate(pairs[:3]):
                print(f"   {i+1}. {os.path.basename(img_path)} ↔ {os.path.basename(mask_path)}")
            if len(pairs) > 3:
                print(f"   ... and {len(pairs)-3} more")
        
        return pairs

    def _get_image_paths(self, input_dir: str) -> List[str]:
        """Get all image paths from input directory, excluding mask files"""
        # For AutoMine dataset: images are .jpg, masks are .png
        # For other datasets: include various image formats but exclude masks
        image_extensions = {'.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # Check if this looks like an AutoMine directory
        all_files = glob.glob(os.path.join(input_dir, "*"))
        jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]
        mask_files = [f for f in all_files if f.endswith('_mask.png')]
        
        if jpg_files and mask_files:
            print("🔍 Detected AutoMine format - using .jpg files as images")
            # This is AutoMine format - only use .jpg files
            image_paths = jpg_files
        else:
            # Standard format - include .png but exclude masks
            image_extensions.add('.png')
            image_paths = []
            
            for ext in image_extensions:
                pattern = os.path.join(input_dir, f"*{ext}")
                found_files = glob.glob(pattern)
                pattern = os.path.join(input_dir, f"*{ext.upper()}")
                found_files.extend(glob.glob(pattern))
                
                # Filter out mask files
                for file_path in found_files:
                    filename = os.path.basename(file_path)
                    # Skip files that are masks (contain _mask in filename)
                    if '_mask' not in filename.lower():
                        image_paths.append(file_path)
        
        image_paths = list(set(image_paths))  # Remove duplicates
        image_paths.sort()
        
        if not image_paths:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"📁 Found {len(image_paths)} images for testing (excluding mask files)")
        if self.config.get('verbose', False):
            # Show first few image names for verification
            print("📋 Sample image files:")
            for i, path in enumerate(image_paths[:5]):
                print(f"   {i+1}. {os.path.basename(path)}")
            if len(image_paths) > 5:
                print(f"   ... and {len(image_paths) - 5} more")
                
            # Show what extensions were used
            extensions_found = set(os.path.splitext(path)[1].lower() for path in image_paths)
            print(f"📝 Extensions found: {extensions_found}")
        
        return image_paths
    
    def _setup_output_directories(self, output_dir: str) -> Dict[str, str]:
        """Setup output directory structure"""
        dirs = {'base': output_dir}
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config.get('save_overlays', True):
            dirs['overlays'] = os.path.join(output_dir, 'overlays')
            os.makedirs(dirs['overlays'], exist_ok=True)
        
        if self.config.get('save_masks', True):
            dirs['masks'] = os.path.join(output_dir, 'masks')
            os.makedirs(dirs['masks'], exist_ok=True)
        
        if self.config.get('save_probability_maps', False):
            dirs['probabilities'] = os.path.join(output_dir, 'probabilities')
            os.makedirs(dirs['probabilities'], exist_ok=True)
        
        return dirs
    
    def _mask_to_color(self, mask: np.ndarray, debug_info: str = "") -> np.ndarray:
        """Convert class prediction mask to RGB image"""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Debug: Show detected classes
        unique_classes = np.unique(mask)
        if self.config.get('verbose', False) and debug_info:
            try:
                print(f"🎨 {debug_info} - Detected classes: {unique_classes}")
                for class_idx in unique_classes:
                    if class_idx < NUM_CLASSES:
                        pixel_count = np.sum(mask == class_idx)
                        percentage = (pixel_count / (h * w)) * 100
                        color = COLOR_MAP.get(class_idx, (128, 128, 128))
                        print(f"   Class {class_idx} ({CLASSES[class_idx]}): {percentage:.2f}% - Color: {color}")
            except (BrokenPipeError, IOError):
                pass
        
        for class_idx, color in COLOR_MAP.items():
            color_mask[mask == class_idx] = color
        
        return color_mask
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, probs: np.ndarray = None, alpha: float = 0.7, debug_info: str = "") -> np.ndarray:
        """Create overlay of mask on image with class labels and probabilities"""
        color_mask = self._mask_to_color(mask, debug_info)
        
        # Debug: Check if overlay is being created properly
        if self.config.get('verbose', False) and debug_info:
            try:
                non_black_pixels = np.sum(np.any(color_mask != [0, 0, 0], axis=2))
                total_pixels = color_mask.shape[0] * color_mask.shape[1]
                print(f"🖼️ {debug_info} - Non-background pixels in color mask: {non_black_pixels}/{total_pixels} ({100*non_black_pixels/total_pixels:.2f}%)")
            except (BrokenPipeError, IOError):
                pass
        
        # Enhance color mask visibility
        color_mask = cv2.convertScaleAbs(color_mask, alpha=1.2, beta=0)
        
        # Create overlay with specified alpha blending
        overlay = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)
        
        # Enhance final overlay
        overlay = cv2.convertScaleAbs(overlay, alpha=1.1, beta=0)
        
        # Add class labels directly on the overlay
        overlay_with_labels = self._add_overlay_labels(overlay, mask, probs)
        
        return overlay_with_labels
    
    def _add_overlay_labels(self, overlay: np.ndarray, mask: np.ndarray, probs: np.ndarray = None) -> np.ndarray:
        """Add class labels and probabilities directly on top of the overlay"""
        # Find unique classes present in the mask (excluding background)
        unique_classes = np.unique(mask)
        detected_classes = [cls for cls in unique_classes if cls > 0 and cls < len(CLASSES)]
        
        if not detected_classes:
            # No defects detected, add message at top
            text = "No road defects detected"
            text_x, text_y = 15, 30
            
            # Black border (outline)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        cv2.putText(overlay, text, 
                                   (text_x + dx, text_y + dy), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 3)
            # White text
            cv2.putText(overlay, text, 
                       (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)
            
            return overlay
        
        # Calculate probabilities for each detected class
        h, w = mask.shape
        total_pixels = h * w
        
        # Add labels for each detected class at the top
        y_start = 25
        y_spacing = 25
        font_scale = 0.6
        
        for i, cls_idx in enumerate(detected_classes):
            if cls_idx < len(CLASSES):
                class_name = CLASSES[cls_idx]
                
                # Calculate percentage using probability if available, otherwise pixel count
                if probs is not None and cls_idx < probs.shape[0]:
                    # Use model probabilities (more accurate)
                    try:
                        # Get the probability map for this class
                        class_prob_map = probs[cls_idx]
                        
                        # Ensure the array is contiguous and the right type for OpenCV
                        class_prob_map = np.ascontiguousarray(class_prob_map, dtype=np.float32)
                        
                        # Check dimensions before resize
                        if len(class_prob_map.shape) != 2:
                            print(f"⚠️ Warning: Probability map has wrong shape {class_prob_map.shape}, expected 2D")
                            raise ValueError("Invalid probability map shape")
                        
                        # Resize probability map to match mask size
                        class_prob_resized = cv2.resize(class_prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Calculate mean probability for pixels classified as this class
                        class_mask = (mask == cls_idx)
                        if np.sum(class_mask) > 0:
                            mean_prob = np.mean(class_prob_resized[class_mask])
                            percentage = mean_prob * 100
                            prob_suffix = f" (prob: {percentage:.1f}%)"
                        else:
                            # Fallback to pixel percentage
                            class_pixels = np.sum(class_mask)
                            percentage = (class_pixels / total_pixels) * 100
                            prob_suffix = f" (area: {percentage:.1f}%)"
                    except Exception as e:
                        # Fallback to pixel count if probability processing fails
                        if self.config.get('verbose', False):
                            print(f"⚠️ Probability processing failed for class {cls_idx}: {e}")
                        class_pixels = np.sum(mask == cls_idx)
                        percentage = (class_pixels / total_pixels) * 100
                        prob_suffix = f" (area: {percentage:.1f}%)"
                else:
                    # Fallback to pixel count percentage
                    class_pixels = np.sum(mask == cls_idx)
                    percentage = (class_pixels / total_pixels) * 100
                    prob_suffix = f" (area: {percentage:.1f}%)"
                
                # Create label with percentage
                label_text = f"{class_name.replace('_', ' ').title()}{prob_suffix}"
                
                # Position text at top of image
                text_x = 15
                text_y = y_start + (i * y_spacing)
                
                # Don't go beyond image height
                if text_y > h - 30:
                    break
                
                # Black border (outline) for readability
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            cv2.putText(overlay, label_text, 
                                       (text_x + dx, text_y + dy), cv2.FONT_HERSHEY_TRIPLEX, 
                                       font_scale, (0, 0, 0), 3)
                
                # White text
                cv2.putText(overlay, label_text, 
                           (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 
                           font_scale, (255, 255, 255), 2)
        
        return overlay
    
    def _save_probability_maps(self, probs: np.ndarray, base_name: str, 
                              original_size: Tuple[int, int], prob_dir: str):
        """Save probability maps for each class"""
        img_prob_dir = os.path.join(prob_dir, base_name)
        os.makedirs(img_prob_dir, exist_ok=True)
        
        for cls_idx, cls_name in enumerate(CLASSES):
            prob_map = probs[cls_idx]
            prob_map_resized = cv2.resize(prob_map, original_size[::-1], interpolation=cv2.INTER_LINEAR)
            
            # Convert to heatmap
            heatmap = (prob_map_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            prob_path = os.path.join(img_prob_dir, f"{cls_name}.png")
            cv2.imwrite(prob_path, heatmap_colored)
    
    def run_inference(self, model_path: str, architecture: str, encoder_override: str = None, 
                     auto_detect_encoder: bool = True) -> Dict:
        """Run inference on all test images"""
        print(f"\n🚀 Starting inference with {architecture}")
        if encoder_override:
            print(f"🔧 Using encoder override: {encoder_override}")
        print("=" * 70)
        
        # Load model
        model = self.load_model(model_path, architecture, encoder_override, auto_detect_encoder)
        
        # Setup data
        image_paths = self._get_image_paths(self.config['input_dir'])
        dataset = TestDataset(image_paths, architecture, self.config['img_size'])
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Setup output directories
        output_dirs = self._setup_output_directories(self.config['output_dir'])
        
        # Statistics tracking
        inference_stats = {
            'total_images': len(image_paths),
            'total_time': 0,
            'class_distributions': {cls: 0 for cls in CLASSES},
            'processing_times': []
        }
        
        # Run inference
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                batch_start = time.time()
                
                images, paths, original_sizes = batch
                images = images.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision if available
                if self.config.get('mixed_precision', True) and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                # Get predictions
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                
                # Process each image in batch
                for i, (pred, path, size) in enumerate(zip(preds, paths, original_sizes)):
                    self._process_single_prediction(
                        pred, path, size, probs[i].cpu().numpy(), output_dirs, inference_stats
                    )
                
                batch_time = time.time() - batch_start
                inference_stats['processing_times'].append(batch_time)
                inference_stats['total_time'] += batch_time
        
        # Compute final statistics
        self._compute_inference_statistics(inference_stats)
        
        print(f"\n✅ Inference complete! Results saved to {self.config['output_dir']}")
        
        # Run evaluation if requested
        evaluation_results = None
        if self.config.get('evaluate', False) and self.config.get('gt_dir'):
            evaluation_results = self.run_evaluation(output_dirs.get('masks'))
        
        return {
            'inference_stats': inference_stats,
            'evaluation_results': evaluation_results,
            'architecture': architecture,
            'config': self.config
        }
    
    def _process_single_prediction(self, pred: np.ndarray, path: str, size: torch.Tensor,
                                  probs: np.ndarray, output_dirs: Dict[str, str],
                                  stats: Dict):
        """Process a single prediction and save outputs"""
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        
        # Safety check: Skip mask files (shouldn't happen after filtering, but just in case)
        if '_mask' in filename.lower():
            print(f"⚠️ Skipping mask file that somehow got through: {filename}")
            return
        
        # Get original dimensions
        size_np = size.numpy()
        h, w = int(size_np[0]), int(size_np[1])
        
        # Resize prediction to original size
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Update class distribution statistics
        unique, counts = np.unique(pred_resized, return_counts=True)
        total_pixels = pred_resized.size
        
        for cls_idx, count in zip(unique, counts):
            if cls_idx < NUM_CLASSES:
                stats['class_distributions'][CLASSES[cls_idx]] += count / total_pixels
        
        # Save mask
        if 'masks' in output_dirs:
            mask_path = os.path.join(output_dirs['masks'], f"{base_name}_mask.png")
            cv2.imwrite(mask_path, pred_resized.astype(np.uint8))
        
        # Save overlay
        if 'overlays' in output_dirs:
            original_image = cv2.imread(path)
            if original_image is None:
                print(f"⚠️ Could not load original image: {path}")
                return
            
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Debug: Check image properties
            if self.config.get('verbose', False):
                try:
                    print(f"🖼️ Original image shape: {original_image.shape}, Prediction shape: {pred_resized.shape}")
                except (BrokenPipeError, IOError):
                    pass
            
            overlay = self._create_overlay(original_image, pred_resized, probs, debug_info=f"Overlay for {filename}")
            overlay_path = os.path.join(output_dirs['overlays'], f"{base_name}_overlay.png")
            
            # Save overlay
            success = cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if self.config.get('verbose', False):
                try:
                    print(f"💾 Overlay saved: {overlay_path} (Success: {success})")
                except (BrokenPipeError, IOError):
                    pass
        
        # Save probability maps
        if 'probabilities' in output_dirs:
            self._save_probability_maps(probs, base_name, (h, w), output_dirs['probabilities'])
        
        # Verbose output with error handling
        if self.config.get('verbose', False):
            try:
                print(f"📸 Processed {filename}")
                print(f"   Unique classes detected: {unique}")
                print(f"   Class distribution:")
                for cls_idx, count in zip(unique, counts):
                    if cls_idx < NUM_CLASSES:
                        percentage = (count / total_pixels) * 100
                        print(f"     {CLASSES[cls_idx]} (class {cls_idx}): {count:,} pixels ({percentage:.2f}%)")
                    else:
                        print(f"     Unknown class {cls_idx}: {count:,} pixels ({(count / total_pixels) * 100:.2f}%)")
                
                # Show defect detection summary
                defect_classes = [1, 2, 3, 4, 5]  # pothole, crack, puddle, distressed_patch, mud
                total_defect_pixels = sum(counts[np.where(unique == cls)[0][0]] for cls in defect_classes if cls in unique)
                if total_defect_pixels > 0:
                    defect_percentage = (total_defect_pixels / total_pixels) * 100
                    print(f"   🔍 Total defect coverage: {total_defect_pixels:,} pixels ({defect_percentage:.2f}%)")
                else:
                    print(f"   ✅ No defects detected in this image")
            except (BrokenPipeError, IOError):
                # Handle broken pipe errors gracefully
                pass
    
    def _compute_inference_statistics(self, stats: Dict):
        """Compute and display inference statistics"""
        print(f"\n📊 Inference Statistics:")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Total Time: {stats['total_time']:.2f}s")
        print(f"   Average Time per Image: {stats['total_time'] / stats['total_images']:.3f}s")
        print(f"   Images per Second: {stats['total_images'] / stats['total_time']:.1f}")
        
        if stats['processing_times']:
            print(f"   Batch Processing Time: {np.mean(stats['processing_times']):.3f}s ± {np.std(stats['processing_times']):.3f}s")
        
        print(f"\n🎨 Average Class Distribution:")
        total_samples = stats['total_images']
        for cls_name, total_pixels in stats['class_distributions'].items():
            avg_percentage = (total_pixels / total_samples) * 100 if total_samples > 0 else 0
            print(f"   {cls_name:>16s}: {avg_percentage:.2f}%")
    
    def run_evaluation(self, pred_dir: str) -> Optional[Dict]:
        """Run evaluation against ground truth"""
        if not self.config.get('gt_dir') or not os.path.exists(self.config['gt_dir']):
            print("❌ Ground truth directory not available for evaluation")
            return None
        
        print(f"\n🔍 Running evaluation...")
        return self._evaluate_predictions(pred_dir, self.config['gt_dir'])
    
    def run_direct_evaluation(self, model_path: str, architecture: str, encoder_override: str = None,
                             auto_detect_encoder: bool = True) -> Optional[Dict]:
        """
        Run evaluation directly on images and ground truth masks without saving predictions.
        Useful for quick evaluation without generating all outputs.
        """
        if not self.config.get('gt_dir') or not os.path.exists(self.config['gt_dir']):
            print("❌ Ground truth directory not available for evaluation")
            return None
        
        print(f"\n🔍 Running direct evaluation with {architecture}...")
        if encoder_override:
            print(f"🔧 Using encoder override: {encoder_override}")
        
        # Load model
        model = self.load_model(model_path, architecture, encoder_override, auto_detect_encoder)
        
        # Get image-mask pairs
        pairs = self._get_image_mask_pairs(self.config['input_dir'], self.config['gt_dir'])
        
        if not pairs:
            print("❌ No valid image-mask pairs found for evaluation")
            return None
        
        total_iou_per_class = np.zeros(NUM_CLASSES)
        total_precision_per_class = np.zeros(NUM_CLASSES)
        total_recall_per_class = np.zeros(NUM_CLASSES)
        total_f1_per_class = np.zeros(NUM_CLASSES)
        total_samples = 0
        class_presence_stats = {}  # Track how often each class appears
        
        # Binary defect detection accumulation
        binary_defect_metrics = {
            'total_binary_iou': 0.0,
            'total_precision': 0.0,
            'total_recall': 0.0,
            'total_f1': 0.0,
            'total_gt_defect_pixels': 0,
            'total_pred_defect_pixels': 0
        }
        
        model.eval()
        with torch.no_grad():
            for image_path, mask_path in tqdm(pairs, desc="Evaluating"):
                try:
                    # Load and preprocess image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    original_size = image.shape[:2]  # (height, width)
                    
                    # Apply transforms for inference - USE EXACT SAME VALIDATION TRANSFORM AS TRAINING
                    transform = get_val_transform()  # This ensures square images just like during training validation
                    transformed = transform(image=image)
                    image_transformed = transformed['image']
                    
                    # DEBUG: Check if validation transform worked correctly
                    if self.config.get('verbose', False):
                        print(f"🔍 Transform Debug:")
                        print(f"   Original image shape: {original_size}")
                        print(f"   After validation transform: {image_transformed.shape}")
                        print(f"   Expected: (512, 512, 3)")
                    
                    # Get preprocessing based on architecture
                    arch_info = get_architecture_info(architecture)
                    if arch_info and arch_info.get('requires_smp', False):
                        # SMP preprocessing
                        encoder = arch_info.get('encoder', 'resnet50')
                        encoder_weights = arch_info.get('encoder_weights', 'imagenet')
                        preprocessing_fn = get_preprocessing_fn(encoder, encoder_weights)
                        
                        image_preprocessed = image_transformed.astype(np.float32) / 255.0
                        image_preprocessed = preprocessing_fn(image_preprocessed)
                    else:
                        # Standard preprocessing
                        image_preprocessed = image_transformed.astype(np.float32) / 255.0
                        image_preprocessed = (image_preprocessed - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    
                    # Convert to tensor with explicit float32 to avoid type mismatches
                    image_preprocessed = image_preprocessed.astype(np.float32)  # Ensure float32
                    image_tensor = torch.from_numpy(image_preprocessed).permute(2, 0, 1).unsqueeze(0)
                    image_tensor = image_tensor.to(self.device, dtype=torch.float32)  # Explicit dtype
                    
                    # Forward pass with explicit type handling
                    fix_tensor_dtype = self.config.get('fix_tensor_dtype', True)
                    
                    # Handle tensor type issues
                    if fix_tensor_dtype:
                        # Use float32 for everything to avoid precision issues
                        image_tensor = image_tensor.to(dtype=torch.float32)
                        if hasattr(model, 'float'):
                            model = model.float()
                    
                    if self.config.get('mixed_precision', True) and torch.cuda.is_available() and not fix_tensor_dtype:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(image_tensor)
                    else:
                        outputs = model(image_tensor)
                    
                    # Get prediction at model size
                    pred = torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy()[0]
                    
                    # DEBUG: Check raw prediction shape from model
                    if self.config.get('verbose', False):
                        print(f"🔍 Model Output Debug:")
                        print(f"   Input tensor shape: {image_tensor.shape}")
                        print(f"   Model output shape: {outputs.shape}")
                        print(f"   Raw prediction shape: {pred.shape}")
                        print(f"   Expected prediction shape: (512, 512)")
                    
                    # CRITICAL: Record exact original dimensions for precise resize
                    orig_h, orig_w = original_size[0], original_size[1]
                    
                    # Resize prediction back to EXACT original size
                    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Load and process ground truth
                    # For AutoMine dataset, always load as grayscale since masks contain class indices
                    if '_mask.png' in mask_path:
                        # AutoMine format - load as grayscale
                        gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if gt_mask_raw is None:
                            if self.config.get('verbose', False):
                                print(f"⚠️ Failed to read AutoMine mask: {mask_path}")
                            continue
                        gt_mask = self._gray_to_class_indices(gt_mask_raw).astype(np.int32)
                    else:
                        # SHREC/R2S format - load as RGB for color processing
                        gt_mask_raw = cv2.imread(mask_path)
                        if gt_mask_raw is None:
                            if self.config.get('verbose', False):
                                print(f"⚠️ Failed to read mask: {mask_path}")
                            continue
                        
                        # Ensure correct data type to prevent tensor type mismatch
                        if len(gt_mask_raw.shape) == 3:
                            gt_mask_raw = cv2.cvtColor(gt_mask_raw, cv2.COLOR_BGR2RGB)
                            gt_mask = self._rgb_to_class_indices(gt_mask_raw).astype(np.int32)
                        else:
                            gt_mask = self._gray_to_class_indices(gt_mask_raw).astype(np.int32)
                    
                    # CRITICAL: Final verification of exact shape match
                    if pred_resized.shape != gt_mask.shape:
                        print(f"⚠️ Shape mismatch detected - applying final alignment")
                        print(f"   Pred: {pred_resized.shape}, GT: {gt_mask.shape}")
                        pred_resized = cv2.resize(pred_resized, (gt_mask.shape[1], gt_mask.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                    
                    # Final assertion to catch any remaining issues
                    assert pred_resized.shape == gt_mask.shape, f"Shape mismatch: {pred_resized.shape} vs {gt_mask.shape}"
                    
                    # Debug spatial alignment for problematic cases
                    if self.config.get('verbose', False):
                        self.debug_spatial_alignment(pred_resized, gt_mask, image_path)
                    
                    # Compute IoU on original resolution masks
                    iou_results = self._compute_iou(pred_resized, gt_mask)
                    total_iou_per_class += np.nan_to_num(iou_results['iou_per_class'])
                    total_precision_per_class += np.nan_to_num(iou_results['precision_per_class'])
                    total_recall_per_class += np.nan_to_num(iou_results['recall_per_class'])
                    total_f1_per_class += np.nan_to_num(iou_results['f1_per_class'])
                    total_samples += 1
                    
                    # Accumulate binary defect detection metrics
                    binary_defect_metrics['total_binary_iou'] += iou_results['binary_defect_iou']
                    binary_defect_metrics['total_precision'] += iou_results['defect_precision']
                    binary_defect_metrics['total_recall'] += iou_results['defect_recall']
                    binary_defect_metrics['total_f1'] += iou_results['defect_f1']
                    binary_defect_metrics['total_gt_defect_pixels'] += iou_results['gt_defect_pixels']
                    binary_defect_metrics['total_pred_defect_pixels'] += iou_results['pred_defect_pixels']
                    
                    # Accumulate class presence statistics
                    for c in iou_results['present_classes']:
                        if c not in class_presence_stats:
                            class_presence_stats[c] = 0
                        class_presence_stats[c] += 1
                    
                except Exception as e:
                    if self.config.get('verbose', False):
                        print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
                    continue
        
        if total_samples == 0:
            print("❌ No valid samples processed for evaluation")
            return {}
        
        # Compute final metrics
        avg_iou_per_class = total_iou_per_class / total_samples
        avg_precision_per_class = total_precision_per_class / total_samples
        avg_recall_per_class = total_recall_per_class / total_samples
        avg_f1_per_class = total_f1_per_class / total_samples
        
        # Calculate average binary defect detection metrics
        avg_binary_defect_iou = binary_defect_metrics['total_binary_iou'] / total_samples
        avg_defect_precision = binary_defect_metrics['total_precision'] / total_samples  
        avg_defect_recall = binary_defect_metrics['total_recall'] / total_samples
        avg_defect_f1 = binary_defect_metrics['total_f1'] / total_samples
        
        # Calculate different types of mean IoU
        valid_ious = avg_iou_per_class[~np.isnan(avg_iou_per_class)]
        standard_mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
        
        # Present classes mean (only classes that appear in GT)
        if class_presence_stats:
            present_classes = list(class_presence_stats.keys())
            present_ious = [avg_iou_per_class[c] for c in present_classes if not np.isnan(avg_iou_per_class[c])]
            present_mean_iou = np.mean(present_ious) if present_ious else 0.0
            
            # Balanced mean (defect classes only, excluding background - class 0)
            defect_classes = [c for c in present_classes if c > 0]  # Exclude background (class 0)
            defect_ious = [avg_iou_per_class[c] for c in defect_classes if not np.isnan(avg_iou_per_class[c])]
            balanced_mean_iou = np.mean(defect_ious) if defect_ious else 0.0
            
            # Weighted mean by frequency of appearance
            total_appearances = sum(class_presence_stats.values())
            weighted_sum = 0.0
            for c in present_classes:
                if not np.isnan(avg_iou_per_class[c]):
                    weight = class_presence_stats[c] / total_appearances
                    weighted_sum += avg_iou_per_class[c] * weight
            weighted_mean_iou = weighted_sum
        else:
            present_mean_iou = 0.0
            balanced_mean_iou = 0.0
            weighted_mean_iou = 0.0
            present_classes = []
        
        # Display results
        print(f"\n{'='*60}")
        print(f"DIRECT EVALUATION RESULTS ({total_samples} images)")
        print(f"{'='*60}")
        for i, class_name in enumerate(CLASSES):
            appearance_count = class_presence_stats.get(i, 0)
            appearance_pct = (appearance_count / total_samples) * 100 if total_samples > 0 else 0
            iou_val = avg_iou_per_class[i]
            f1_val = avg_f1_per_class[i]
            
            # Format metrics for display
            iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "  N/A"
            f1_str = f"{f1_val:.4f}" if not np.isnan(f1_val) else "  N/A"
            
            print(f"{class_name:>16s}: IoU={iou_str}, F1={f1_str} (appears in {appearance_count}/{total_samples} images, {appearance_pct:.1f}%)")
        
        print(f"{'='*60}")
        print(f"{'Standard Mean IoU':>16s}: {standard_mean_iou:.4f} (all classes)")
        print(f"{'Present Mean IoU':>16s}: {present_mean_iou:.4f} (only classes in GT)")
        print(f"{'Weighted Mean IoU':>16s}: {weighted_mean_iou:.4f} (weighted by frequency)")
        print(f"{'Balanced Mean IoU':>16s}: {balanced_mean_iou:.4f} (defects only, no background)")
        print(f"{'Classes in GT':>16s}: {len(present_classes)}/{NUM_CLASSES}")
        print(f"{'='*60}")
        print(f"🎯 BINARY DEFECT DETECTION (AutoMine defect classes [1,3,4] vs Background)")
        print(f"{'Binary IoU':>16s}: {avg_binary_defect_iou:.4f}")
        print(f"{'Precision':>16s}: {avg_defect_precision:.4f}")
        print(f"{'Recall':>16s}: {avg_defect_recall:.4f}")
        print(f"{'F1 Score':>16s}: {avg_defect_f1:.4f}")
        total_pixels = binary_defect_metrics['total_gt_defect_pixels'] + binary_defect_metrics['total_pred_defect_pixels']
        if total_pixels > 0:
            gt_pct = (binary_defect_metrics['total_gt_defect_pixels'] / total_pixels) * 100
            pred_pct = (binary_defect_metrics['total_pred_defect_pixels'] / total_pixels) * 100
            print(f"{'GT Defect Pixels':>16s}: {binary_defect_metrics['total_gt_defect_pixels']:,} ({gt_pct:.1f}%)")
            print(f"{'Pred Defect Pixels':>16s}: {binary_defect_metrics['total_pred_defect_pixels']:,} ({pred_pct:.1f}%)")
        print(f"{'='*60}")
        
        return {
            'iou_per_class': avg_iou_per_class.tolist(),
            'precision_per_class': avg_precision_per_class.tolist(),
            'recall_per_class': avg_recall_per_class.tolist(),
            'f1_per_class': avg_f1_per_class.tolist(),
            'standard_mean_iou': standard_mean_iou,
            'present_mean_iou': present_mean_iou,
            'weighted_mean_iou': weighted_mean_iou,
            'balanced_mean_iou': balanced_mean_iou,
            'class_presence_stats': class_presence_stats,
            'present_classes': present_classes,
            'num_samples': total_samples,
            'class_names': CLASSES,
            'evaluation_type': 'direct',
            # Binary defect detection metrics
            'binary_defect_iou': avg_binary_defect_iou,
            'binary_defect_precision': avg_defect_precision,
            'binary_defect_recall': avg_defect_recall,
            'binary_defect_f1': avg_defect_f1,
            'total_gt_defect_pixels': binary_defect_metrics['total_gt_defect_pixels'],
            'total_pred_defect_pixels': binary_defect_metrics['total_pred_defect_pixels']
        }

    def run_evaluation(self, pred_dir: str) -> Optional[Dict]:
        """Run evaluation against ground truth"""
        if not self.config.get('gt_dir') or not os.path.exists(self.config['gt_dir']):
            print("❌ Ground truth directory not available for evaluation")
            return None
        
        print(f"\n🔍 Running evaluation...")
        return self._evaluate_predictions(pred_dir, self.config['gt_dir'])
    
    def _evaluate_predictions(self, pred_dir: str, gt_dir: str) -> Dict:
        """Evaluate predictions against ground truth masks"""
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_mask.png')]
        
        if not pred_files:
            print("❌ No prediction mask files found")
            return {}
        
        total_iou_per_class = np.zeros(NUM_CLASSES)
        total_precision_per_class = np.zeros(NUM_CLASSES)
        total_recall_per_class = np.zeros(NUM_CLASSES)
        total_f1_per_class = np.zeros(NUM_CLASSES)
        total_samples = 0
        class_presence_stats = {}  # Track how often each class appears
        
        # Binary defect detection accumulation
        binary_defect_metrics = {
            'total_binary_iou': 0.0,
            'total_precision': 0.0,
            'total_recall': 0.0,
            'total_f1': 0.0,
            'total_gt_defect_pixels': 0,
            'total_pred_defect_pixels': 0
        }
        
        print(f"📋 Evaluating {len(pred_files)} predictions...")
        
        for pred_file in tqdm(pred_files, desc="Evaluating"):
            # Find corresponding ground truth
            base_name = pred_file.replace('_mask.png', '')
            
            # Try different ground truth naming conventions
            gt_candidates = [
                # AutoMine format: image.jpg -> image_mask.png
                f"{base_name}_mask.png",
                # Standard formats
                f"{base_name}.png",
                f"{base_name}_gt.png",
                f"{base_name}.tif",
                f"{base_name}.tiff",
                # SHREC/other formats
                f"{base_name}.jpg___fuse.png",
                f"{base_name}_label.png"
            ]
            
            gt_file = None
            for candidate in gt_candidates:
                gt_path = os.path.join(gt_dir, candidate)
                if os.path.exists(gt_path):
                    gt_file = candidate
                    break
            
            if gt_file is None:
                if self.config.get('verbose', False):
                    print(f"⚠️ No ground truth found for {base_name}")
                continue
            
            # Load masks
            pred_path = os.path.join(pred_dir, pred_file)
            gt_path = os.path.join(gt_dir, gt_file)
            
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            # Load ground truth mask - handle AutoMine format properly
            if '_mask.png' in gt_file:
                # AutoMine format - load as grayscale
                gt_mask_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask_raw is None:
                    continue
                gt_mask = self._gray_to_class_indices(gt_mask_raw).astype(np.int32)
            else:
                # SHREC/R2S format - load as RGB for color processing
                gt_mask_raw = cv2.imread(gt_path)
                if gt_mask_raw is None:
                    continue
                
                # Process ground truth mask with explicit data type
                if len(gt_mask_raw.shape) == 3:
                    gt_mask_raw = cv2.cvtColor(gt_mask_raw, cv2.COLOR_BGR2RGB)
                    gt_mask = self._rgb_to_class_indices(gt_mask_raw).astype(np.int32)
                else:
                    gt_mask = self._gray_to_class_indices(gt_mask_raw).astype(np.int32)
            
            if pred_mask is None:
                continue
            
            # CRITICAL: Ensure exact spatial alignment
            # GT mask is at its original resolution - resize prediction carefully
            if pred_mask.shape != gt_mask.shape:
                print(f"⚠️ Shape mismatch in standard evaluation - applying careful alignment")
                print(f"   Pred: {pred_mask.shape}, GT: {gt_mask.shape}")
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Final assertion to catch any remaining issues
            assert pred_mask.shape == gt_mask.shape, f"Shape mismatch: {pred_mask.shape} vs {gt_mask.shape}"
            
            # Debug spatial alignment for problematic cases
            if self.config.get('verbose', False):
                self.debug_spatial_alignment(pred_mask, gt_mask, pred_file)
            
            # Clean per-image evaluation header
            if not self.config.get('verbose', False):
                # Simple header for clean output
                print(f"\n📋 {base_name}")
            
            # Compute IoU on original resolution masks
            iou_results = self._compute_iou(pred_mask, gt_mask)
            total_iou_per_class += np.nan_to_num(iou_results['iou_per_class'])
            total_precision_per_class += np.nan_to_num(iou_results['precision_per_class'])
            total_recall_per_class += np.nan_to_num(iou_results['recall_per_class'])
            total_f1_per_class += np.nan_to_num(iou_results['f1_per_class'])
            total_samples += 1
            
            # Accumulate binary defect detection metrics
            binary_defect_metrics['total_binary_iou'] += iou_results['binary_defect_iou']
            binary_defect_metrics['total_precision'] += iou_results['defect_precision']
            binary_defect_metrics['total_recall'] += iou_results['defect_recall']
            binary_defect_metrics['total_f1'] += iou_results['defect_f1']
            binary_defect_metrics['total_gt_defect_pixels'] += iou_results['gt_defect_pixels']
            binary_defect_metrics['total_pred_defect_pixels'] += iou_results['pred_defect_pixels']
            
            # Accumulate class presence statistics
            for c in iou_results['present_classes']:
                if c not in class_presence_stats:
                    class_presence_stats[c] = 0
                class_presence_stats[c] += 1
        
        if total_samples == 0:
            print("❌ No valid prediction-ground truth pairs found")
            return {}
        
        # Compute final metrics
        avg_iou_per_class = total_iou_per_class / total_samples
        avg_precision_per_class = total_precision_per_class / total_samples
        avg_recall_per_class = total_recall_per_class / total_samples
        avg_f1_per_class = total_f1_per_class / total_samples
        
        # Calculate average binary defect detection metrics
        avg_binary_defect_iou = binary_defect_metrics['total_binary_iou'] / total_samples
        avg_defect_precision = binary_defect_metrics['total_precision'] / total_samples  
        avg_defect_recall = binary_defect_metrics['total_recall'] / total_samples
        avg_defect_f1 = binary_defect_metrics['total_f1'] / total_samples
        
        # Calculate different types of mean IoU
        valid_ious = avg_iou_per_class[~np.isnan(avg_iou_per_class)]
        standard_mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
        
        # Present classes mean (only classes that appear in GT)
        if class_presence_stats:
            present_classes = list(class_presence_stats.keys())
            present_ious = [avg_iou_per_class[c] for c in present_classes if not np.isnan(avg_iou_per_class[c])]
            present_mean_iou = np.mean(present_ious) if present_ious else 0.0
            
            # Balanced mean (defect classes only, excluding background - class 0)
            defect_classes = [c for c in present_classes if c > 0]  # Exclude background (class 0)
            defect_ious = [avg_iou_per_class[c] for c in defect_classes if not np.isnan(avg_iou_per_class[c])]
            balanced_mean_iou = np.mean(defect_ious) if defect_ious else 0.0
            
            # Weighted mean by frequency of appearance
            total_appearances = sum(class_presence_stats.values())
            weighted_sum = 0.0
            for c in present_classes:
                if not np.isnan(avg_iou_per_class[c]):
                    weight = class_presence_stats[c] / total_appearances
                    weighted_sum += avg_iou_per_class[c] * weight
            weighted_mean_iou = weighted_sum
        else:
            present_mean_iou = 0.0
            balanced_mean_iou = 0.0
            weighted_mean_iou = 0.0
            present_classes = []
        
        # Display results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS ({total_samples} images)")
        print(f"{'='*60}")
        for i, class_name in enumerate(CLASSES):
            appearance_count = class_presence_stats.get(i, 0)
            appearance_pct = (appearance_count / total_samples) * 100 if total_samples > 0 else 0
            iou_val = avg_iou_per_class[i]
            f1_val = avg_f1_per_class[i]
            
            # Format metrics for display
            iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "  N/A"
            f1_str = f"{f1_val:.4f}" if not np.isnan(f1_val) else "  N/A"
            
            print(f"{class_name:>16s}: IoU={iou_str}, F1={f1_str} (appears in {appearance_count}/{total_samples} images, {appearance_pct:.1f}%)")
        
        print(f"{'='*60}")
        print(f"{'Standard Mean IoU':>16s}: {standard_mean_iou:.4f} (all classes)")
        print(f"{'Present Mean IoU':>16s}: {present_mean_iou:.4f} (only classes in GT)")
        print(f"{'Weighted Mean IoU':>16s}: {weighted_mean_iou:.4f} (weighted by frequency)")
        print(f"{'Balanced Mean IoU':>16s}: {balanced_mean_iou:.4f} (defects only, no background)")
        print(f"{'Classes in GT':>16s}: {len(present_classes)}/{NUM_CLASSES}")
        print(f"{'='*60}")
        print(f"🎯 BINARY DEFECT DETECTION (AutoMine defect classes [1,3,4] vs Background)")
        print(f"{'Binary IoU':>16s}: {avg_binary_defect_iou:.4f}")
        print(f"{'Precision':>16s}: {avg_defect_precision:.4f}")
        print(f"{'Recall':>16s}: {avg_defect_recall:.4f}")
        print(f"{'F1 Score':>16s}: {avg_defect_f1:.4f}")
        total_pixels = binary_defect_metrics['total_gt_defect_pixels'] + binary_defect_metrics['total_pred_defect_pixels']
        if total_pixels > 0:
            gt_pct = (binary_defect_metrics['total_gt_defect_pixels'] / total_pixels) * 100
            pred_pct = (binary_defect_metrics['total_pred_defect_pixels'] / total_pixels) * 100
            print(f"{'GT Defect Pixels':>16s}: {binary_defect_metrics['total_gt_defect_pixels']:,} ({gt_pct:.1f}%)")
            print(f"{'Pred Defect Pixels':>16s}: {binary_defect_metrics['total_pred_defect_pixels']:,} ({pred_pct:.1f}%)")
        print(f"{'='*60}")
        
        return {
            'iou_per_class': avg_iou_per_class.tolist(),
            'precision_per_class': avg_precision_per_class.tolist(),
            'recall_per_class': avg_recall_per_class.tolist(),
            'f1_per_class': avg_f1_per_class.tolist(),
            'standard_mean_iou': standard_mean_iou,
            'present_mean_iou': present_mean_iou,
            'weighted_mean_iou': weighted_mean_iou,
            'balanced_mean_iou': balanced_mean_iou,
            'class_presence_stats': class_presence_stats,
            'present_classes': present_classes,
            'num_samples': total_samples,
            'class_names': CLASSES,
            # Binary defect detection metrics
            'binary_defect_iou': avg_binary_defect_iou,
            'binary_defect_precision': avg_defect_precision,
            'binary_defect_recall': avg_defect_recall,
            'binary_defect_f1': avg_defect_f1,
            'total_gt_defect_pixels': binary_defect_metrics['total_gt_defect_pixels'],
            'total_pred_defect_pixels': binary_defect_metrics['total_pred_defect_pixels']
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file with proper serialization"""
        import json
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy types to JSON-serializable types
        serializable_results = convert_numpy_types(results)
        
        # Add metadata
        serializable_results['metadata'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'config': self.config
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"💾 Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            # Fallback: save basic summary
            try:
                summary = {
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'basic_info': {
                        'architecture': results.get('architecture', 'unknown'),
                        'device': str(self.device)
                    }
                }
                with open(output_path.replace('.json', '_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"💾 Basic summary saved instead")
            except Exception as e2:
                print(f"❌ Failed to save even basic summary: {e2}")
    
    def debug_spatial_alignment(self, pred_resized: np.ndarray, gt_mask: np.ndarray, image_path: str) -> bool:
        """Debug spatial alignment issues"""
        print(f"🔍 Spatial Debug for {os.path.basename(image_path)}:")
        print(f"   Prediction shape: {pred_resized.shape}")
        print(f"   GT mask shape: {gt_mask.shape}")
        
        if pred_resized.shape != gt_mask.shape:
            print(f"   ❌ SHAPE MISMATCH! This will cause zero overlap.")
            return False
        
        # Check if defects are detected in similar regions
        pred_defects = pred_resized > 0
        gt_defects = gt_mask > 0
        
        if np.any(gt_defects):
            pred_center = np.mean(np.where(pred_defects), axis=1) if np.any(pred_defects) else [0, 0]
            gt_center = np.mean(np.where(gt_defects), axis=1)
            
            distance = np.sqrt(np.sum((pred_center - gt_center) ** 2))
            print(f"   Defect center distance: {distance:.1f} pixels")
            
            if distance > 50:  # Arbitrary threshold
                print(f"   ⚠️ Large spatial offset detected!")
        
        return True

    def _compute_iou(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """Compute IoU for each class with detailed debugging and proper weighting"""
        
        # DEBUG: Check shapes for spatial alignment
        print(f"🔍 IoU Debug - Pred shape: {pred.shape}, Target shape: {target.shape}")
        
        if pred.shape != target.shape:
            print(f"❌ CRITICAL: Shape mismatch will cause zero IoU!")
            print(f"   This explains your 0.0000 IoU scores!")
            print(f"   Pred shape: {pred.shape}")
            print(f"   GT shape: {target.shape}")
        
        # Ensure consistent data types to prevent DoubleTensor/HalfTensor issues
        pred = pred.flatten().astype(np.int32)
        target = target.flatten().astype(np.int32)
        
        iou_per_class = []
        present_classes = []  # Track which classes are actually present in GT
        class_weights = []    # Track relative importance of each class
        
        total_target_pixels = len(target)
        
        # Get unique classes for clean comparison output
        pred_unique = np.unique(pred)
        target_unique = np.unique(target)
        
        # Clean per-image comparison output
        pred_classes = [CLASSES[c] for c in pred_unique if c < len(CLASSES)]
        gt_classes = [CLASSES[c] for c in target_unique if c < len(CLASSES)]
        
        # Only show comparison if there are defects or if verbose mode
        has_gt_defects = any(c > 0 for c in target_unique)
        has_pred_defects = any(c > 0 for c in pred_unique)
        
        if has_gt_defects or has_pred_defects or self.config.get('verbose', False):
            print(f"📊 Prediction vs Ground Truth:")
            print(f"   Model predicted: {pred_classes}")
            print(f"   Ground truth has: {gt_classes}")
            
            # Show binary defect status
            if has_gt_defects and has_pred_defects:
                print(f"   Status: ✅ Model detected defects (GT has defects)")
            elif has_gt_defects and not has_pred_defects:
                print(f"   Status: ❌ Model missed defects (GT has defects)")
            elif not has_gt_defects and has_pred_defects:
                print(f"   Status: ⚠️  Model found defects (GT has none - false positive)")
            else:
                print(f"   Status: ✅ Both clean (correct)")
        
        if self.config.get('verbose', False):
            print(f"🔍 Detailed IoU computation:")
            print(f"   Prediction classes: {pred_unique}")
            print(f"   Ground truth classes: {target_unique}")
        
        for c in range(NUM_CLASSES):
            pred_c = pred == c
            target_c = target == c
            
            inter = np.logical_and(pred_c, target_c).sum()
            union = np.logical_or(pred_c, target_c).sum()
            
            pred_count = pred_c.sum()
            target_count = target_c.sum()
            
            # Calculate IoU
            if union == 0:
                # If neither prediction nor target has this class
                iou = float('nan')
            else:
                iou = float(inter) / float(union)
            
            # Track class presence and weight
            if target_count > 0:  # Class is present in ground truth
                present_classes.append(c)
                class_weight = target_count / total_target_pixels
                class_weights.append(class_weight)
            else:
                class_weights.append(0.0)  # Not present in GT
            
            if self.config.get('verbose', False) and (pred_count > 0 or target_count > 0):
                class_name = CLASSES[c] if c < len(CLASSES) else f"class_{c}"
                weight_pct = (class_weights[-1] * 100) if class_weights else 0
                print(f"   {class_name}: pred={pred_count}, gt={target_count}, inter={inter}, union={union}, IoU={iou:.4f}, weight={weight_pct:.2f}%")
            
            iou_per_class.append(iou)
        
        # Calculate per-class precision, recall, and F1 scores
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for c in range(NUM_CLASSES):
            pred_c = pred == c
            target_c = target == c
            
            tp = np.logical_and(pred_c, target_c).sum()  # True positives
            fp = np.logical_and(pred_c, ~target_c).sum()  # False positives
            fn = np.logical_and(~pred_c, target_c).sum()  # False negatives
            
            # Precision: TP / (TP + FP)
            if tp + fp > 0:
                precision = float(tp) / float(tp + fp)
            else:
                precision = float('nan')  # No predictions for this class
            
            # Recall: TP / (TP + FN)
            if tp + fn > 0:
                recall = float(tp) / float(tp + fn)
            else:
                recall = float('nan')  # No ground truth for this class
            
            # F1: 2 * (precision * recall) / (precision + recall)
            if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = float('nan')
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
            
            if self.config.get('verbose', False) and (tp + fp > 0 or tp + fn > 0):
                class_name = CLASSES[c] if c < len(CLASSES) else f"class_{c}"
                print(f"   {class_name}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        
        # Calculate different types of mean IoU
        iou_array = np.array(iou_per_class)
        
        # Standard mean (including NaN classes)
        valid_ious = iou_array[~np.isnan(iou_array)]
        standard_mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
        
        # Present classes mean (only classes that appear in GT)
        if present_classes:
            present_ious = [iou_per_class[c] for c in present_classes if not np.isnan(iou_per_class[c])]
            present_mean_iou = np.mean(present_ious) if present_ious else 0.0
        else:
            present_mean_iou = 0.0
        
        # Weighted mean (by GT class frequency)
        if present_classes and any(w > 0 for w in class_weights):
            weighted_sum = 0.0
            total_weight = 0.0
            for c in present_classes:
                if not np.isnan(iou_per_class[c]) and class_weights[c] > 0:
                    weighted_sum += iou_per_class[c] * class_weights[c]
                    total_weight += class_weights[c]
            weighted_mean_iou = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            weighted_mean_iou = 0.0
        
        # Binary defect detection IoU (any defect vs background)
        # AutoMine only has specific defect classes after mapping:
        # - Original value 1 (defect) → class 4 (distressed_patch)
        # - Original value 2 (pothole) → class 1 (pothole) 
        # - Original value 3 (puddle) → class 3 (puddle)
        # So AutoMine defect classes are: [1, 3, 4] (pothole, puddle, distressed_patch)
        
        # Create binary masks: 0 = background, 1 = any defect
        automine_defect_classes = [1, 3, 4]  # Only AutoMine defect classes after mapping
        gt_defects = np.isin(target, automine_defect_classes)  # AutoMine defect classes in GT
        pred_defects = np.isin(pred, automine_defect_classes)  # Same defect classes in prediction
        
        # Calculate binary IoU for defect detection
        defect_inter = np.logical_and(gt_defects, pred_defects).sum()
        defect_union = np.logical_or(gt_defects, pred_defects).sum()
        
        if defect_union > 0:
            binary_defect_iou = float(defect_inter) / float(defect_union)
        else:
            binary_defect_iou = 0.0  # No defects in GT or prediction
        
        # Additional binary metrics
        defect_tp = defect_inter
        defect_fp = np.logical_and(pred_defects, ~gt_defects).sum()
        defect_fn = np.logical_and(~pred_defects, gt_defects).sum()
        
        # Precision: Of all predicted defects, how many were actually defects?
        defect_precision = float(defect_tp) / float(defect_tp + defect_fp) if (defect_tp + defect_fp) > 0 else 0.0
        
        # Recall: Of all actual defects, how many did we detect?
        defect_recall = float(defect_tp) / float(defect_tp + defect_fn) if (defect_tp + defect_fn) > 0 else 0.0
        
        # F1 score
        defect_f1 = 2 * defect_precision * defect_recall / (defect_precision + defect_recall) if (defect_precision + defect_recall) > 0 else 0.0
        
        if self.config.get('verbose', False) or (gt_defects.sum() > 0 or pred_defects.sum() > 0):
            gt_defect_pixels = gt_defects.sum()
            pred_defect_pixels = pred_defects.sum()
            total_pixels = len(target)
            
            # Only show detailed binary info if verbose or if there are defects
            if self.config.get('verbose', False):
                print(f"🎯 Binary Defect Detection:")
                print(f"   GT defects: {gt_defect_pixels} pixels ({100*gt_defect_pixels/total_pixels:.2f}%)")
                print(f"   Pred defects: {pred_defect_pixels} pixels ({100*pred_defect_pixels/total_pixels:.2f}%)")
                print(f"   Intersection: {defect_inter} pixels")
                print(f"   Union: {defect_union} pixels")
                print(f"   Binary IoU: {binary_defect_iou:.4f}")
                print(f"   Precision: {defect_precision:.4f}")
                print(f"   Recall: {defect_recall:.4f}")
                print(f"   F1 Score: {defect_f1:.4f}")
            elif gt_defect_pixels > 0 or pred_defect_pixels > 0:
                # Simple binary summary for interesting cases
                print(f"   Binary detection: IoU={binary_defect_iou:.4f}, F1={defect_f1:.4f}")
                print(f"   📝 Note: Binary detection uses AutoMine defect classes {automine_defect_classes} only")
        
        return {
            'iou_per_class': iou_array,
            'precision_per_class': np.array(precision_per_class),
            'recall_per_class': np.array(recall_per_class),
            'f1_per_class': np.array(f1_per_class),
            'standard_mean_iou': standard_mean_iou,
            'present_mean_iou': present_mean_iou,
            'weighted_mean_iou': weighted_mean_iou,
            'present_classes': present_classes,
            # Binary defect detection metrics
            'binary_defect_iou': binary_defect_iou,
            'defect_precision': defect_precision,
            'defect_recall': defect_recall,
            'defect_f1': defect_f1,
            'gt_defect_pixels': gt_defects.sum(),
            'pred_defect_pixels': pred_defects.sum()
        }
    
    def _rgb_to_class_indices(self, mask_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB mask to class indices"""
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
    
    def _gray_to_class_indices(self, mask_gray: np.ndarray) -> np.ndarray:
        """Convert grayscale AutoMine mask to class indices using training mapping"""
        # Create a copy with explicit dtype to avoid issues
        mask_indices = np.zeros_like(mask_gray, dtype=np.uint8)
        
        # Apply the exact same mapping used during training
        for original_value, new_class in AUTOMINE_MAPPING.items():
            mask_indices[mask_gray == original_value] = new_class
        
        # Debug information if in verbose mode
        unique_original = np.unique(mask_gray)
        unique_mapped = np.unique(mask_indices)
        
        # Always show debug info for defect detection
        has_defects = any(val in [1, 2, 3] for val in unique_original)
        if self.config.get('verbose', False) or has_defects:
            # Only show clean summary if evaluation is running
            if not self.config.get('verbose', False):
                # Clean output - just show defect summary
                if has_defects:
                    defect_values = [v for v in unique_original if v in [1, 2, 3]]
                    print(f"🎯 GT has defects: {defect_values} → mapped to classes {[AUTOMINE_MAPPING.get(v, 0) for v in defect_values]}")
                return mask_indices
            
            # Verbose mode - show full details
            if has_defects:
                print(f"🎯 DEFECTS FOUND in mask! Original values: {unique_original}")
            
            print(f"🔍 AutoMine GT mask conversion:")
            print(f"   Original values: {unique_original}")
            print(f"   Mapped values: {unique_mapped}")
            print(f"   Mapping applied:")
            
            for orig_val in unique_original:
                mapped_val = AUTOMINE_MAPPING.get(orig_val, 0)
                pixel_count = np.sum(mask_gray == orig_val)
                pixel_pct = (pixel_count / mask_gray.size) * 100
                class_name = CLASSES[mapped_val] if mapped_val < len(CLASSES) else 'unknown'
                print(f"     {orig_val} → {mapped_val} ({class_name}): {pixel_count} pixels ({pixel_pct:.2f}%)")
            
            # Check for unmapped values
            unmapped = [v for v in unique_original if v not in AUTOMINE_MAPPING]
            if unmapped:
                print(f"   ⚠️ Unmapped values found: {unmapped} (will default to background)")
                # Map any unmapped values to background
                for v in unmapped:
                    mask_indices[mask_gray == v] = 0
                    
            # Show final class distribution
            final_unique, final_counts = np.unique(mask_indices, return_counts=True)
            print(f"   Final class distribution:")
            for cls_idx, count in zip(final_unique, final_counts):
                if cls_idx < len(CLASSES):
                    pct = (count / mask_gray.size) * 100
                    print(f"     Class {cls_idx} ({CLASSES[cls_idx]}): {count} pixels ({pct:.2f}%)")
                    
            # Summary for defect files
            if has_defects:
                defect_classes = [cls for cls in final_unique if cls in [1, 3, 4]]  # pothole, puddle, distressed_patch
                if defect_classes:
                    print(f"   ✅ Successfully mapped to defect classes: {defect_classes}")
                else:
                    print(f"   ❌ Failed to map to any defect classes!")
            
        return mask_indices

    def validate_automine_mapping(self):
        """Validate that AutoMine mapping is consistent with training configuration"""
        print("🔍 Validating AutoMine mapping consistency...")
        
        # Expected mapping from config.py
        expected_mapping = {
            0: 0,  # background -> background
            1: 4,  # defect -> distressed_patch
            2: 1,  # pothole -> pothole
            3: 3,  # puddle -> puddle
            4: 0,  # road -> background
            255: 0 # Unknown/invalid -> background
        }
        
        print("📋 AutoMine Original Classes (from _classes.csv):")
        automine_classes = {
            0: "background",
            1: "defect", 
            2: "pothole",
            3: "puddle",
            4: "road"
        }
        
        for orig_val, orig_name in automine_classes.items():
            mapped_val = AUTOMINE_MAPPING.get(orig_val, 0)
            mapped_name = CLASSES[mapped_val] if mapped_val < len(CLASSES) else "unknown"
            expected_val = expected_mapping.get(orig_val, 0)
            
            status = "✅" if mapped_val == expected_val else "❌"
            print(f"   {status} {orig_val} ({orig_name}) → {mapped_val} ({mapped_name})")
            
            if mapped_val != expected_val:
                print(f"      Expected: {expected_val} ({CLASSES[expected_val]})")
        
        print(f"\n📊 Final Model Classes:")
        for i, class_name in enumerate(CLASSES):
            print(f"   {i}: {class_name}")
        
        # Check if mapping matches training config
        mapping_matches = all(AUTOMINE_MAPPING.get(k, 0) == v for k, v in expected_mapping.items())
        if mapping_matches:
            print("✅ AutoMine mapping matches training configuration")
        else:
            print("❌ AutoMine mapping differs from training configuration")
            print("   This will cause evaluation errors!")
        
        return mapping_matches

def load_config_file(config_path: str) -> Dict:
    """Load configuration from a Python file"""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract configuration variables
    config = {}
    for attr in dir(config_module):
        if not attr.startswith('_') and attr.isupper():
            config[attr.lower()] = getattr(config_module, attr)
    
    return config


def main():
    """Main function with simplified command line interface"""
    parser = argparse.ArgumentParser(description="Modular Road Defect Segmentation Testing")
    
    # Only essential arguments - everything else comes from config
    parser.add_argument('--config', type=str, default='test_config.py',
                       help='Configuration file to load settings from (default: test_config.py)')
    parser.add_argument('--list_architectures', action='store_true',
                       help='List available architectures and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Override config verbose setting')
    parser.add_argument('--direct_evaluation', action='store_true',
                       help='Override config direct_evaluation setting')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of test images for debugging (overrides config)')
    
    args = parser.parse_args()
    
    # List architectures if requested
    if args.list_architectures:
        print("\n🏗️ Available Architectures for Testing:")
        list_architectures()
        return
    
    # Load configuration from file
    print(f"📁 Loading configuration from: {args.config}")
    try:
        config = load_config_file(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        print("Make sure test_config.py exists and is properly formatted")
        return
    
    # Apply command line overrides
    if args.verbose:
        config['verbose'] = True
    if args.direct_evaluation:
        config['direct_evaluation'] = True
    
    # Validate required configuration
    required_fields = ['architecture', 'model_path', 'input_dir', 'output_dir']
    missing_fields = [field for field in required_fields if not config.get(field)]
    
    if missing_fields:
        print(f"❌ Missing required configuration fields: {missing_fields}")
        print("Please check your test_config.py file")
        return
    
    # Extract settings from config
    architecture = config['architecture']
    model_path = config['model_path']
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    
    # Optional settings with defaults
    gt_dir = config.get('gt_dir')
    encoder_override = config.get('encoder_override')
    auto_detect_encoder = config.get('auto_detect_encoder', True)
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)
    img_size = config.get('img_size', 512)
    device = config.get('device', 'auto')
    save_overlays = config.get('save_overlays', True)
    save_masks = config.get('save_masks', True)
    save_probability_maps = config.get('save_probability_maps', False)
    evaluate = config.get('evaluate', True)
    direct_evaluation = config.get('direct_evaluation', False)
    mixed_precision = config.get('mixed_precision', True)
    verbose = config.get('verbose', False)
    fix_tensor_dtype = config.get('fix_tensor_dtype', True)
    
    # Apply command line overrides
    if args.verbose:
        verbose = True
    if args.direct_evaluation:
        direct_evaluation = True
    if args.limit is not None:
        config['limit'] = args.limit
    
    # Validate architecture
    available_architectures = get_available_architectures()
    if architecture not in available_architectures:
        print(f"❌ Architecture '{architecture}' not available.")
        print(f"Available architectures: {available_architectures}")
        return
    
    # Validate encoder override if specified
    if encoder_override:
        supported_encoders = get_supported_encoders(architecture)
        if encoder_override not in supported_encoders:
            print(f"❌ Encoder '{encoder_override}' not supported for {architecture}.")
            if len(supported_encoders) > 1:
                print(f"Supported encoders: {supported_encoders}")
            return
        else:
            print(f"✅ Encoder override '{encoder_override}' is supported for {architecture}")
    
    # Display encoder information for multi-encoder architectures
    supported_encoders = get_supported_encoders(architecture)
    if len(supported_encoders) > 1:
        print(f"🔧 Architecture '{architecture}' supports multiple encoders: {supported_encoders}")
        encoder_variants = get_encoder_variants(architecture)
        if encoder_variants:
            print("📋 Available encoder variants:")
            for variant, info in encoder_variants.items():
                print(f"   {variant}: {info.get('description', 'No description')}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Setup configuration
    config = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'gt_dir': gt_dir,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'img_size': img_size,
        'device': device,
        'save_overlays': save_overlays,
        'save_masks': save_masks,
        'save_probability_maps': save_probability_maps,
        'evaluate': evaluate,
        'direct_evaluation': direct_evaluation,
        'mixed_precision': mixed_precision,
        'verbose': verbose,
        'fix_tensor_dtype': fix_tensor_dtype
    }
    
    # Display configuration
    print("\n🔧 Test Configuration:")
    print("=" * 50)
    print(f"Architecture: {architecture}")
    if encoder_override:
        print(f"Encoder Override: {encoder_override}")
    elif auto_detect_encoder:
        print(f"Auto-detect Encoder: {auto_detect_encoder}")
    print(f"Model Path: {model_path}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Ground Truth Directory: {gt_dir or 'None'}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    print(f"Save Overlays: {save_overlays}")
    print(f"Save Masks: {save_masks}")
    print(f"Save Probability Maps: {save_probability_maps}")
    print(f"Evaluate: {evaluate}")
    print(f"Direct Evaluation: {direct_evaluation}")
    print("=" * 50)
    
    # Initialize tester and run
    tester = ModelTester(config)
    
    try:
        if direct_evaluation:
            # Run direct evaluation only (no outputs saved)
            print("\n🎯 Running direct evaluation mode...")
            results = tester.run_direct_evaluation(model_path, architecture, encoder_override, auto_detect_encoder)
            
            if results:
                # Save minimal results
                os.makedirs(output_dir, exist_ok=True)
                results_path = os.path.join(output_dir, 'evaluation_results.json')
                tester.save_results({'evaluation_results': results, 'architecture': architecture}, results_path)
                
                print(f"\n🎉 Direct evaluation completed!")
                print(f"📊 Results summary:")
                print(f"   Architecture: {architecture}")
                print(f"   Images evaluated: {results['num_samples']}")
                
                # Use the most appropriate mean IoU metric
                if 'present_mean_iou' in results:
                    print(f"   Present Mean IoU: {results['present_mean_iou']:.4f} (classes in GT)")
                if 'weighted_mean_iou' in results:
                    print(f"   Weighted Mean IoU: {results['weighted_mean_iou']:.4f} (frequency weighted)")
                if 'standard_mean_iou' in results:
                    print(f"   Standard Mean IoU: {results['standard_mean_iou']:.4f} (all classes)")
                    
                if 'present_classes' in results:
                    print(f"   Classes in GT: {len(results['present_classes'])}/{NUM_CLASSES}")
            else:
                print("❌ Direct evaluation failed")
        else:
            # Run full inference with optional evaluation
            results = tester.run_inference(model_path, architecture, encoder_override, auto_detect_encoder)
            
            # Save results
            results_path = os.path.join(output_dir, 'test_results.json')
            tester.save_results(results, results_path)
            
            print(f"\n🎉 Testing completed successfully!")
            print(f"📊 Results summary:")
            print(f"   Architecture: {architecture}")
            print(f"   Images processed: {results['inference_stats']['total_images']}")
            print(f"   Total time: {results['inference_stats']['total_time']:.2f}s")
            
            if results['evaluation_results']:
                eval_results = results['evaluation_results']
                # Use the present_mean_iou as the primary metric since it's most meaningful
                mean_iou_key = 'present_mean_iou' if 'present_mean_iou' in eval_results else 'standard_mean_iou'
                print(f"   Mean IoU: {eval_results[mean_iou_key]:.4f} ({mean_iou_key.replace('_', ' ')})")
                print(f"   Classes in GT: {len(eval_results.get('present_classes', []))}/{len(eval_results.get('class_names', []))}")
                
                # Add dataset imbalance warning if applicable
                present_classes = eval_results.get('present_classes', [])
                if len(present_classes) <= 2 and 0 in present_classes:
                    print(f"   ⚠️  Dataset appears heavily imbalanced (mostly background/road)")
                    print(f"   💡 Use 'Present Mean IoU' for more meaningful evaluation")
        
    except BrokenPipeError:
        # Handle broken pipe error (when output is piped and pipe is closed)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Handle broken pipe at the top level
        sys.exit(0)
