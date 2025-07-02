#!/usr/bin/env python3
"""
Simple and Reliable Testing Script for Road Defect Segmentation

Clean implementation that integrates with the modular training system.
Focused on reliability and proper logging.

Usage:
    python test_simple.py
    
All settings are loaded from test_config.py
"""

import sys
import os
import json
import time
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Core dependencies
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Image processing
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import modular training dependencies
try:
    from architectures import (
        create_model, get_available_architectures, get_architecture_info,
        ARCHITECTURE_CONFIGS, get_supported_encoders, create_model_with_encoder
    )
    from augmentation import get_val_transform
    from config import IMG_SIZE, CLASSES, NUM_CLASSES
    # Import PotholeMixDataset for correct mask pairing
    from datasets import PotholeMixDataset, R2S100KDataset
    print("✅ Modular training system loaded successfully and PotholeMixDataset, R2S100KDataset imported")
except ImportError as e:
    print(f"❌ Error importing modular training system or datasets: {e}")
    sys.exit(1)

# Try to import SMP for preprocessing
try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    SMP_AVAILABLE = True
    print("✅ SMP available for preprocessing")
except ImportError:
    SMP_AVAILABLE = False
    print("⚠️ SMP not available - using standard preprocessing")


# Standard class list (should match your training)
CLASSES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
NUM_CLASSES = len(CLASSES)

# AutoMine to standard class mapping (update as needed)
AUTOMINE_MAPPING = {
    0: 0,  # background
    1: 4,  # defect -> distressed_patch
    2: 1,  # pothole
    3: 3,  # puddle
    4: 0,  # road -> background
}

# Add this near the top of your script, after CLASSES is defined:
class_colors = {
    0: (0, 0, 0),         # background: black
    1: (0, 0, 255),       # pothole: blue
    2: (0, 255, 0),       # crack: green
    3: (140, 160, 222),   # puddle: light blue
    4: (119, 61, 128),    # distressed_patch: purple
    5: (112, 84, 62)      # mud: brown
}

def remap_automine_mask(mask):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for k, v in AUTOMINE_MAPPING.items():
        remapped[mask == k] = v
    return remapped


class SimpleTestDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, image_paths: List[str], architecture: str):
        self.image_paths = image_paths
        self.architecture = architecture
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (height, width)
        
        # Apply validation transform (same as training)
        transform = get_val_transform()
        transformed = transform(image=image)
        image_transformed = transformed['image']
        
        # Apply preprocessing based on architecture
        arch_info = get_architecture_info(self.architecture)
        if arch_info and arch_info.get('requires_smp', False) and SMP_AVAILABLE:
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
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_preprocessed).permute(2, 0, 1)
        
        return image_tensor, image_path, original_size
    
    def __len__(self) -> int:
        return len(self.image_paths)


class SimpleTester:
    """Simple and reliable tester"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._setup_device()
        self.log_file = self._setup_logging()
        
        self.log(f"🔧 Initialized SimpleTester")
        self.log(f"   Device: {self.device}")
        self.log(f"   Architecture: {config['architecture']}")
        self.log(f"   Model: {config['model_path']}")
        
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name()
            return device
        else:
            return torch.device('cpu')
    
    def _setup_logging(self) -> str:
        """Setup logging to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"test_simple_{timestamp}.log"
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", log_file)
        
        return log_path
    
    def log(self, message: str):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} {message}\n")
            f.flush()
    
    def load_model(self) -> torch.nn.Module:
        """Load model"""
        architecture = self.config['architecture']
        model_path = self.config['model_path']
        
        self.log(f"🏗️ Loading {architecture} model from {model_path}")
        
        # Validate architecture
        if architecture not in ARCHITECTURE_CONFIGS:
            available = get_available_architectures()
            raise ValueError(f"Architecture '{architecture}' not available. Choose from: {available}")
        
        # Create model
        model = create_model(
            architecture=architecture,
            num_classes=NUM_CLASSES,
            img_size=IMG_SIZE
        )
        
        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            self.log("⚠️ Loading with weights_only=False for compatibility")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                self.log(f"📚 Loaded checkpoint from epoch {checkpoint['epoch']}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        self.log(f"✅ Model loaded successfully")
        return model
    
    def get_image_paths(self) -> List[str]:
        """Get image paths from input directory, only those with defects in GT mask"""
        input_dir = self.config['input_dir']
        image_paths = []
        all_images = glob.glob(os.path.join(input_dir, "*.jpg"))
        for image_path in all_images:
            gt_path = self.get_gt_mask_path(image_path)
            if gt_path and os.path.exists(gt_path):
                gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_raw is not None:
                    unique_vals = np.unique(gt_raw)
                    # Check for AutoMine defect values: 1, 2, 3
                    if any(val in [1, 2, 3] for val in unique_vals):
                        image_paths.append(image_path)
                        print(f"✅ Found defect image: {os.path.basename(image_path)} with GT values {unique_vals}")
                    else:
                        print(f"⏩ Skipped (no defect): {os.path.basename(image_path)} with GT values {unique_vals}")
                else:
                    print(f"⚠️ Could not read GT mask: {os.path.basename(gt_path)}")
            else:
                print(f"⚠️ GT mask missing for: {os.path.basename(image_path)}")
        print(f"🎯 Found {len(image_paths)} images with defects out of {len(all_images)} total")
        return image_paths
    
    def get_gt_mask_path(self, image_path: str) -> Optional[str]:
        """Get corresponding ground truth mask path"""
        if not self.config.get('gt_dir'):
            return None
            
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        
        # AutoMine format: image.jpg -> image_mask.png
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(self.config['gt_dir'], mask_name)
        
        if os.path.exists(mask_path):
            return mask_path
        return None
    
    def _automine_gray_to_class(self, gray_mask: np.ndarray) -> np.ndarray:
        """Convert AutoMine grayscale mask to class indices"""
        mask_indices = np.zeros_like(gray_mask, dtype=np.uint8)
        
        # Apply the exact same mapping used during training
        for original_value, new_class in AUTOMINE_MAPPING.items():
            mask_indices[gray_mask == original_value] = new_class
        
        # Debug output
        unique_original = np.unique(gray_mask)
        unique_mapped = np.unique(mask_indices)
        
        print(f"🔍 AutoMine GT conversion: {unique_original} → {unique_mapped}")
        
        return mask_indices.astype(np.int32)
    
    def compute_iou(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """Compute IoU metrics, including binary F1"""
        results = {}
        # Per-class IoU
        iou_per_class = []
        for class_id in range(NUM_CLASSES):
            pred_mask = (pred == class_id)
            gt_mask = (gt == class_id)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union == 0:
                iou = float('nan')  # Class not present
            else:
                iou = intersection / union
            iou_per_class.append(iou)
        results['iou_per_class'] = np.array(iou_per_class)
        # Mean IoU (excluding NaN)
        valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
        results['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0
        # Binary defect detection (defects vs background)
        pred_defects = (pred > 0).astype(int)
        gt_defects = (gt > 0).astype(int)
        intersection = np.logical_and(pred_defects, gt_defects).sum()
        union = np.logical_or(pred_defects, gt_defects).sum()
        results['binary_iou'] = intersection / union if union > 0 else 0.0
        # Binary F1
        tp = intersection
        fp = np.logical_and(pred_defects == 1, gt_defects == 0).sum()
        fn = np.logical_and(pred_defects == 0, gt_defects == 1).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            binary_f1 = 2 * (precision * recall) / (precision + recall)
        else:
            binary_f1 = 0.0
        results['binary_f1'] = binary_f1
        results['binary_precision'] = precision
        results['binary_recall'] = recall
        return results
    
    def debug_gt_mask(self, gt_path: str) -> None:
        """Debug what's actually in the ground truth mask"""
        if gt_path and os.path.exists(gt_path):
            gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                unique_vals = np.unique(gt_raw)
                print(f"🔍 GT Debug for {os.path.basename(gt_path)}:")
                print(f"   Raw GT values: {unique_vals}")
                
                # Show pixel counts
                for val in unique_vals:
                    count = np.sum(gt_raw == val)
                    percentage = (count / gt_raw.size) * 100
                    print(f"   Value {val}: {count:,} pixels ({percentage:.2f}%)")
                
                # Apply mapping and show result
                gt_mapped = self._automine_gray_to_class(gt_raw)
                unique_mapped = np.unique(gt_mapped)
                print(f"   Mapped values: {unique_mapped}")
                
                # Check if any defects remain after mapping
                defect_pixels = np.sum(gt_mapped > 0)
                total_pixels = gt_mapped.size
                print(f"   Final defects: {defect_pixels:,}/{total_pixels:,} ({100*defect_pixels/total_pixels:.2f}%)")
                
    def run_evaluation(self) -> Dict:
        """Run evaluation"""
        self.log("🚀 Starting evaluation...")
        # Load model
        model = self.load_model()
        # Get image paths or robust pairs
        dataset_name = self.config.get('dataset', 'automine').lower()
        if dataset_name == 'pothole_mix':
            pairs = self.get_pothole_mix_samples(self.config['input_dir'])
            if not pairs:
                self.log("❌ No valid pothole_mix samples found")
                return {}
            image_paths = [img for img, _ in pairs]
            mask_paths = {img: mask for img, mask in pairs}
        elif dataset_name == 'r2s100k':
            pairs = self.get_r2s100k_samples(self.config['input_dir'], self.config.get('gt_dir'))
            if not pairs:
                self.log("❌ No valid r2s100k samples found")
                return {}
            image_paths = [img for img, _ in pairs]
            mask_paths = {img: mask for img, mask in pairs}
        else:
            image_paths = self.get_image_paths()
            mask_paths = None
        # Apply limit if specified
        limit = self.config.get('limit')
        if limit and limit < len(image_paths):
            image_paths = image_paths[:limit]
            self.log(f"📊 Limited to {limit} images for testing")
        # Setup dataset and dataloader
        dataset = SimpleTestDataset(image_paths, self.config['architecture'])
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=lambda x: list(zip(*x))
        )
        if dataset_name == 'automine':
            return self._run_eval_automine(model, dataloader)
        elif dataset_name == 'pothole_mix':
            return self._run_eval_pothole_mix(model, dataloader, mask_paths)
        elif dataset_name == 'r2s100k':
            return self._run_eval_r2s100k(model, dataloader, mask_paths)
        else:
            self.log(f"❌ Unknown dataset: {dataset_name}")
            return {}

    def _run_eval_automine(self, model, dataloader):
        # Evaluation metrics
        total_samples = 0
        total_iou = 0.0
        total_binary_iou = 0.0
        class_ious = np.zeros(NUM_CLASSES)
        
        self.log(f"📊 Processing {len(dataloader.dataset.image_paths)} images...")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, paths, original_sizes) in enumerate(tqdm(dataloader, desc="Evaluating")):
                # images: tuple of tensors, paths: tuple of str, original_sizes: tuple of (h, w)
                images = torch.stack(images).to(self.device).float()
                paths = list(paths)
                original_sizes = list(original_sizes)
                preds = torch.argmax(F.softmax(model(images), dim=1), dim=1).cpu().numpy()
                # Now, each original_size is a tuple (h, w)
                for i, (pred, path, original_size) in enumerate(zip(preds, paths, original_sizes)):
                    if isinstance(original_size, (tuple, list)) and len(original_size) == 2:
                        orig_h, orig_w = int(original_size[0]), int(original_size[1])
                    else:
                        print(f"[ERROR] Invalid original_size for {path}: {original_size}. Skipping.")
                        continue
                    
                    # Resize prediction back to original size
                    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Load ground truth if available
                    gt_path = self.get_gt_mask_path(path)
                    if gt_path:
                        gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                        if gt_raw is not None:
                            # DEBUG: Show what we're working with
                            unique_raw = np.unique(gt_raw)
                            print(f"🔍 Processing {os.path.basename(path)}: Raw GT values {unique_raw}")
                            
                            # Apply PROPER AutoMine mapping
                            gt_mask = np.zeros_like(gt_raw, dtype=np.int32)
                            
                            # CORRECTED MAPPING:
                            gt_mask[gt_raw == 0] = 0  # background → background
                            gt_mask[gt_raw == 1] = 4  # defect → distressed_patch  
                            gt_mask[gt_raw == 2] = 1  # pothole → pothole
                            gt_mask[gt_raw == 3] = 3  # puddle → puddle
                            gt_mask[gt_raw == 4] = 0  # road → background
                            
                            # DEBUG: Show mapping result
                            unique_mapped = np.unique(gt_mask)
                            defect_pixels = np.sum(gt_mask > 0)
                            print(f"   Mapped to classes: {unique_mapped}, defect pixels: {defect_pixels}")
                            
                            # Ensure shapes match
                            if pred_resized.shape != gt_mask.shape:
                                pred_resized = cv2.resize(pred_resized, (gt_mask.shape[1], gt_mask.shape[0]), 
                                                        interpolation=cv2.INTER_NEAREST)
                            
                            # Show prediction too
                            unique_pred = np.unique(pred_resized)
                            pred_defect_pixels = np.sum(pred_resized > 0)
                            print(f"   Prediction classes: {unique_pred}, defect pixels: {pred_defect_pixels}")

                            # --- VISUALIZATION ---
                            # Load original image (RGB)
                            orig_img = cv2.imread(path)
                            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                            # Colorize GT mask
                            gt_color = np.zeros_like(orig_img)
                            color_map = {
                                0: (0, 0, 0),         # background: black
                                1: (0, 0, 255),       # pothole: blue
                                2: (0, 255, 0),       # crack: green
                                3: (140, 160, 222),   # puddle: light blue
                                4: (119, 61, 128),    # distressed_patch: purple
                                5: (112, 84, 62)      # mud: brown
                            }
                            for cls, color in color_map.items():
                                gt_color[gt_mask == cls] = color

                            # Colorize prediction
                            pred_color = np.zeros_like(orig_img)
                            for cls, color in color_map.items():
                                pred_color[pred_resized == cls] = color

                            # Overlay GT and prediction on original image
                            gt_overlay = cv2.addWeighted(orig_img, 0.5, gt_color, 0.5, 0)
                            pred_overlay = cv2.addWeighted(orig_img, 0.5, pred_color, 0.5, 0)

                            # Concatenate for visual comparison
                            vis = np.concatenate([
                                orig_img,
                                gt_overlay,
                                pred_overlay
                            ], axis=1)

                            # Save visualization
                            out_dir = os.path.join("visual_results")
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, os.path.basename(path).replace('.jpg', '_vis.png'))
                            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                            print(f"   Visualization saved: {out_path}")
                            
                            # Compute metrics from final mask vs. GT
                            metrics = self.compute_iou(pred_resized, gt_mask)
                            total_iou += metrics['mean_iou']
                            total_binary_iou += metrics['binary_iou']
                            class_ious += np.nan_to_num(metrics['iou_per_class'])
                            total_samples += 1
                            if 'binary_f1_sum' not in locals():
                                binary_f1_sum = 0.0
                                binary_precision_sum = 0.0
                                binary_recall_sum = 0.0
                            binary_f1_sum += metrics['binary_f1']
                            binary_precision_sum += metrics['binary_precision']
                            binary_recall_sum += metrics['binary_recall']
                            # Save three-way comparison image (GT, Prediction, Prediction+Cascade)
                            # Log per-image metrics
                            self.log(f"{os.path.basename(path)}: mIoU={metrics['mean_iou']:.4f}, BinIoU={metrics['binary_iou']:.4f}, BinF1={metrics['binary_f1']:.4f}")
                            
                            # Log progress every 10 images
                            if self.config.get('verbose') and total_samples % 10 == 0:
                                self.log(f"   Processed {total_samples} images, current mIoU: {total_iou/total_samples:.4f}")

        # MOVED OUTSIDE THE LOOPS - Compute final metrics AFTER all images are processed
        if total_samples > 0:
            avg_iou = total_iou / total_samples
            avg_binary_iou = total_binary_iou / total_samples
            avg_class_ious = class_ious / total_samples
            avg_binary_f1 = binary_f1_sum / total_samples if 'binary_f1_sum' in locals() else 0.0
            avg_binary_precision = binary_precision_sum / total_samples if 'binary_precision_sum' in locals() else 0.0
            avg_binary_recall = binary_recall_sum / total_samples if 'binary_recall_sum' in locals() else 0.0
            self.log("=" * 60)
            self.log("EVALUATION RESULTS")
            self.log("=" * 60)
            
            for i, class_name in enumerate(CLASSES):
                iou_val = avg_class_ious[i]
                iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "  N/A"
                self.log(f"{class_name:>16s}: IoU={iou_str}")
            
            self.log("=" * 60)
            self.log(f"{'Mean IoU':>16s}: {avg_iou:.4f}")
            self.log(f"{'Binary IoU':>16s}: {avg_binary_iou:.4f}")
            self.log(f"{'Binary F1':>16s}: {avg_binary_f1:.4f}")
            self.log(f"{'Bin Precision':>16s}: {avg_binary_precision:.4f}")
            self.log(f"{'Bin Recall':>16s}: {avg_binary_recall:.4f}")
            self.log(f"{'Samples':>16s}: {total_samples}")
            self.log("=" * 60)
            
            results = {
                'mean_iou': avg_iou,
                'binary_iou': avg_binary_iou,
                'binary_f1': avg_binary_f1,
                'binary_precision': avg_binary_precision,
                'binary_recall': avg_binary_recall,
                'class_ious': avg_class_ious.tolist(),
                'num_samples': total_samples,
                'architecture': self.config['architecture']
            }
        else:
            self.log("❌ No valid samples with ground truth found")
            results = {}
        
        self.log(f"✅ Evaluation completed! Log saved to: {self.log_file}")
        return results

    def _run_eval_pothole_mix(self, model, dataloader, mask_paths=None):
        total_samples = 0
        total_iou = 0.0
        total_binary_iou = 0.0
        class_ious = np.zeros(NUM_CLASSES)
        self.log(f"📊 Processing {len(dataloader.dataset.image_paths)} images (pothole_mix)...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, paths, original_sizes) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images = torch.stack(images).to(self.device).float()
                paths = list(paths)
                original_sizes = list(original_sizes)
                preds = torch.argmax(F.softmax(model(images), dim=1), dim=1).cpu().numpy()
                for i, (pred, path, original_size) in enumerate(zip(preds, paths, original_sizes)):
                    if isinstance(original_size, (tuple, list)) and len(original_size) == 2:
                        orig_h, orig_w = int(original_size[0]), int(original_size[1])
                    else:
                        print(f"[ERROR] Invalid original_size for {path}: {original_size}. Skipping.")
                        continue
                    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    gt_path = mask_paths[path] if mask_paths and path in mask_paths else self.get_gt_mask_path(path)
                    if gt_path:
                        gt_raw = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                        if gt_raw is not None:
                            gt_mask = remap_pothole_mix_mask(gt_raw)
                            # Ensure gt_mask matches original image size
                            orig_img = cv2.imread(path)
                            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                            if gt_mask.shape != orig_img.shape[:2]:
                                gt_mask = cv2.resize(gt_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            unique_raw = np.unique(gt_raw.reshape(-1, 3), axis=0)
                            print(f"🔍 Processing {os.path.basename(path)}: Raw GT RGB values {unique_raw}")
                            unique_mapped = np.unique(gt_mask)
                            defect_pixels = np.sum(gt_mask > 0)
                            print(f"   Mapped to classes: {unique_mapped}, defect pixels: {defect_pixels}")
                            if pred_resized.shape != gt_mask.shape:
                                pred_resized = cv2.resize(pred_resized, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                            unique_pred = np.unique(pred_resized)
                            pred_defect_pixels = np.sum(pred_resized > 0)
                            print(f"   Prediction classes: {unique_pred}, defect pixels: {pred_defect_pixels}")
                            # Visualization (optional, similar to automine)
                            gt_color = np.zeros_like(orig_img)
                            color_map = {
                                0: (0, 0, 0),         # background: black
                                1: (0, 0, 255),       # pothole: blue
                                2: (0, 255, 0),       # crack: green
                                3: (140, 160, 222),   # puddle: light blue
                                4: (119, 61, 128),    # distressed_patch: purple
                                5: (112, 84, 62)      # mud: brown
                            }
                            for cls, color in color_map.items():
                                gt_color[gt_mask == cls] = color
                            pred_color = np.zeros_like(orig_img)
                            for cls, color in color_map.items():
                                pred_color[pred_resized == cls] = color
                            gt_overlay = cv2.addWeighted(orig_img, 0.5, gt_color, 0.5, 0)
                            pred_overlay = cv2.addWeighted(orig_img, 0.5, pred_color, 0.5, 0)
                            vis = np.concatenate([
                                orig_img,
                                gt_overlay,
                                pred_overlay
                            ], axis=1)
                            out_dir = os.path.join("visual_results")
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, os.path.basename(path).replace('.jpg', '_vis.png'))
                            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                            print(f"   Visualization saved: {out_path}")
                            metrics = self.compute_iou(pred_resized, gt_mask)
                            total_iou += metrics['mean_iou']
                            total_binary_iou += metrics['binary_iou']
                            class_ious += np.nan_to_num(metrics['iou_per_class'])
                            total_samples += 1
                            if 'binary_f1_sum' not in locals():
                                binary_f1_sum = 0.0
                                binary_precision_sum = 0.0
                                binary_recall_sum = 0.0
                            binary_f1_sum += metrics['binary_f1']
                            binary_precision_sum += metrics['binary_precision']
                            binary_recall_sum += metrics['binary_recall']
                            # Log per-image metrics
                            self.log(f"{os.path.basename(path)}: mIoU={metrics['mean_iou']:.4f}, BinIoU={metrics['binary_iou']:.4f}, BinF1={metrics['binary_f1']:.4f}")
                            if self.config.get('verbose') and total_samples % 10 == 0:
                                self.log(f"   Processed {total_samples} images, current mIoU: {total_iou/total_samples:.4f}")
        if total_samples > 0:
            avg_iou = total_iou / total_samples
            avg_binary_iou = total_binary_iou / total_samples
            avg_class_ious = class_ious / total_samples
            avg_binary_f1 = binary_f1_sum / total_samples if 'binary_f1_sum' in locals() else 0.0
            avg_binary_precision = binary_precision_sum / total_samples if 'binary_precision_sum' in locals() else 0.0
            avg_binary_recall = binary_recall_sum / total_samples if 'binary_recall_sum' in locals() else 0.0
            self.log("=" * 60)
            self.log("EVALUATION RESULTS")
            self.log("=" * 60)
            for i, class_name in enumerate(CLASSES):
                iou_val = avg_class_ious[i]
                iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "  N/A"
                self.log(f"{class_name:>16s}: IoU={iou_str}")
            self.log("=" * 60)
            self.log(f"{'Mean IoU':>16s}: {avg_iou:.4f}")
            self.log(f"{'Binary IoU':>16s}: {avg_binary_iou:.4f}")
            self.log(f"{'Binary F1':>16s}: {avg_binary_f1:.4f}")
            self.log(f"{'Bin Precision':>16s}: {avg_binary_precision:.4f}")
            self.log(f"{'Bin Recall':>16s}: {avg_binary_recall:.4f}")
            self.log(f"{'Samples':>16s}: {total_samples}")
            self.log("=" * 60)
            results = {
                'mean_iou': avg_iou,
                'binary_iou': avg_binary_iou,
                'binary_f1': avg_binary_f1,
                'binary_precision': avg_binary_precision,
                'binary_recall': avg_binary_recall,
                'class_ious': avg_class_ious.tolist(),
                'num_samples': total_samples,
                'architecture': self.config['architecture']
            }
        else:
            self.log("❌ No valid samples with ground truth found")
            results = {}
        self.log(f"✅ Evaluation completed! Log saved to: {self.log_file}")
        return results

    def _run_eval_r2s100k(self, model, dataloader, mask_paths=None):
        total_samples = 0
        total_iou = 0.0
        total_binary_iou = 0.0
        class_ious = np.zeros(NUM_CLASSES)
        self.log(f"📊 Processing {len(dataloader.dataset.image_paths)} images (r2s100k)...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, paths, original_sizes) in enumerate(tqdm(dataloader, desc="Evaluating")):
                images = torch.stack(images).to(self.device).float()
                paths = list(paths)
                original_sizes = list(original_sizes)
                preds = torch.argmax(F.softmax(model(images), dim=1), dim=1).cpu().numpy()
                for i, (pred, path, original_size) in enumerate(zip(preds, paths, original_sizes)):
                    if isinstance(original_size, (tuple, list)) and len(original_size) == 2:
                        orig_h, orig_w = int(original_size[0]), int(original_size[1])
                    else:
                        print(f"[ERROR] Invalid original_size for {path}: {original_size}. Skipping.")
                        continue
                    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    gt_path = mask_paths[path] if mask_paths and path in mask_paths else self.get_gt_mask_path(path)
                    if gt_path:
                        gt_raw = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                        if gt_raw is not None:
                            gt_mask = remap_r2s100k_mask(gt_raw)
                            orig_img = cv2.imread(path)
                            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                            if gt_mask.shape != orig_img.shape[:2]:
                                gt_mask = cv2.resize(gt_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            unique_mapped = np.unique(gt_mask)
                            defect_pixels = np.sum(gt_mask > 0)
                            print(f"   Mapped to classes: {unique_mapped}, defect pixels: {defect_pixels}")
                            if pred_resized.shape != gt_mask.shape:
                                pred_resized = cv2.resize(pred_resized, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                            unique_pred = np.unique(pred_resized)
                            pred_defect_pixels = np.sum(pred_resized > 0)
                            print(f"   Prediction classes: {unique_pred}, defect pixels: {pred_defect_pixels}")
                            orig_img = cv2.imread(path)
                            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                            gt_color = np.zeros_like(orig_img)
                            color_map = {
                                0: (0, 0, 0),         # background: black
                                1: (0, 0, 255),       # pothole: blue
                                2: (0, 255, 0),       # crack: green
                                3: (140, 160, 222),   # puddle: light blue
                                4: (119, 61, 128),    # distressed_patch: purple
                                5: (112, 84, 62)      # mud: brown
                            }
                            for cls, color in color_map.items():
                                gt_color[gt_mask == cls] = color
                            pred_color = np.zeros_like(orig_img)
                            for cls, color in color_map.items():
                                pred_color[pred_resized == cls] = color
                            gt_overlay = cv2.addWeighted(orig_img, 0.5, gt_color, 0.5, 0)
                            pred_overlay = cv2.addWeighted(orig_img, 0.5, pred_color, 0.5, 0)
                            vis = np.concatenate([
                                orig_img,
                                gt_overlay,
                                pred_overlay
                            ], axis=1)
                            out_dir = os.path.join("visual_results")
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, os.path.basename(path).replace('.jpg', '_vis.png'))
                            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                            print(f"   Visualization saved: {out_path}")
                            metrics = self.compute_iou(pred_resized, gt_mask)
                            total_iou += metrics['mean_iou']
                            total_binary_iou += metrics['binary_iou']
                            class_ious += np.nan_to_num(metrics['iou_per_class'])
                            total_samples += 1
                            if 'binary_f1_sum' not in locals():
                                binary_f1_sum = 0.0
                                binary_precision_sum = 0.0
                                binary_recall_sum = 0.0
                            binary_f1_sum += metrics['binary_f1']
                            binary_precision_sum += metrics['binary_precision']
                            binary_recall_sum += metrics['binary_recall']
                            self.log(f"{os.path.basename(path)}: mIoU={metrics['mean_iou']:.4f}, BinIoU={metrics['binary_iou']:.4f}, BinF1={metrics['binary_f1']:.4f}")
                            if self.config.get('verbose') and total_samples % 10 == 0:
                                self.log(f"   Processed {total_samples} images, current mIoU: {total_iou/total_samples:.4f}")
        if total_samples > 0:
            avg_iou = total_iou / total_samples
            avg_binary_iou = total_binary_iou / total_samples
            avg_class_ious = class_ious / total_samples
            avg_binary_f1 = binary_f1_sum / total_samples if 'binary_f1_sum' in locals() else 0.0
            avg_binary_precision = binary_precision_sum / total_samples if 'binary_precision_sum' in locals() else 0.0
            avg_binary_recall = binary_recall_sum / total_samples if 'binary_recall_sum' in locals() else 0.0
            self.log("=" * 60)
            self.log("EVALUATION RESULTS")
            self.log("=" * 60)
            for i, class_name in enumerate(CLASSES):
                iou_val = avg_class_ious[i]
                iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "  N/A"
                self.log(f"{class_name:>16s}: IoU={iou_str}")
            self.log("=" * 60)
            self.log(f"{'Mean IoU':>16s}: {avg_iou:.4f}")
            self.log(f"{'Binary IoU':>16s}: {avg_binary_iou:.4f}")
            self.log(f"{'Binary F1':>16s}: {avg_binary_f1:.4f}")
            self.log(f"{'Bin Precision':>16s}: {avg_binary_precision:.4f}")
            self.log(f"{'Bin Recall':>16s}: {avg_binary_recall:.4f}")
            self.log(f"{'Samples':>16s}: {total_samples}")
            self.log("=" * 60)
            results = {
                'mean_iou': avg_iou,
                'binary_iou': avg_binary_iou,
                'binary_f1': avg_binary_f1,
                'binary_precision': avg_binary_precision,
                'binary_recall': avg_binary_recall,
                'class_ious': avg_class_ious.tolist(),
                'num_samples': total_samples,
                'architecture': self.config['architecture']
            }
        else:
            self.log("❌ No valid samples with ground truth found")
            results = {}
        self.log(f"✅ Evaluation completed! Log saved to: {self.log_file}")
        return results

    def get_pothole_mix_samples(self, input_dir):
        """Try to use PotholeMixDataset to get (img, mask) pairs for the test set."""
        try:
            dataset = PotholeMixDataset(input_dir, mode='test')
            if len(dataset) == 0:
                print(f"⚠️ PotholeMixDataset found no samples in {input_dir}. Check directory structure.")
                return []
            print(f"✅ PotholeMixDataset loaded {len(dataset)} samples from {input_dir}")
            return [(img, dataset.get_mask_path(i)) for i, (img, _) in enumerate(dataset.samples)]
        except Exception as e:
            print(f"❌ Could not use PotholeMixDataset for {input_dir}: {e}")
            return []

    def get_r2s100k_samples(self, images_dir, masks_dir=None):
        """Use R2S100KDataset to get (img, mask) pairs for the test set."""
        try:
            dataset = R2S100KDataset(images_dir, masks_dir)
            if len(dataset) == 0:
                print(f"⚠️ R2S100KDataset found no samples in {images_dir}. Check directory structure.")
                return []
            print(f"✅ R2S100KDataset loaded {len(dataset)} samples from {images_dir}")
            return [(os.path.join(images_dir, dataset.image_files[i]), dataset.get_mask_path(i)) for i in range(len(dataset))]
        except Exception as e:
            print(f"❌ Could not use R2S100KDataset for {images_dir}: {e}")
            return []

def load_config() -> Dict:
    """Load configuration from test_config.py"""
    try:
        import test_config
        config = {
            'architecture': getattr(test_config, 'ARCHITECTURE', 'unetplusplus'),
            'model_path': getattr(test_config, 'MODEL_PATH', './models/best_model.pth'),
            'input_dir': getattr(test_config, 'INPUT_DIR', '../data/automine/test'),
            'gt_dir': getattr(test_config, 'GT_DIR', '../data/automine/test'),
            'batch_size': getattr(test_config, 'BATCH_SIZE', 8),  # Use 8 as default for faster testing
            'num_workers': getattr(test_config, 'NUM_WORKERS', 4),
            'mixed_precision': getattr(test_config, 'MIXED_PRECISION', True),
            'verbose': getattr(test_config, 'VERBOSE', True),
            'limit': getattr(test_config, 'LIMIT', None),
            'dataset': getattr(test_config, 'DATASET', 'automine')  # NEW: dataset name
        }
        print(f"✅ Configuration loaded from test_config.py")
        return config
    except ImportError:
        print("❌ test_config.py not found, using defaults")
        return {
            'architecture': 'unetplusplus',
            'model_path': './models/best_model.pth',
            'input_dir': '../data/automine/test',
            'gt_dir': '../data/automine/test',
            'batch_size': 8,  # Use 8 as default for faster testing
            'num_workers': 4,
            'mixed_precision': True,
            'verbose': True,
            'limit': None,
            'dataset': 'automine'  # NEW: dataset name
        }


# Add dataset-specific mask remapping functions
POTHOLE_MIX_CLASS_MAP = {0: 0, 1: 1, 2: 2}  # background, pothole, crack (for 3-class)
R2S_COLORS = {
    "water_puddle": (140, 160, 222),  # Maps to class 'puddle' (3)
    "distressed_patch": (119, 61, 128), # Maps to class 'distressed_patch' (4)
    "mud": (112, 84, 62)              # Maps to class 'mud' (5)
}

SHREC_COLORS = {
    "crack_red": (255, 0, 0),   # Maps to class 'crack' (2)
    "crack_green": (0, 255, 0), # Maps to class 'crack' (2)
    "pothole": (0, 0, 255)      # Maps to class 'pothole' (1)
}

def remap_pothole_mix_mask(mask_rgb):
    # mask_rgb: H x W x 3, np.uint8
    mask = mask_rgb // 255
    target = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    mask_indices = np.zeros_like(target, dtype=np.uint8)
    mask_indices[target == 1] = 0  # background
    mask_indices[target == 2] = 2  # crack
    mask_indices[target == 3] = 0
    mask_indices[target == 4] = 1  # pothole
    mask_indices[target == 5] = 0
    mask_indices[target == 6] = 0
    mask_indices[target == 7] = 0
    return mask_indices

def remap_r2s100k_mask(mask_rgb):
    # mask_rgb: H x W x 3, np.uint8
    height, width = mask_rgb.shape[:2]
    mask_indices = np.zeros((height, width), dtype=np.uint8)
    # R2S colors
    for label, color in R2S_COLORS.items():
        r, g, b = color
        matching_pixels = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        if label == "water_puddle":
            mask_indices[matching_pixels] = 3
        elif label == "distressed_patch":
            mask_indices[matching_pixels] = 4
        elif label == "mud":
            mask_indices[matching_pixels] = 5
    # SHREC colors (for cracks/potholes)
    for label, color in SHREC_COLORS.items():
        r, g, b = color
        matching_pixels = (mask_rgb[:, :, 0] == r) & (mask_rgb[:, :, 1] == g) & (mask_rgb[:, :, 2] == b)
        if label in ["crack_red", "crack_green"]:
            mask_indices[matching_pixels] = 2
        elif label == "pothole":
            mask_indices[matching_pixels] = 1
    return mask_indices

def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        
        # Create tester and run evaluation
        tester = SimpleTester(config)
        results = tester.run_evaluation()
        
        # Save results
        if results:
            results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📊 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
