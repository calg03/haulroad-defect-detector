#!/usr/bin/env python
# filepath: /home/cloli/experimentation/cascade-road-segmentation/src/code/inference_unetpp_defect_multidata.py
"""
Inference script for UNet++ with SCSE blocks and EfficientNet-B5 encoder
for road defect segmentation.
"""
import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = None

import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# -----------------------------
# HARDCODED CONFIGURATION - MODIFY THESE AS NEEDED
# -----------------------------
MODEL_PATH = "./models/besto/unetpp_road_defect_gold.pt"  # Path to your trained model
INPUT_DIR = "../data/automine/test"  # Directory containing test images
OUTPUT_DIR = "./results/defect_segm"  # Output directory

# ENHANCED: Multi-dataset ground truth directory auto-detection
# Set GT_DIR to None for auto-detection based on INPUT_DIR pattern, or specify explicitly
GT_DIR = None  # Will auto-detect based on INPUT_DIR pattern

# Explicit GT directory mapping for each dataset pattern
# Format: input_pattern -> ground_truth_directory
GT_DIR_MAPPINGS = {
    # Automine dataset patterns
    "automine/test": "../data/automine/test",  # Same dir, masks have _mask.png suffix
    "automine/val": "../data/automine/val",
    "train/train_test": "../data/train/train_test",  # Alternative automine path
    "valid/train_test": "../data/valid/train_test",
    
    # PotholeMix dataset patterns
    "pothole_mix/testing": "../data/pothole_mix/testing",
    "pothole_mix/validation": "../data/pothole_mix/validation", 
    "pothole-mix/test": "../data/pothole-mix/testing",
    "pothole-mix/validation": "../data/pothole-mix/validation",
    
    # RTK dataset patterns
    "RTK/test": "../data/RTK/test",
    "RTK/validation": "../data/RTK/validation",
    "RTK": "../data/RTK",  # Single RTK directory
    
    # R2S100K dataset patterns
    "r2s100k/test": "../data/r2s100k/test-labels",
    "r2s100k/val": "../data/r2s100k/val-labels",
    "r2s100k/test-images": "../data/r2s100k/test-labels",
    "r2s100k/val-images": "../data/r2s100k/val-labels",
}

BATCH_SIZE = 8
SAVE_OVERLAYS = True
SAVE_PROBABILITY_MAPS = True
SAVE_MASKS = True  # Set to True if you want evaluation
EVALUATE = True  # Set to True if you have ground truth masks

# Model configuration - MUST match training exactly
ENCODER = 'efficientnet-b5'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 512  # Match training configuration

# Define the classes same as in training
CLASSES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
NUM_CLASSES = len(CLASSES)

# Color mapping for visualization - EXACTLY matching training colors
COLOR_MAP = {
    0: (0, 0, 0),                # background: black
    1: (0, 0, 255),              # pothole: blue (matching SHREC_COLORS pothole)
    2: (0, 255, 0),              # crack: green (matching SHREC_COLORS crack_green)
    3: (140, 160, 222),          # puddle: light blue (matching R2S_COLORS water_puddle)
    4: (119, 61, 128),           # distressed_patch: purple (matching R2S_COLORS distressed_patch)
    5: (112, 84, 62)             # mud: brown (matching R2S_COLORS mud)
}

# -----------------------------
# Color Mapping Functions (FROM TRAINING SCRIPT)
# -----------------------------
# SHREC colors (pothole_mix)
SHREC_COLORS = {
    "crack_red": (255, 0, 0),   # Maps to class 'crack' (2)
    "crack_green": (0, 255, 0), # Maps to class 'crack' (2)
    "pothole": (0, 0, 255)      # Maps to class 'pothole' (1)
}

# R2S100K colors
R2S_COLORS = {
    "water_puddle": (140, 160, 222),  # Maps to class 'puddle' (3)
    "distressed_patch": (119, 61, 128), # Maps to class 'distressed_patch' (4)
    "mud": (112, 84, 62)              # Maps to class 'mud' (5)
}

# automine classes mapping (from _classes.csv)
AUTOMINE_MAPPING = {
    0: 0,  # background -> background
    3: 2,  # defect -> crack
    11: 1,  # pothole -> pothole
    12: 3,  # puddle -> puddle
    13: 3,  # puddles -> puddle
    14: 0,  # road -> background
}

def rgb_to_class_indices(mask_rgb):
    """Convert RGB mask to class indices based on color mappings - FROM TRAINING"""
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
    """Convert grayscale mask to class indices based on pixel value mappings - FROM TRAINING"""
    mask_indices = np.zeros_like(mask_gray, dtype=np.uint8)
    for original_value, new_class in AUTOMINE_MAPPING.items():
        mask_indices[mask_gray == original_value] = new_class
    # Any unlisted class is mapped to background (0)
    return mask_indices

# -----------------------------
# Dataset Class for Inference
# -----------------------------
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None, preprocessing=None):
        self.image_paths = image_paths
        self.transform = transform
        self.preprocessing = preprocessing
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image size for later resizing - store as tuple to avoid dimension issues
        original_size = torch.tensor([image.shape[0], image.shape[1]])  # (height, width)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Apply preprocessing
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image)
            image = preprocessed['image']
        
        return image, image_path, original_size
    
    def __len__(self):
        return len(self.image_paths)

# -----------------------------
# Helper Functions
# -----------------------------
def create_model(num_classes):
    """Create UNet++ model with SCSE blocks - MUST match training configuration exactly"""
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_depth=5,
        encoder_weights=ENCODER_WEIGHTS,
        decoder_use_batchnorm=True, 
        decoder_channels=(256, 128, 64, 32, 16),  # Match training config exactly
        decoder_attention_type="scse", 
        in_channels=3,
        classes=num_classes, 
        activation=None  # No activation - softmax applied in forward pass
    )
    return model

def get_preprocessing(preprocessing_fn):
    """Creates preprocessing pipeline EXACTLY matching training setup"""
    def _preprocess(img, **kwargs):
        # Convert to float32 and normalize - MUST match training exactly
        img = img.astype(np.float32) / 255.0
        img = preprocessing_fn(img)
        return img.astype(np.float32)
    return A.Compose([
        A.Lambda(name="preproc", image=_preprocess),
        ToTensorV2(transpose_mask=True)
    ])

def mask_to_color(mask):
    """Convert class prediction mask to RGB image for visualization"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Debug: Print unique values in mask
    unique_values = np.unique(mask)
    print(f"Unique values in prediction mask: {unique_values}")
    
    for class_idx, color in COLOR_MAP.items():
        pixels_of_class = np.sum(mask == class_idx)
        if pixels_of_class > 0:
            print(f"Found {pixels_of_class} pixels of class {CLASSES[class_idx]} (idx={class_idx})")
        color_mask[mask == class_idx] = color
    
    return color_mask

def create_overlay(image, mask, alpha=0.7):
    """Create an overlay of mask on image with transparency"""
    # Create color mask with brighter colors
    color_mask = mask_to_color(mask)
    # Increase brightness of the mask to make it more visible
    color_mask = cv2.convertScaleAbs(color_mask, alpha=1.2, beta=0)
    
    # Create more vibrant overlay
    overlay = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)
    # Enhance contrast slightly
    return cv2.convertScaleAbs(overlay, alpha=1.1, beta=0)

def compute_comprehensive_metrics(pred, target, num_classes, min_pixels=0):
    """
    Compute comprehensive evaluation metrics for each class:
    - IoU (Intersection over Union)
    - Precision (True Positives / (True Positives + False Positives))
    - Recall (True Positives / (True Positives + False Negatives))
    - F1 Score (2 * Precision * Recall / (Precision + Recall))
    - Dice Score (2 * Intersection / (Pred + Target))
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Initialize metric arrays
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    dice_per_class = []
    present_mask = []
    
    # Compute confusion matrix elements for each class
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        
        # True Positives, False Positives, False Negatives, True Negatives
        tp = np.logical_and(pred_c, target_c).sum()
        fp = np.logical_and(pred_c, ~target_c).sum()
        fn = np.logical_and(~pred_c, target_c).sum()
        tn = np.logical_and(~pred_c, ~target_c).sum()
        
        # IoU (Intersection over Union)
        union = tp + fp + fn
        if union == 0:
            iou = 0.0 if target_c.sum() > 0 else float('nan')
        else:
            iou = tp / union
        
        # Precision
        if tp + fp == 0:
            precision = 0.0 if tp == 0 else float('nan')
        else:
            precision = tp / (tp + fp)
        
        # Recall (Sensitivity)
        if tp + fn == 0:
            recall = 0.0 if tp == 0 else float('nan')
        else:
            recall = tp / (tp + fn)
        
        # F1 Score
        if precision + recall == 0 or np.isnan(precision) or np.isnan(recall):
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        # Dice Score (similar to F1 but calculated differently)
        if tp + tp + fp + fn == 0:
            dice = 0.0 if target_c.sum() > 0 else float('nan')
        else:
            dice = 2 * tp / (2 * tp + fp + fn)
        
        # Store metrics
        iou_per_class.append(iou)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        dice_per_class.append(dice)
        present_mask.append(target_c.sum() >= min_pixels)
    
    # Calculate mean metrics (excluding NaN values)
    def safe_mean(values):
        valid_values = [v for v in values if not np.isnan(v)]
        return float(np.mean(valid_values)) if valid_values else 0.0
    
    def safe_mean_present(values, present):
        valid_values = [v for v, p in zip(values, present) if p and not np.isnan(v)]
        return float(np.mean(valid_values)) if valid_values else 0.0
    
    # Overall metrics
    miou_all = safe_mean(iou_per_class)
    miou_present = safe_mean_present(iou_per_class, present_mask)
    
    mprecision_all = safe_mean(precision_per_class)
    mprecision_present = safe_mean_present(precision_per_class, present_mask)
    
    mrecall_all = safe_mean(recall_per_class)
    mrecall_present = safe_mean_present(recall_per_class, present_mask)
    
    mf1_all = safe_mean(f1_per_class)
    mf1_present = safe_mean_present(f1_per_class, present_mask)
    
    mdice_all = safe_mean(dice_per_class)
    mdice_present = safe_mean_present(dice_per_class, present_mask)
    
    return {
        'iou_per_class': iou_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'dice_per_class': dice_per_class,
        'present_mask': present_mask,
        'miou_all': miou_all,
        'miou_present': miou_present,
        'mprecision_all': mprecision_all,
        'mprecision_present': mprecision_present,
        'mrecall_all': mrecall_all,
        'mrecall_present': mrecall_present,
        'mf1_all': mf1_all,
        'mf1_present': mf1_present,
        'mdice_all': mdice_all,
        'mdice_present': mdice_present
    }

def evaluate_predictions(pred_dir, gt_dir, num_classes=NUM_CLASSES, class_names=CLASSES, dataset_type=None, gt_structure=None):
    """
    Comprehensive evaluation of predictions against ground truth masks.
    Uses dataset-specific ground truth loading patterns from training script.
    """
    if not os.path.exists(pred_dir):
        print("Error: Prediction directory not found")
        return None
    
    if not gt_dir or not os.path.exists(gt_dir):
        print("Error: Ground truth directory not found or not specified")
        return None
    
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_mask.png')]
    if not pred_files:
        print("Error: No prediction mask files found (_mask.png)")
        return None
    
    # Initialize accumulators for all metrics
    total_metrics = {
        'iou_per_class': np.zeros(num_classes),
        'precision_per_class': np.zeros(num_classes),
        'recall_per_class': np.zeros(num_classes),
        'f1_per_class': np.zeros(num_classes),
        'dice_per_class': np.zeros(num_classes),
        'valid_samples_per_class': np.zeros(num_classes)  # Track valid samples per class
    }
    
    total_samples = 0
    valid_pairs = []
    failed_pairs = []
    
    print(f"\nEvaluating {len(pred_files)} predictions using {dataset_type} dataset patterns...")
    
    for pred_file in tqdm(pred_files, desc="Evaluating predictions"):
        # Create corresponding image filename
        base_name = pred_file.replace('_mask.png', '')
        
        # Create hypothetical image path to find GT (we need the original image path pattern)
        image_extensions = ['.jpg', '.jpeg', '.png']  
        image_path = None
        for ext in image_extensions:
            potential_image = f"{base_name}{ext}"
            if dataset_type and gt_structure:
                # Use dataset-specific finding
                gt_file_path = find_ground_truth_file(potential_image, gt_dir, dataset_type, gt_structure)
                if gt_file_path:
                    image_path = potential_image
                    break
        
        # Fallback to generic pattern matching if dataset-specific fails
        if not image_path:
            gt_candidates = [
                f"{base_name}.png",
                f"{base_name}_mask.png", 
                f"{base_name}.jpg___fuse.png",
                f"{base_name}_segmentation.png"
            ]
            
            gt_file_path = None
            for candidate in gt_candidates:
                potential_path = os.path.join(gt_dir, candidate)
                if os.path.exists(potential_path):
                    gt_file_path = potential_path
                    break
        else:
            gt_file_path = find_ground_truth_file(image_path, gt_dir, dataset_type, gt_structure)
        
        if not gt_file_path or not os.path.exists(gt_file_path):
            failed_pairs.append((pred_file, "No corresponding GT found"))
            continue
        
        # Load prediction and ground truth
        pred_path = os.path.join(pred_dir, pred_file)
        
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_file_path)
        
        if pred_mask is None or gt_mask is None:
            failed_pairs.append((pred_file, f"Failed to load: pred={pred_mask is not None}, gt={gt_mask is not None}"))
            continue
        
        # Convert ground truth using dataset-specific method
        if dataset_type and gt_structure:
            conversion_func = gt_structure.get("conversion_function")
            color_format = gt_structure.get("color_format", "rgb")
            
            if conversion_func == "gray_to_class_indices":
                # Automine dataset - grayscale masks
                if len(gt_mask.shape) == 3:
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
                gt_mask = gray_to_class_indices(gt_mask)
                
            elif conversion_func == "rgb_to_class_indices":
                # PotholeMix and R2S100K datasets - RGB masks
                if len(gt_mask.shape) == 3:
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
                    gt_mask = rgb_to_class_indices(gt_mask)
                else:
                    # Already grayscale, assume class indices
                    pass
                    
            elif conversion_func == "remap_rtk_mask":
                # RTK dataset - specific remapping
                if len(gt_mask.shape) == 3:
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
                gt_mask = remap_rtk_mask(gt_mask)
            else:
                # Generic fallback
                if len(gt_mask.shape) == 3:
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
                    gt_mask = rgb_to_class_indices(gt_mask)
                else:
                    gt_mask = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Generic conversion (original logic)
            if len(gt_mask.shape) == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
                gt_mask = rgb_to_class_indices(gt_mask)
            else:
                gt_mask = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure same dimensions
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Compute comprehensive metrics for this image
        metrics = compute_comprehensive_metrics(pred_mask, gt_mask, num_classes)
        
        # Accumulate valid metrics (excluding NaN values)
        for i in range(num_classes):
            if not np.isnan(metrics['iou_per_class'][i]):
                total_metrics['iou_per_class'][i] += metrics['iou_per_class'][i]
                total_metrics['valid_samples_per_class'][i] += 1
            
            if not np.isnan(metrics['precision_per_class'][i]):
                total_metrics['precision_per_class'][i] += metrics['precision_per_class'][i]
            
            if not np.isnan(metrics['recall_per_class'][i]):
                total_metrics['recall_per_class'][i] += metrics['recall_per_class'][i]
            
            if not np.isnan(metrics['f1_per_class'][i]):
                total_metrics['f1_per_class'][i] += metrics['f1_per_class'][i]
            
            if not np.isnan(metrics['dice_per_class'][i]):
                total_metrics['dice_per_class'][i] += metrics['dice_per_class'][i]
        
        valid_pairs.append((pred_file, os.path.basename(gt_file_path)))
        total_samples += 1
    
    if total_samples == 0:
        print("Error: No valid prediction-ground truth pairs found")
        if failed_pairs:
            print("Failed pairs:")
            for pred_file, reason in failed_pairs[:10]:  # Show first 10 failures
                print(f"  {pred_file}: {reason}")
        return None
    
    # Report failed pairs
    if failed_pairs:
        print(f"\nWarning: {len(failed_pairs)} pairs failed to load:")
        for pred_file, reason in failed_pairs[:5]:  # Show first 5 failures
            print(f"  {pred_file}: {reason}")
        if len(failed_pairs) > 5:
            print(f"  ... and {len(failed_pairs) - 5} more")
    
    # Compute average metrics per class
    avg_metrics = {}
    for metric_name in ['iou_per_class', 'precision_per_class', 'recall_per_class', 'f1_per_class', 'dice_per_class']:
        avg_metrics[metric_name] = np.zeros(num_classes)
        for i in range(num_classes):
            if total_metrics['valid_samples_per_class'][i] > 0:
                avg_metrics[metric_name][i] = total_metrics[metric_name][i] / total_metrics['valid_samples_per_class'][i]
            else:
                avg_metrics[metric_name][i] = 0.0
    
    # Compute overall mean metrics
    def safe_mean(values):
        valid_values = [v for v in values if v > 0]  # Exclude zero values (classes not present)
        return np.mean(valid_values) if valid_values else 0.0
    
    mean_metrics = {
        'mean_iou': safe_mean(avg_metrics['iou_per_class']),
        'mean_precision': safe_mean(avg_metrics['precision_per_class']),
        'mean_recall': safe_mean(avg_metrics['recall_per_class']),
        'mean_f1': safe_mean(avg_metrics['f1_per_class']),
        'mean_dice': safe_mean(avg_metrics['dice_per_class'])
    }
    
    # Print comprehensive results
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION RESULTS ({total_samples} images)")
    if dataset_type:
        print(f"Dataset Type: {dataset_type.upper()}")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Class':<18} {'IoU':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Dice':<8} {'Samples':<8}")
    print(f"{'-'*80}")
    
    # Per-class results
    for i, class_name in enumerate(class_names):
        iou = avg_metrics['iou_per_class'][i]
        precision = avg_metrics['precision_per_class'][i]
        recall = avg_metrics['recall_per_class'][i]
        f1 = avg_metrics['f1_per_class'][i]
        dice = avg_metrics['dice_per_class'][i]
        samples = int(total_metrics['valid_samples_per_class'][i])
        
        print(f"{class_name:<18} {iou:<8.4f} {precision:<10.4f} {recall:<8.4f} {f1:<10.4f} {dice:<8.4f} {samples:<8}")
    
    print(f"{'-'*80}")
    
    # Overall means
    print(f"{'MEAN (all classes)':<18} {mean_metrics['mean_iou']:<8.4f} {mean_metrics['mean_precision']:<10.4f} {mean_metrics['mean_recall']:<8.4f} {mean_metrics['mean_f1']:<10.4f} {mean_metrics['mean_dice']:<8.4f} {total_samples:<8}")
    
    print(f"{'='*80}")
    
    # Additional statistics
    print(f"\nDataset-Specific Information:")
    if dataset_type:
        print(f"- Dataset type: {dataset_type}")
        print(f"- Ground truth directory: {gt_dir}")
        if gt_structure:
            conversion_func = gt_structure.get("conversion_function", "Unknown")
            color_format = gt_structure.get("color_format", "Unknown")
            print(f"- Conversion method: {conversion_func}")
            print(f"- Color format: {color_format}")
    
    print(f"\nStatistics:")
    print(f"- Total test images processed: {total_samples}")
    print(f"- Valid prediction-GT pairs: {len(valid_pairs)}")
    print(f"- Failed pairs: {len(failed_pairs)}")
    
    # Class presence analysis
    print(f"\nClass Presence Analysis:")
    for i, class_name in enumerate(class_names):
        presence_count = int(total_metrics['valid_samples_per_class'][i])
        presence_pct = (presence_count / total_samples * 100) if total_samples > 0 else 0
        print(f"- {class_name}: Present in {presence_count}/{total_samples} images ({presence_pct:.1f}%)")
    
    # Performance insights
    print(f"\nPerformance Insights:")
    
    # Best performing classes
    non_zero_f1 = [(i, f1) for i, f1 in enumerate(avg_metrics['f1_per_class']) if f1 > 0]
    if non_zero_f1:
        best_class_idx, best_f1 = max(non_zero_f1, key=lambda x: x[1])
        worst_class_idx, worst_f1 = min(non_zero_f1, key=lambda x: x[1])
        
        print(f"- Best performing class: {class_names[best_class_idx]} (F1: {best_f1:.4f})")
        print(f"- Worst performing class: {class_names[worst_class_idx]} (F1: {worst_f1:.4f})")
    
    # Return comprehensive results
    return {
        'per_class_metrics': avg_metrics,
        'mean_metrics': mean_metrics,
        'class_presence': total_metrics['valid_samples_per_class'],
        'total_samples': total_samples,
        'valid_pairs': valid_pairs,
        'failed_pairs': failed_pairs,
        'class_names': class_names,
        'dataset_type': dataset_type
    }

def save_detailed_results(results, output_dir):
    """
    Save detailed evaluation results to CSV files for further analysis.
    """
    import pandas as pd
    
    if results is None:
        return
    
    # Create results directory
    results_dir = os.path.join(output_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Per-class metrics summary
    per_class_data = {
        'Class': results['class_names'],
        'IoU': results['per_class_metrics']['iou_per_class'],
        'Precision': results['per_class_metrics']['precision_per_class'],
        'Recall': results['per_class_metrics']['recall_per_class'],
        'F1_Score': results['per_class_metrics']['f1_per_class'],
        'Dice_Score': results['per_class_metrics']['dice_per_class'],
        'Samples_Count': results['class_presence'],
        'Presence_Percentage': [count/results['total_samples']*100 for count in results['class_presence']]
    }
    
    per_class_df = pd.DataFrame(per_class_data)
    per_class_csv = os.path.join(results_dir, "per_class_metrics.csv")
    per_class_df.to_csv(per_class_csv, index=False, float_format='%.4f')
    print(f"Per-class metrics saved to: {per_class_csv}")
    
    # 2. Overall metrics summary
    overall_data = {
        'Metric': ['Mean_IoU', 'Mean_Precision', 'Mean_Recall', 'Mean_F1', 'Mean_Dice'],
        'Value': [
            results['mean_metrics']['mean_iou'],
            results['mean_metrics']['mean_precision'],
            results['mean_metrics']['mean_recall'],
            results['mean_metrics']['mean_f1'],
            results['mean_metrics']['mean_dice']
        ]
    }
    
    overall_df = pd.DataFrame(overall_data)
    overall_csv = os.path.join(results_dir, "overall_metrics.csv")
    overall_df.to_csv(overall_csv, index=False, float_format='%.4f')
    print(f"Overall metrics saved to: {overall_csv}")
    
    # 3. Detailed configuration and metadata
    config_data = {
        'Parameter': [
            'Model_Path', 'Input_Directory', 'Ground_Truth_Directory',
            'Dataset_Type', 'Total_Images', 'Valid_Pairs', 'Failed_Pairs',
            'Image_Size', 'Encoder', 'Number_of_Classes', 'Device', 'Batch_Size'
        ],
        'Value': [
            MODEL_PATH, INPUT_DIR, GT_DIR if GT_DIR else 'N/A',
            results.get('dataset_type', 'Unknown'), results['total_samples'], 
            len(results['valid_pairs']), len(results.get('failed_pairs', [])),
            IMG_SIZE, ENCODER, NUM_CLASSES, DEVICE, BATCH_SIZE
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    config_csv = os.path.join(results_dir, "evaluation_config.csv")
    config_df.to_csv(config_csv, index=False)
    print(f"Evaluation configuration saved to: {config_csv}")
    
    # 4. Create a comprehensive summary report
    summary_path = os.path.join(results_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROAD DEFECT SEGMENTATION - EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: UNet++ with {ENCODER} encoder\n")
        f.write(f"Test Images: {results['total_samples']}\n")
        f.write(f"Classes: {', '.join(results['class_names'])}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"- Mean IoU: {results['mean_metrics']['mean_iou']:.4f}\n")
        f.write(f"- Mean F1-Score: {results['mean_metrics']['mean_f1']:.4f}\n")
        f.write(f"- Mean Precision: {results['mean_metrics']['mean_precision']:.4f}\n")
        f.write(f"- Mean Recall: {results['mean_metrics']['mean_recall']:.4f}\n")
        f.write(f"- Mean Dice Score: {results['mean_metrics']['mean_dice']:.4f}\n\n")
        
        f.write("PER-CLASS PERFORMANCE:\n")
        for i, class_name in enumerate(results['class_names']):
            f.write(f"\n{class_name}:\n")
            f.write(f"  - IoU: {results['per_class_metrics']['iou_per_class'][i]:.4f}\n")
            f.write(f"  - F1-Score: {results['per_class_metrics']['f1_per_class'][i]:.4f}\n")
            f.write(f"  - Precision: {results['per_class_metrics']['precision_per_class'][i]:.4f}\n")
            f.write(f"  - Recall: {results['per_class_metrics']['recall_per_class'][i]:.4f}\n")
            f.write(f"  - Present in: {int(results['class_presence'][i])}/{results['total_samples']} images\n")
        
        # Performance insights
        non_zero_f1 = [(i, f1) for i, f1 in enumerate(results['per_class_metrics']['f1_per_class']) if f1 > 0]
        if non_zero_f1:
            best_class_idx, best_f1 = max(non_zero_f1, key=lambda x: x[1])
            worst_class_idx, worst_f1 = min(non_zero_f1, key=lambda x: x[1])
            
            f.write(f"\nPERFORMANCE INSIGHTS:\n")
            f.write(f"- Best performing class: {results['class_names'][best_class_idx]} (F1: {best_f1:.4f})\n")
            f.write(f"- Worst performing class: {results['class_names'][worst_class_idx]} (F1: {worst_f1:.4f})\n")
    
    print(f"Summary report saved to: {summary_path}")
    print(f"\nAll evaluation results saved to: {results_dir}")

def run_inference():
    """Run inference on all images in the given directory with option to save masks for evaluation"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create subdirectories
    if SAVE_PROBABILITY_MAPS:
        prob_dir = os.path.join(OUTPUT_DIR, "probability_maps")
        os.makedirs(prob_dir, exist_ok=True)
    
    if SAVE_MASKS:
        mask_dir = os.path.join(OUTPUT_DIR, "masks")
        os.makedirs(mask_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg')
    image_paths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(image_paths)} images for inference")
    
    # Load the model
    model = create_model(NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle both direct state_dict and checkpoint formats
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        print(f"Loaded checkpoint from epoch {state_dict.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully on {DEVICE}")
    
    # Prepare transforms and preprocessing
    transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE)
    ])
    
    preprocessing_fn = get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths, transform=transform, preprocessing=preprocessing)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            images, paths, original_sizes = batch
            
            # Move images to device
            images = images.to(DEVICE)
            
            # Forward pass - apply mixed precision if available
            if torch.cuda.is_available() and hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)
                
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Process each image in the batch
            for i, (pred, path, size) in enumerate(zip(preds, paths, original_sizes)):
                # Create output filename FIRST
                filename = os.path.basename(path)
                base_name = os.path.splitext(filename)[0]
                
                # Get original image dimensions - ensure we get only the height and width
                size_np = size.numpy()
                
                try:
                    if len(size_np.shape) > 0:  # It's an array or tensor
                        h, w = size_np[0], size_np[1]
                    else:  # It's a tuple
                        h, w = size_np
                    
                    # Ensure dimensions are valid
                    h, w = int(h), int(w)
                    if h <= 0 or w <= 0:
                        print(f"Warning: Invalid dimensions for image {path}: h={h}, w={w}")
                        # Load original image to get dimensions as fallback
                        orig_img = cv2.imread(path)
                        h, w = orig_img.shape[:2]
                except Exception as e:
                    print(f"Error processing dimensions for {path}: {e}")
                    print(f"Size tensor shape: {size_np.shape}, content: {size_np}")
                    # Load original image to get dimensions as fallback
                    orig_img = cv2.imread(path)
                    h, w = orig_img.shape[:2]
                
                # Resize prediction to original size
                pred_resized = cv2.resize(pred, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
                
                # Debug: Print class distribution
                unique, counts = np.unique(pred_resized, return_counts=True)
                total_pixels = pred_resized.size
                print(f"\nClass distribution for {filename}:")
                for cls_idx, count in zip(unique, counts):
                    if cls_idx < NUM_CLASSES:
                        percentage = (count / total_pixels) * 100
                        print(f"  Class {CLASSES[cls_idx]}: {percentage:.2f}%")
                
                # Save prediction mask for evaluation if requested
                if SAVE_MASKS:
                    mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
                    cv2.imwrite(mask_path, pred_resized.astype(np.uint8))
                    print(f"  Saved mask: {mask_path}")
                
                # Create and save overlay
                if SAVE_OVERLAYS:
                    # Load original image for overlay
                    original_image = cv2.imread(path)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    
                    # Create overlay
                    overlay_image = create_overlay(original_image, pred_resized)
                    overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")
                    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
                    print(f"  Saved overlay: {overlay_path}")
                
                # Save probability maps if requested
                if SAVE_PROBABILITY_MAPS:
                    # Get probabilities for each class
                    prob_maps = probs[i].cpu().numpy()
                    
                    # Create a directory for this image's probability maps
                    img_prob_dir = os.path.join(prob_dir, base_name)
                    os.makedirs(img_prob_dir, exist_ok=True)
                    
                    # Save probability map for each class
                    for cls_idx, cls_name in enumerate(CLASSES):
                        # Resize to original dimensions
                        prob_map = prob_maps[cls_idx]
                        prob_map_resized = cv2.resize(prob_map, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
                        
                        # Convert to heatmap for visualization
                        heatmap = (prob_map_resized * 255).astype(np.uint8)
                        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        
                        # Save the heatmap
                        cv2.imwrite(os.path.join(img_prob_dir, f"{cls_name}.png"), heatmap_colored)
    
    print(f"\nInference complete! Results saved to {OUTPUT_DIR}")
    
    # Run evaluation if requested and masks were saved
    if EVALUATE and SAVE_MASKS and GT_DIR:
        mask_dir = os.path.join(OUTPUT_DIR, "masks")
        
        # Auto-detect dataset type for proper ground truth handling
        detected_type, detected_gt_dir, gt_structure = detect_dataset_type_and_gt_dir(INPUT_DIR)
        
        # Use detected ground truth directory if not already set
        eval_gt_dir = GT_DIR
        if not eval_gt_dir and detected_gt_dir:
            eval_gt_dir = detected_gt_dir
            print(f"Using auto-detected ground truth directory: {eval_gt_dir}")
        
        # Run evaluation with dataset-specific handling
        results = evaluate_predictions(
            mask_dir, 
            eval_gt_dir, 
            NUM_CLASSES, 
            CLASSES, 
            dataset_type=detected_type, 
            gt_structure=gt_structure
        )
        
        # Save detailed results to CSV files
        if results is not None:
            save_detailed_results(results, OUTPUT_DIR)
        
        return results
    elif EVALUATE and not SAVE_MASKS:
        print("\nNote: Evaluation requires SAVE_MASKS to be enabled")
    elif EVALUATE and not GT_DIR:
        print("\nNote: Evaluation requires GT_DIR to be specified or auto-detectable")
    
    return None

def detect_dataset_type_and_gt_dir(input_dir):
    """
    Detect dataset type and corresponding ground truth directory based on training script patterns.
    Returns (dataset_type, gt_dir, gt_structure_info)
    """
    input_dir = os.path.abspath(input_dir)
    
    # Try to match against known patterns
    for pattern, gt_dir in GT_DIR_MAPPINGS.items():
        if pattern in input_dir.lower():
            if os.path.exists(gt_dir):
                dataset_type = pattern.split('/')[0]  # Extract dataset name
                print(f"Auto-detected dataset: {dataset_type}")
                print(f"Ground truth directory: {gt_dir}")
                return dataset_type, gt_dir, _get_gt_structure_info(dataset_type)
    
    # Manual detection based on directory structure
    parent_dir = os.path.dirname(input_dir)
    dir_name = os.path.basename(input_dir)
    
    # Check for Automine pattern (same directory with _mask.png files)
    if any(f.endswith('_mask.png') for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))):
        return "automine", input_dir, _get_gt_structure_info("automine")
    
    # Check for PotholeMix pattern (subdirectories with images/masks structure)
    pothole_subdirs = ['cnr-road-dataset', 'cracks-and-potholes-in-road', 'crack500', 'pothole600', 'gaps384', 'edmcrack600']
    if any(os.path.exists(os.path.join(input_dir, subdir, 'images')) for subdir in pothole_subdirs):
        return "pothole_mix", input_dir, _get_gt_structure_info("pothole_mix")
    
    # Check for RTK pattern (images/ and masks/ directories)
    if (os.path.exists(os.path.join(parent_dir, 'images')) and 
        os.path.exists(os.path.join(parent_dir, 'masks'))):
        return "rtk", parent_dir, _get_gt_structure_info("rtk")
    
    # Check for R2S100K pattern (separate labels directory)
    if 'r2s100k' in input_dir.lower():
        labels_dir = input_dir.replace('images', 'labels').replace('test', 'test-labels').replace('val', 'val-labels')
        if os.path.exists(labels_dir):
            return "r2s100k", labels_dir, _get_gt_structure_info("r2s100k")
    
    print(f"Warning: Could not auto-detect dataset type for {input_dir}")
    return "unknown", None, None

def _get_gt_structure_info(dataset_type):
    """Get ground truth file structure information for each dataset type"""
    structures = {
        "automine": {
            "mask_suffix": "_mask.png",
            "same_directory": True,
            "color_format": "grayscale",
            "conversion_function": "gray_to_class_indices"
        },
        "pothole_mix": {
            "subdirectories": ['cnr-road-dataset', 'cracks-and-potholes-in-road', 'crack500', 'pothole600', 'gaps384', 'edmcrack600'],
            "mask_dir": "masks",
            "image_dir": "images", 
            "same_directory": False,
            "color_format": "rgb",
            "conversion_function": "rgb_to_class_indices"
        },
        "rtk": {
            "mask_dir": "masks",
            "image_dir": "images",
            "same_directory": False,
            "color_format": "grayscale",
            "conversion_function": "remap_rtk_mask"
        },
        "r2s100k": {
            "mask_candidates": [
                "{base_name}.png",
                "{base_name}_mask.png", 
                "{base_name}_segmentation.png",
                "{base_name}.jpg___fuse.png"
            ],
            "same_directory": False,
            "color_format": "rgb",
            "conversion_function": "rgb_to_class_indices"
        }
    }
    return structures.get(dataset_type, {})

def find_ground_truth_file(image_path, gt_dir, dataset_type, gt_structure):
    """
    Find corresponding ground truth file for a given image path using dataset-specific patterns.
    Exactly matches the patterns used in the training script.
    """
    if not gt_dir or not os.path.exists(gt_dir):
        return None
    
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    
    if dataset_type == "automine":
        # Automine: same directory, _mask.png suffix
        mask_file = f"{base_name}_mask.png"
        mask_path = os.path.join(gt_dir, mask_file)
        return mask_path if os.path.exists(mask_path) else None
    
    elif dataset_type == "pothole_mix":
        # PotholeMix: subdirectories with images/masks structure
        for subdir in gt_structure.get("subdirectories", []):
            mask_dir = os.path.join(gt_dir, subdir, "masks")
            if os.path.exists(mask_dir):
                mask_file = f"{base_name}.png"  # Most masks are PNG
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    return mask_path
        return None
    
    elif dataset_type == "rtk":
        # RTK: masks/ directory with various naming patterns
        mask_dir = os.path.join(gt_dir, "masks") if gt_structure.get("mask_dir") else gt_dir
        if os.path.exists(mask_dir):
            # Try various mask file patterns
            mask_candidates = [
                f"{base_name}.png",
                f"{base_name}_mask.png",
                f"{base_name}.jpg",
                f"{base_name}.jpeg"
            ]
            for mask_file in mask_candidates:
                mask_path = os.path.join(mask_dir, mask_file)
                if os.path.exists(mask_path):
                    return mask_path
        return None
    
    elif dataset_type == "r2s100k":
        # R2S100K: various naming patterns as defined in training script
        mask_candidates = [
            f"{base_name}.png",
            f"{base_name}_mask.png", 
            f"{base_name}_segmentation.png",
            f"{base_name}.jpg___fuse.png"
        ]
        for mask_file in mask_candidates:
            mask_path = os.path.join(gt_dir, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        return None
    
    return None

def remap_rtk_mask(mask):
    """Remap RTK mask indices to model classes - EXACT COPY from training script"""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    RTK_TO_MODEL = {
        11: 1,  # pothole → pothole
        12: 2,  # craks → crack  
        10: 3,  # waterPuddle → puddle
        9: 4,   # patchs → distressed_patch
        # everything else → background (0)
    }
    for orig_idx, model_idx in RTK_TO_MODEL.items():
        new_mask[mask == orig_idx] = model_idx
    return new_mask

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("Road Defect Segmentation Inference Script")
    print("="*60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Save Overlays: {SAVE_OVERLAYS}")
    print(f"Save Probability Maps: {SAVE_PROBABILITY_MAPS}")
    print(f"Save Masks: {SAVE_MASKS}")
    print(f"Evaluate: {EVALUATE}")
    print(f"Classes: {CLASSES}")
    print("="*60)
    
    # Auto-detect dataset type and ground truth directory
    detected_type, detected_gt_dir, gt_structure = detect_dataset_type_and_gt_dir(INPUT_DIR)
    
    if GT_DIR is None:
        if detected_gt_dir:
            GT_DIR = detected_gt_dir
            print(f"✓ Auto-detected ground truth directory: {GT_DIR}")
        else:
            print("⚠ Warning: Ground truth directory could not be auto-detected")
    else:
        print(f"✓ Using specified ground truth directory: {GT_DIR}")
    
    if detected_type:
        print(f"✓ Dataset type: {detected_type.upper()}")
        if gt_structure:
            conversion_func = gt_structure.get("conversion_function", "Unknown")
            color_format = gt_structure.get("color_format", "Unknown")
            print(f"✓ Ground truth format: {color_format} -> {conversion_func}")
    else:
        print("⚠ Warning: Dataset type could not be determined")
    
    print("="*60)
    
    results = run_inference()
    
    if results:
        print(f"\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60)
        print(f"Mean IoU: {results['mean_metrics']['mean_iou']:.4f}")
        print(f"Mean F1-Score: {results['mean_metrics']['mean_f1']:.4f}")
        print(f"Mean Precision: {results['mean_metrics']['mean_precision']:.4f}")
        print(f"Mean Recall: {results['mean_metrics']['mean_recall']:.4f}")
        print(f"Mean Dice Score: {results['mean_metrics']['mean_dice']:.4f}")
        print(f"Evaluated on {results['total_samples']} images")
        if 'failed_pairs' in results and results['failed_pairs']:
            print(f"Failed pairs: {len(results['failed_pairs'])}")
        print("="*60)
        print("Detailed results have been saved to CSV files in the results directory.")
        
        # Save detailed results to CSV
        save_detailed_results(results, OUTPUT_DIR)
