#!/usr/bin/env python
"""
Metrics computation functions for road defect segmentation evaluation.
"""

import torch
import numpy as np


def compute_iou(pred, target, num_classes: int, min_pixels: int = 0):
    """
    Compute IoU metrics per class and overall.
    
    Args:
        pred : Tensor  (N,H,W)   predicted class indices
        target : Tensor (N,H,W)   ground-truth class indices
        num_classes : int
        min_pixels : int - classes with GT pixel-count < min_pixels are ignored in mean
    
    Returns:
        iou_per_class : list[float]  length == num_classes
        miou_all      : float        mean over *all* classes
        miou_present  : float        mean over classes that satisfy the pixel threshold
    """
    # Flatten to 1-D for easy logical ops
    pred = pred.view(-1)
    target = target.view(-1)

    iou_per_class = []
    present_mask = []

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        # Avoid div-by-zero; IoU is 0 when union==0 and class absent in GT
        iou = (inter / (union + 1e-6)).item() if union > 0 else 0.0
        iou_per_class.append(iou)

        # "present" means GT pixels ≥ min_pixels
        present_mask.append(target_c.sum().item() >= min_pixels)

    miou_all = float(np.mean(iou_per_class))
    miou_present = float(np.mean([iou for iou, present in zip(iou_per_class, present_mask) if present]))

    return iou_per_class, miou_all, miou_present


def compute_comprehensive_metrics(pred, target, num_classes: int, min_pixels: int = 0):
    """
    Compute comprehensive segmentation metrics including IoU, F1, Precision, Recall, and Dice.
    
    Args:
        pred : Tensor  (N,H,W)   predicted class indices
        target : Tensor (N,H,W)   ground-truth class indices
        num_classes : int
        min_pixels : int
    
    Returns:
        dict: Comprehensive metrics including IoU, F1, Precision, Recall, Dice
    """
    # Flatten to 1-D for easy logical ops
    pred = pred.view(-1)
    target = target.view(-1)
    
    metrics = {
        'iou_per_class': [],
        'f1_per_class': [],
        'precision_per_class': [],
        'recall_per_class': [],
        'dice_per_class': [],
        'present_mask': []
    }
    
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        
        # True/False positives/negatives
        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        tn = (~pred_c & ~target_c).sum().float()
        
        # IoU calculation
        union = (pred_c | target_c).sum().float()
        iou = (tp / (union + 1e-6)).item() if union > 0 else 0.0
        
        # Precision, Recall, F1
        precision = (tp / (tp + fp + 1e-6)).item()
        recall = (tp / (tp + fn + 1e-6)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-6)) if (precision + recall) > 0 else 0.0
        
        # Dice coefficient
        dice = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
        
        metrics['iou_per_class'].append(iou)
        metrics['f1_per_class'].append(f1)
        metrics['precision_per_class'].append(precision)
        metrics['recall_per_class'].append(recall)
        metrics['dice_per_class'].append(dice)
        
        # "present" means GT pixels ≥ min_pixels
        present = target_c.sum().item() >= min_pixels
        metrics['present_mask'].append(present)
    
    # Aggregate metrics
    metrics['miou_all'] = float(np.mean(metrics['iou_per_class']))
    metrics['miou_present'] = float(np.mean([iou for iou, present in 
                                           zip(metrics['iou_per_class'], metrics['present_mask']) if present]))
    
    metrics['mf1_all'] = float(np.mean(metrics['f1_per_class']))
    metrics['mf1_present'] = float(np.mean([f1 for f1, present in 
                                          zip(metrics['f1_per_class'], metrics['present_mask']) if present]))
    
    metrics['mprecision_present'] = float(np.mean([prec for prec, present in 
                                                  zip(metrics['precision_per_class'], metrics['present_mask']) if present]))
    
    metrics['mrecall_present'] = float(np.mean([rec for rec, present in 
                                               zip(metrics['recall_per_class'], metrics['present_mask']) if present]))
    
    metrics['mdice_present'] = float(np.mean([dice for dice, present in 
                                            zip(metrics['dice_per_class'], metrics['present_mask']) if present]))
    
    # mAP-like metric: average precision across classes weighted by presence
    present_classes = sum(metrics['present_mask'])
    if present_classes > 0:
        metrics['map_present'] = float(np.mean([prec for prec, present in 
                                              zip(metrics['precision_per_class'], metrics['present_mask']) if present]))
    else:
        metrics['map_present'] = 0.0
    
    return metrics


def compute_binary_defect_metrics(pred, target):
    """
    Compute binary metrics for defect (any class >0) vs. no defect (class 0).
    
    Args:
        pred: Tensor (N,H,W) or (H,W) predicted class indices
        target: Tensor (N,H,W) or (H,W) ground-truth class indices
    
    Returns:
        dict: binary_iou, binary_f1, binary_precision, binary_recall, tp, fp, fn, tn
    """
    # Use .reshape(-1) for compatibility with both numpy and torch
    pred_bin = (pred > 0).reshape(-1)
    target_bin = (target > 0).reshape(-1)
    tp = (pred_bin & target_bin).sum().item() if hasattr(pred_bin, 'sum') else np.sum(pred_bin & target_bin)
    fp = (pred_bin & ~target_bin).sum().item() if hasattr(pred_bin, 'sum') else np.sum(pred_bin & ~target_bin)
    fn = (~pred_bin & target_bin).sum().item() if hasattr(pred_bin, 'sum') else np.sum(~pred_bin & target_bin)
    tn = (~pred_bin & ~target_bin).sum().item() if hasattr(pred_bin, 'sum') else np.sum(~pred_bin & ~target_bin)
    union = (pred_bin | target_bin).sum().item() if hasattr(pred_bin, 'sum') else np.sum(pred_bin | target_bin)
    intersection = (pred_bin & target_bin).sum().item() if hasattr(pred_bin, 'sum') else np.sum(pred_bin & target_bin)
    binary_iou = intersection / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    binary_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'binary_iou': binary_iou,
        'binary_f1': binary_f1,
        'binary_precision': precision,
        'binary_recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def compute_class_weights(dataset, num_classes=6, method='inverse_freq'):
    """
    Calculate class weights based on pixel distribution in the dataset.
    
    Args:
        dataset: ConcatDataset containing all training data
        num_classes: Number of classes
        method: 'inverse_freq' or 'balanced'
    
    Returns:
        torch.Tensor: Class weights for loss function
    """
    print(f"\nCalculating class weights using {method} method...")
    
    # Count pixels per class
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_samples = 0
    
    # Sample a subset for efficiency (adjust if needed)
    sample_size = min(len(dataset), 500)  # Reduced to prevent memory issues
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        try:
            _, mask = dataset[idx]
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()
            
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if 0 <= cls < num_classes:
                    class_pixel_counts[cls] += count
            total_samples += 1
            
            # Explicit cleanup
            del mask
            
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue
    
    # Calculate weights based on method
    if method == 'inverse_freq':
        # Inverse frequency weighting
        class_frequencies = class_pixel_counts / (class_pixel_counts.sum() + 1e-8)
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (class_frequencies + 1e-6)
        # Normalize so background weight is reasonable
        weights = weights / weights[0] * 0.5
        
    elif method == 'balanced':
        # Balanced class weighting (sklearn style)
        n_samples = class_pixel_counts.sum()
        weights = n_samples / (num_classes * (class_pixel_counts + 1e-8))
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply more aggressive weighting for extremely rare classes
    class_frequencies = class_pixel_counts / (class_pixel_counts.sum() + 1e-8)
    for i, freq in enumerate(class_frequencies):
        if freq < 0.0001:  # Extremely rare classes (< 0.01%)
            weights[i] = 100.0
        elif freq < 0.001:  # Very rare classes (< 0.1%)
            weights[i] = 50.0
        elif freq < 0.01:  # Rare classes (< 1%)
            weights[i] = 20.0
    
    # Cap maximum weight to prevent extreme values
    weights = np.clip(weights, 0.1, 100.0)
    
    print(f"Sampled {total_samples} images from {len(dataset)} total")
    print("\nClass pixel distribution:")
    CLASSES = ['Background', 'Crack', 'Pothole', 'Rut', 'Patch', 'Other']
    for cls in range(num_classes):
        percentage = (class_pixel_counts[cls] / (class_pixel_counts.sum() + 1e-8)) * 100
        print(f"  {CLASSES[cls]:>16s}: {class_pixel_counts[cls]:>12,} pixels ({percentage:5.2f}%) → weight: {weights[cls]:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32)


class StratifiedBatchSampler:
    """
    Stratified sampler that ensures each batch contains both background and defect samples.
    This prevents all-background batches that cause training collapse.
    """
    def __init__(self, dataset, batch_size, min_defect_ratio=0.3, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_defect_ratio = min_defect_ratio
        self.shuffle = shuffle
        
        # Categorize samples into background-only and defect-containing
        self.background_indices = []
        self.defect_indices = []
        
        print("Analyzing dataset for stratified sampling...")
        
        # FIXED: Limit analysis to prevent memory issues with very large datasets
        max_samples_to_analyze = min(len(dataset), 5000)  # Prevent OOM on huge datasets
        
        # Import tqdm here to avoid import issues
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback if tqdm not available
            def tqdm(iterable, desc=""):
                return iterable
        
        for i in tqdm(range(max_samples_to_analyze), desc="Categorizing samples"):
            try:
                # FIXED: Handle both ConcatDataset and regular datasets
                # Use direct dataset access instead of get_mask_path to avoid Subset issues
                try:
                    _, mask = dataset[i]
                    if torch.is_tensor(mask):
                        mask = mask.detach().cpu().numpy()  # Ensure CPU and detached
                    
                    if np.any(mask > 0):
                        self.defect_indices.append(i)
                    else:
                        self.background_indices.append(i)
                    
                    # FIXED: Explicit cleanup
                    del mask
                except Exception as e:
                    print(f"Warning: Error analyzing sample {i}: {e}")
                    self.background_indices.append(i)
                    
            except Exception as e:
                print(f"Warning: Error analyzing sample {i}: {e}")
                self.background_indices.append(i)
        
        # FIXED: If we only analyzed a subset, extrapolate the ratio
        if max_samples_to_analyze < len(dataset):
            total_analyzed = len(self.defect_indices) + len(self.background_indices)
            if total_analyzed > 0:
                defect_ratio = len(self.defect_indices) / total_analyzed
                remaining_samples = len(dataset) - max_samples_to_analyze
                
                # Distribute remaining samples based on observed ratio
                estimated_defects = int(remaining_samples * defect_ratio)
                estimated_backgrounds = remaining_samples - estimated_defects
                
                # Add remaining indices (we'll check them dynamically during training)
                for i in range(max_samples_to_analyze, len(dataset)):
                    if len(self.defect_indices) < max_samples_to_analyze * defect_ratio + estimated_defects:
                        self.defect_indices.append(i)
                    else:
                        self.background_indices.append(i)
        
        print(f"Found {len(self.defect_indices)} defect samples and {len(self.background_indices)} background samples")
        
        # Ensure we have both types
        if len(self.defect_indices) == 0:
            raise ValueError("No defect samples found! Check your dataset.")
        if len(self.background_indices) == 0:
            print("Warning: No background-only samples found.")
            self.background_indices = self.defect_indices[:len(self.defect_indices)//2]

        self.min_defect_per_batch = max(1, int(batch_size * min_defect_ratio))
    
    def __iter__(self):
        """Iterator method for batch sampling"""
        if self.shuffle:
            np.random.shuffle(self.defect_indices)
            np.random.shuffle(self.background_indices)
        
        # FIXED: Use cycle-like behavior to handle cases where we have fewer samples than needed
        defect_cycle = iter(self.defect_indices)
        background_cycle = iter(self.background_indices)
        
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch = []
            
            # Add required defect samples
            for _ in range(self.min_defect_per_batch):
                try:
                    batch.append(next(defect_cycle))
                except StopIteration:
                    # Reset and shuffle if we run out
                    if self.shuffle:
                        np.random.shuffle(self.defect_indices)
                    defect_cycle = iter(self.defect_indices)
                    batch.append(next(defect_cycle))
            
            # Fill rest with background samples
            remaining = self.batch_size - len(batch)
            for _ in range(remaining):
                try:
                    batch.append(next(background_cycle))
                except StopIteration:
                    # Reset and shuffle if we run out
                    if self.shuffle:
                        np.random.shuffle(self.background_indices)
                    background_cycle = iter(self.background_indices)
                    batch.append(next(background_cycle))
            
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield batch
    
    def __len__(self):
        """FIXED: Add missing __len__ method for iterator protocol"""
        return len(self.dataset) // self.batch_size
