#!/usr/bin/env python3
"""
Robust Road+Defect Segmentation Cascade Evaluation Script

- Uses a road segmentation model (SegFormer) to mask road regions.
- Applies a defect segmentation model (UNet++) only on road regions using tiling.
- Aggregates predictions and reconstructs the full defect mask.
- Evaluates using the same metrics as test_simple.py: per-class IoU, binary IoU, binary F1, precision, recall.
- Saves visualizations and outputs a metrics summary JSON.
- Robust error handling, logging, and debug output.

Update the model paths and test directories as needed.
"""
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import segmentation_models_pytorch as smp
import json
import sys
import logging
from datetime import datetime
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
# Import comprehensive metrics from metrics.py
from metrics import compute_comprehensive_metrics, compute_binary_defect_metrics
# Import mask mapping function for consistency with training
try:
    from test_simple import remap_automine_mask
except ImportError:
    def remap_automine_mask(mask):
        return mask
# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SegFormer (road)
SEGFORMER_CHECKPOINT = "/home/cloli/experimentation/cascade-road-segmentation/src/utils/segformer/best_epoch26_besto.pth"
SEGFORMER_PRETRAIN   = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
TARGET_SIZE          = (512, 512)

# UNet++ (defects)
ENCODER          = "efficientnet-b5"
ENCODER_WEIGHTS  = "imagenet"
UNETPP_WEIGHTS   = "/home/cloli/experimentation/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt"
IMG_H, IMG_W     = 512, 512
CLASSES          = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
NUM_CLASSES      = len(CLASSES)

# Color mapping for visualization
DEFECT_COLOR_MAP = {
    0: (0, 0, 0),                # background: black
    1: (0, 0, 255),              # pothole: blue 
    2: (0, 255, 0),              # crack: green
    3: (140, 160, 222),          # puddle: light blue
    4: (119, 61, 128),           # distressed_patch: purple
    5: (112, 84, 62)             # mud: brown
}

# TILING
TILE  = 512
STRIDE = 256
MIN_COVER = 0.05

TEST_DIR   = "/home/cloli/experimentation/cascade-road-segmentation/src/data/automine/valid"
GT_DIR     = "/home/cloli/experimentation/cascade-road-segmentation/src/data/automine/valid"
OUTPUT_DIR = "./output_cascade_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Robust Logging Setup ---
def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- Robust Model Loading ---
def load_segformer(checkpoint_path: str, pretrained_name: str, device: torch.device):
    try:
        processor = SegformerImageProcessor.from_pretrained(pretrained_name)
        model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name, num_labels=2, ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        model.to(device).eval()
        logging.info(f"Loaded SegFormer from {checkpoint_path}")
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load SegFormer: {e}")
        raise

def load_unetpp(weights_path: str, encoder: str, encoder_weights: str, num_classes: int, device: torch.device):
    try:
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=True, 
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type="scse", 
            in_channels=3,
            classes=num_classes, 
            activation=None
        ).to(device).eval()
        # Robust weight loading as in test_simple.py
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        except TypeError:
            # For older torch versions or if weights_only is not supported
            state_dict = torch.load(weights_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        logging.info(f"Loaded UNet++ from {weights_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load UNet++: {e}")
        raise

# --- Preprocessing (from test_simple.py) ---
def get_preprocessing(preprocessing_fn):
    def _preprocess(img, **kwargs):
        return preprocessing_fn(img).astype(np.float32)
    return lambda image: _preprocess(image)

def mask_to_rgb(mask: np.ndarray):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in DEFECT_COLOR_MAP.items():
        rgb[mask == class_idx] = color
    return rgb

def sliding_windows(H, W, tile=TILE, stride=STRIDE):
    if H <= tile and W <= tile:
        yield 0, 0, W, H
        return
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            yield x, y, min(x + tile, W), min(y + tile, H)
    if W > tile:
        for y in range(0, H - tile + 1, stride):
            yield W - tile, y, W, y + tile
    if H > tile:
        for x in range(0, W - tile + 1, stride):
            yield x, H - tile, x + tile, H
    if W > tile and H > tile:
        yield W - tile, H - tile, W, H

def remap_to_binary_defect(mask: np.ndarray) -> np.ndarray:
    """
    Remap all defect classes (1,2,3,4,5) to 1, background (0) to 0.
    This is used for fair binary metrics.
    """
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    return binary_mask

# --- Main Cascade Evaluation Function ---
def run_cascade_evaluation():
    setup_logging(os.path.join(OUTPUT_DIR, 'cascade_eval.log'))
    logging.info("Starting cascade evaluation...")
    device = DEVICE
    segformer, processor = load_segformer(SEGFORMER_CHECKPOINT, SEGFORMER_PRETRAIN, device)
    unetpp = load_unetpp(UNETPP_WEIGHTS, ENCODER, ENCODER_WEIGHTS, NUM_CLASSES, device)
    preproc_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preproc_fn_helper = get_preprocessing(preproc_fn)
    results = []
    results_full = []
    total_iou = 0.0
    total_binary_iou = 0.0
    total_binary_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    class_ious = np.zeros(NUM_CLASSES)
    total_samples = 0
    all_images = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')))
    for img_path in tqdm(all_images, desc="Cascade Evaluation"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            orig_pil = Image.open(img_path).convert("RGB")
            W, H     = orig_pil.size
            orig_np  = np.array(orig_pil)
            inp = processor(images=orig_pil, return_tensors="pt", size=TARGET_SIZE)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                logits = segformer(**inp).logits
                road   = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            road_rgb = orig_np * road[:, :, None]

            # --- Cascade (road-masked) prediction ---
            probs_acc  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
            counts_acc = np.zeros((H, W), dtype=np.float32)
            confidence_acc = np.zeros((H, W), dtype=np.float32)  # Track max confidence per pixel

            for x1, y1, x2, y2 in sliding_windows(H, W):
                h_tile, w_tile = y2 - y1, x2 - x1
                if h_tile == 0 or w_tile == 0:
                    continue
                patch_mask = road[y1:y2, x1:x2]
                if patch_mask.size == 0 or patch_mask.mean() < MIN_COVER:
                    continue
                patch_rgb = road_rgb[y1:y2, x1:x2]
                if patch_rgb.size == 0 or np.sum(patch_rgb) == 0:
                    continue
                patch_rgb = patch_rgb.astype(np.uint8)
                patch_res = cv2.resize(patch_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                patch_res = preproc_fn_helper(patch_res)
                tensor = to_tensor(patch_res).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits2 = unetpp(tensor)
                    probs2  = F.softmax(logits2, dim=1)[0].cpu().numpy()
                
                # Resize probabilities back to patch size
                probs2 = np.stack([cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) for p in probs2])
                
                # Calculate max confidence for this patch
                max_confidence = np.max(probs2, axis=0)
                
                # Only include predictions where confidence > 0.6
                high_conf_mask = max_confidence > 0.6
                
                # Update accumulated probabilities only for high-confidence pixels
                probs_acc[:, y1:y2, x1:x2][:, high_conf_mask] += probs2[:, high_conf_mask]
                counts_acc[y1:y2, x1:x2][high_conf_mask] += 1
                confidence_acc[y1:y2, x1:x2] = np.maximum(confidence_acc[y1:y2, x1:x2], max_confidence)

            # Handle pixels with no predictions (low confidence everywhere)
            counts_acc[counts_acc == 0] = 1
            probs_acc /= counts_acc

            pred_full = probs_acc.argmax(0).astype(np.uint8)
            # Ensure no -1 values in predictions
            pred_full = np.clip(pred_full, 0, NUM_CLASSES-1)
            # Optional: Set low-confidence areas to background (class 0)
            low_conf_mask = confidence_acc < 0.6
            pred_full[low_conf_mask] = 0 
            # --- Full-image UNet++ prediction (no road mask) ---
            probs_acc_full  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
            counts_acc_full = np.zeros((H, W), dtype=np.float32)
            confidence_acc_full = np.zeros((H, W), dtype=np.float32)
            for x1, y1, x2, y2 in sliding_windows(H, W):
                h_tile, w_tile = y2 - y1, x2 - x1
                if h_tile == 0 or w_tile == 0:
                    continue
                patch_rgb = orig_np[y1:y2, x1:x2].astype(np.uint8)
                patch_res = cv2.resize(patch_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                patch_res = preproc_fn_helper(patch_res)
                tensor = to_tensor(patch_res).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits2 = unetpp(tensor)
                    probs2  = F.softmax(logits2, dim=1)[0].cpu().numpy()
                probs2 = np.stack([cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) for p in probs2])
                max_confidence = np.max(probs2, axis=0)
                high_conf_mask = max_confidence > 0.6
                probs_acc_full[:, y1:y2, x1:x2][:, high_conf_mask] += probs2[:, high_conf_mask]
                counts_acc_full[y1:y2, x1:x2][high_conf_mask] += 1
                confidence_acc_full[y1:y2, x1:x2] = np.maximum(confidence_acc_full[y1:y2, x1:x2], max_confidence)
            counts_acc_full[counts_acc_full == 0] = 1
            probs_acc_full /= counts_acc_full
            pred_fullimg = probs_acc_full.argmax(0).astype(np.uint8)
            pred_fullimg = np.clip(pred_fullimg, 0, NUM_CLASSES-1)
            low_conf_mask_full = confidence_acc_full < 0.6
            pred_fullimg[low_conf_mask_full] = 0
            # --- Combine cascade and full-image predictions (best of both inside road boundary) ---
            # Find outermost contour of the road mask
            road_mask = (road == 1).astype(np.uint8)
            contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            road_boundary_mask = np.zeros_like(road_mask)
            if contours:
                cv2.drawContours(road_boundary_mask, contours, -1, 1, thickness=cv2.FILLED)
            # For each pixel: if inside road boundary, take defect if either pred_full or pred_fullimg predicts defect
            combined_pred = np.where(
                (road_boundary_mask == 1) & ((pred_full > 0) | (pred_fullimg > 0)),
                np.maximum(pred_full, pred_fullimg),
                0
            ).astype(np.uint8)
            combined_pred = np.clip(combined_pred, 0, NUM_CLASSES-1)
            # --- Save all predicted masks as PNG (single output per image) ---
            pred_mask_rgb = mask_to_rgb(pred_full)
            pred_mask_rgb = cv2.resize(pred_mask_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
            out_mask_path = os.path.join(OUTPUT_DIR, f"{img_name}_pred_mask.png")
            #cv2.imwrite(out_mask_path, cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2BGR))
            pred_fullimg_rgb = mask_to_rgb(pred_fullimg)
            pred_fullimg_rgb = cv2.resize(pred_fullimg_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
            out_fullimg_path = os.path.join(OUTPUT_DIR, f"{img_name}_fullimg_pred_mask.png")
            #cv2.imwrite(out_fullimg_path, cv2.cvtColor(pred_fullimg_rgb, cv2.COLOR_RGB2BGR))
            combined_pred_rgb = mask_to_rgb(combined_pred)
            combined_pred_rgb = cv2.resize(combined_pred_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
            out_combined_path = os.path.join(OUTPUT_DIR, f"{img_name}_combined_pred_mask.png")
            #cv2.imwrite(out_combined_path, cv2.cvtColor(combined_pred_rgb, cv2.COLOR_RGB2BGR))
            # --- Remap GT and all predictions to binary for fair metrics and visualization ---
            gt_path = os.path.join(GT_DIR, img_name + '_mask.png')
            if os.path.exists(gt_path):
                gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_raw is not None:
                    gt_mask = remap_automine_mask(gt_raw)
                    gt_mask = np.clip(gt_mask, 0, NUM_CLASSES-1)
                    gt_binary = remap_to_binary_defect(gt_mask)
                    pred_binary = remap_to_binary_defect(pred_full)
                    pred_fullimg_binary = remap_to_binary_defect(pred_fullimg)
                    combined_pred_binary = remap_to_binary_defect(combined_pred)
                    # Visualize only if there are defects in GT
                    if np.any(gt_binary == 1):
                        # --- Calculate metrics for all three prediction types ---
                        metrics_dict = {}
                        for name, binary_mask in zip(
                            ['cascade', 'fullimg', 'combined'],
                            [pred_binary, pred_fullimg_binary, combined_pred_binary]
                        ):
                            # Binary metrics only
                            bin_metrics = compute_binary_defect_metrics(binary_mask, gt_binary)
                            # Add defect context
                            total_pixels = gt_binary.size
                            defect_pixels = int(np.sum(gt_binary == 1))
                            defect_ratio = defect_pixels / total_pixels if total_pixels > 0 else 0.0
                            if defect_ratio < 0.01:
                                defect_size_category = 'small'
                            elif defect_ratio < 0.05:
                                defect_size_category = 'medium'
                            else:
                                defect_size_category = 'large'
                            metrics_dict[name] = {
                                'recall': bin_metrics.get('binary_recall', 0.0),
                                'precision': bin_metrics.get('binary_precision', 0.0),
                                'f1': bin_metrics.get('binary_f1', 0.0),
                                'mIoU': bin_metrics.get('binary_iou', 0.0),
                                'defect_pixel_ratio': defect_ratio,
                                'defect_pixel_count': defect_pixels,
                                'total_pixels': total_pixels,
                                'defect_size_category': defect_size_category
                            }
                        # Save per-image metrics
                        metrics_path = os.path.join(OUTPUT_DIR, f"{img_name}_metrics.json")
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics_dict, f, indent=2)
                        # --- Visualize combined best-of mask (FP, TP, FN, TN) ---
                        tp = ((gt_binary == 1) & (combined_pred_binary == 1)).astype(np.uint8)
                        fp = ((gt_binary == 0) & (combined_pred_binary == 1)).astype(np.uint8)
                        fn = ((gt_binary == 1) & (combined_pred_binary == 0)).astype(np.uint8)
                        tn = ((gt_binary == 0) & (combined_pred_binary == 0)).astype(np.uint8)
                        vis = np.zeros((H, W, 3), dtype=np.uint8)
                        vis[tn == 1] = (0, 0, 0)         # TN: black
                        vis[tp == 1] = (0, 255, 0)       # TP: green (RGB)
                        vis[fp == 1] = (255, 0, 0)       # FP: red (RGB)
                        vis[fn == 1] = (255, 255, 0)     # FN: yellow (RGB)
                        vis_path = os.path.join(OUTPUT_DIR, f"{img_name}_combined_bin_vis.png")
                        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
    # --- Final summary ---
    # Aggregate binary metrics for all images with defects
    import scipy.stats
    import random

    def compute_balanced_metrics(all_metrics):
        """Weight each image equally regardless of defect size"""
        balanced_summary = {}
        for method in ['cascade', 'fullimg', 'combined']:
            method_metrics = []
            for img_metrics in all_metrics:
                if method in img_metrics:
                    method_metrics.append(img_metrics[method])
            if method_metrics:
                balanced_summary[method] = {
                    'recall': float(np.mean([m['recall'] for m in method_metrics])),
                    'precision': float(np.mean([m['precision'] for m in method_metrics])),
                    'f1': float(np.mean([m['f1'] for m in method_metrics])),
                    'mIoU': float(np.mean([m['mIoU'] for m in method_metrics])),
                    'num_images': len(method_metrics)
                }
        return balanced_summary

    def stratify_by_defect_size(metrics_list):
        """Group images by defect size and sample equally from each group"""
        small_defects = []   # < 1% defect pixels
        medium_defects = []  # 1-5% defect pixels
        large_defects = []   # > 5% defect pixels
        for metrics in metrics_list:
            # Use combined if available, else cascade
            m = metrics.get('combined', metrics.get('cascade', {}))
            defect_ratio = m.get('defect_pixel_ratio', 0)
            if defect_ratio < 0.01:
                small_defects.append(metrics)
            elif defect_ratio < 0.05:
                medium_defects.append(metrics)
            else:
                large_defects.append(metrics)
        min_samples = min(len(small_defects), len(medium_defects), len(large_defects))
        if min_samples > 0:
            balanced_sample = (
                random.sample(small_defects, min_samples) +
                random.sample(medium_defects, min_samples) +
                random.sample(large_defects, min_samples)
            )
            return balanced_sample
        else:
            return metrics_list

    def compute_robust_summary(all_metrics):
        """Use median and trimmed means instead of simple averages"""
        summary = {}
        for method in ['cascade', 'fullimg', 'combined']:
            method_values = {'recall': [], 'precision': [], 'f1': [], 'mIoU': []}
            for img_metrics in all_metrics:
                if method in img_metrics:
                    for metric in method_values:
                        method_values[metric].append(img_metrics[method][metric])
            if method_values['recall']:
                summary[method] = {}
                for metric, values in method_values.items():
                    summary[method].update({
                        f'{metric}_mean': float(np.mean(values)),
                        f'{metric}_median': float(np.median(values)),
                        f'{metric}_trimmed_mean': float(scipy.stats.trim_mean(values, 0.2)),
                        f'{metric}_std': float(np.std(values))
                    })
        return summary

    # Collect all per-image metrics
    all_metrics = []
    for img_path in all_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        metrics_path = os.path.join(OUTPUT_DIR, f"{img_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            all_metrics.append(metrics)
    if not all_metrics:
        summary = {}
    else:
        summary = {
            'balanced_by_image': compute_balanced_metrics(all_metrics),
            'robust_statistics': compute_robust_summary(all_metrics),
            'stratified_by_size': compute_balanced_metrics(stratify_by_defect_size(all_metrics))
        }
    with open(os.path.join(OUTPUT_DIR, 'cascade_eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info("\n=== CASCADE EVALUATION SUMMARY ===\n" + json.dumps(summary, indent=2))
    # Save full-image and combined results as well
    with open(os.path.join(OUTPUT_DIR, 'fullimg_eval_results.json'), 'w') as f:
        json.dump(results_full, f, indent=2)
    if 'results_combined' in locals():
        with open(os.path.join(OUTPUT_DIR, 'combined_eval_results.json'), 'w') as f:
            json.dump(results_combined, f, indent=2)

    # --- Per-image bar chart for F1, recall, precision ---
    # Only for images with defects (GT-based, consistent across all pred types)
    filtered = []
    img_labels = []
    for i, metrics in enumerate(all_metrics):
        m = metrics.get('combined', metrics.get('cascade', {}))
        if m.get('defect_pixel_count', 0) > 0:
            filtered.append(metrics)
            img_name = os.path.splitext(os.path.basename(all_images[i]))[0]
            img_labels.append(img_name)
    if filtered:
        x = np.arange(len(img_labels))
        width = 0.25
        for metric in metric_names:
            plt.figure(figsize=(max(10, len(img_labels)*0.5), 6))
            for i, pred in enumerate(pred_types):
                vals = [img.get(pred, {}).get(metric, 0.0) for img in filtered]
                plt.bar(x + i*width, vals, width, label=pred)
            plt.xticks(x + width, img_labels, rotation=90)
            plt.ylim(0, 1)
            plt.ylabel(metric.capitalize())
            plt.title(f'Per-image {metric.capitalize()} by Prediction Type')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'per_image_{metric}_bar.png'))
            plt.close()

    # --- Per-image bar charts grouped by defect size ---
    defect_sizes = ['small', 'medium', 'large']
    for size in defect_sizes:
        filtered_by_size = []
        img_labels_by_size = []
        for i, metrics in enumerate(all_metrics):
            m = metrics.get('combined', metrics.get('cascade', {}))
            if (m.get('defect_pixel_count', 0) > 0 and 
                m.get('defect_size_category', None) == size):
                filtered_by_size.append(metrics)
                img_name = os.path.splitext(os.path.basename(all_images[i]))[0]
                img_labels_by_size.append(img_name)
        if filtered_by_size:
            x = np.arange(len(img_labels_by_size))
            width = 0.25
            for metric in metric_names:
                plt.figure(figsize=(max(10, len(img_labels_by_size)*0.5), 6))
                for i, pred in enumerate(pred_types):
                    vals = [img.get(pred, {}).get(metric, 0.0) for img in filtered_by_size]
                    plt.bar(x + i*width, vals, width, label=pred)
                plt.xticks(x + width, img_labels_by_size, rotation=90)
                plt.ylim(0, 1)
                plt.ylabel(metric.capitalize())
                plt.title(f'Per-image {metric.capitalize()} by Prediction Type ({size} defects)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'per_image_{metric}_bar_{size}.png'))
                plt.close()

    # --- Summary bar charts: median per defect size and prediction type ---
    for metric in metric_names:
        medians = []  # shape: [defect_size][pred_type]
        for size in defect_sizes:
            size_medians = []
            for pred in pred_types:
                vals = []
                for i, img in enumerate(all_metrics):
                    gt_info = img.get('combined', img.get('cascade', {}))
                    if (gt_info.get('defect_size_category', None) == size and 
                        gt_info.get('defect_pixel_count', 0) > 0 and
                        pred in img):
                        vals.append(img[pred].get(metric, 0.0))
                if vals:
                    size_medians.append(np.median(vals))
                else:
                    size_medians.append(0.0)
            medians.append(size_medians)
        medians = np.array(medians)  # shape: (3, 3)
        x = np.arange(len(defect_sizes))
        width = 0.2
        plt.figure(figsize=(8, 6))
        for i, pred in enumerate(pred_types):
            plt.bar(x + i*width, medians[:, i], width, label=pred)
        plt.xticks(x + width, defect_sizes)
        plt.ylim(0, 1)
        plt.ylabel(f'Median {metric.capitalize()}')
        plt.title(f'Median {metric.capitalize()} by Defect Size and Prediction Type')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'summary_{metric}_by_defect_size.png'))
        plt.close()

if __name__ == "__main__":
    metric_names = ['f1', 'recall', 'precision']
    pred_types = ['cascade', 'fullimg', 'combined']
    defect_sizes = ['small', 'medium', 'large']
    run_cascade_evaluation()
