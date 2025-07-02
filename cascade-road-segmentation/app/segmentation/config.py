#!/usr/bin/env python
"""
Configuration constants for segmentation module - Simplified for deployment
"""

import os

# -----------------------------
# Class Configuration
# -----------------------------
CLASSES = ['background', 'pothole', 'crack', 'puddle', 'distressed_patch', 'mud']
NUM_CLASSES = len(CLASSES)

# -----------------------------
# Color Mappings
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

# Automine classes mapping
AUTOMINE_MAPPING = {
    0: 0,  # background -> background
    1: 4,  # defect -> distressed_patch
    2: 1,  # pothole -> pothole
    3: 3,  # puddle -> puddle
    4: 0,  # road -> background
}

# RTK mapping
RTK_TO_MODEL = {
    11: 1,  # pothole → pothole
    12: 2,  # craks → crack
    10: 3,  # waterPuddle → puddle
    9: 4,   # patchs → distressed_patch
}

# Fallback paths for dataset locations (simplified for deployment)
FALLBACK_PATHS = {
    'pothole_mix_train': [
        "../data/pothole_mix/training", 
        "../data/pothole_mix/train", 
        "../data/pothole-mix/training"
    ],
    'pothole_mix_val': [
        "../data/pothole_mix/validation", 
        "../data/pothole_mix/val", 
        "../data/pothole-mix/validation"
    ],
    'automine_train': [
        "../data/train/train_train", 
        "../data/automine/train"
    ],
    'automine_val': [
        "../data/valid/train_valid", 
        "../data/automine/val"
    ],
    'r2s100k_train': [
        "../data/r2s100k/train"
    ],
    'r2s100k_train_labels': [
        "../data/r2s100k/train-labels"
    ],
    'r2s100k_val_labels': [
        "../data/r2s100k/val_labels",
        "../data/r2s100k/val-labels"
    ]
}

# Device configuration - simplified
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Image size
IMG_SIZE = 512