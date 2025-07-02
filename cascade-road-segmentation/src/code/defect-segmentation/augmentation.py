#!/usr/bin/env python
"""
Data augmentation and preprocessing utilities for road defect segmentation.
Exactly matches the original script transformations.
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE


def get_train_transform():
    """Get training augmentation pipeline matching the original script exactly"""
    return A.Compose([
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),      # ← tupla (h, w)
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1)
        ], p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
            A.Compose([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
                A.Affine(scale=(0.9, 1.1), translate_percent=0.02, rotate=(-5, 5), p=1),
            ]),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=0.2, p=1)
        ], p=0.3),
    ])


def get_val_transform():
    """Get validation augmentation pipeline matching the original script exactly"""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE)
    ])


def get_preprocessing(preprocessing_fn):
    """Creates preprocessing pipeline with memory optimizations matching original script"""
    def _preprocess(img, **kwargs):
        # Convert to float32 only after preprocessing to save memory
        return preprocessing_fn(img).astype(np.float32)

    return A.Compose([
        A.Lambda(name="preproc", image=_preprocess),
        ToTensorV2()
    ])
