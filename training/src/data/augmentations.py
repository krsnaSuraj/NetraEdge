"""
Data augmentation pipeline for face training.

Uses albumentations for efficient, GPU-accelerated augmentations.
Designed to simulate real-world conditions:
- Harsh sunlight / low light
- Camera noise
- Partial face occlusion
- Head rotation / tilt
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_recognition_train_transforms() -> A.Compose:
    """Augmentation pipeline for face recognition training."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3),
            contrast_limit=(-0.3, 0.3),
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=25,
            val_shift_limit=20,
            p=0.5,
        ),
        A.Rotate(limit=15, border_mode=0, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(5, 15), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomResizedCrop(
            height=112, width=112,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.6,
        ),
        A.Resize(height=112, width=112),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_recognition_val_transforms() -> A.Compose:
    """Minimal transforms for validation/test."""
    return A.Compose([
        A.Resize(height=112, width=112),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_liveness_train_transforms() -> A.Compose:
    """Augmentation pipeline for liveness detection training."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3),
            contrast_limit=(-0.3, 0.3),
            p=0.7,
        ),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(5, 15), p=0.4),
        A.Resize(height=112, width=112),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_liveness_val_transforms() -> A.Compose:
    """Minimal transforms for liveness validation."""
    return A.Compose([
        A.Resize(height=112, width=112),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
