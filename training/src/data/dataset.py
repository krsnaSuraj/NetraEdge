"""
Face dataset loader for training and evaluation.

Supports:
- VGGFace2 (large-scale face recognition)
- LFW (benchmark)
- IMFDB (Indian demographics)
- CelebA-Spoof (liveness detection)
- NUAA (print attack)
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class FaceRecognitionDataset(Dataset):
    """
    Generic face recognition dataset.

    Expects directory structure:
        root/
            person_001/
                img_001.jpg
                img_002.jpg
            person_002/
                ...
    """

    def __init__(
        self,
        root: str,
        transform=None,
        max_identities: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples(max_identities)

    def _load_samples(self, max_identities: Optional[int]) -> None:
        if not self.root.exists():
            return

        identities = sorted([
            d for d in self.root.iterdir()
            if d.is_dir()
        ])

        if max_identities is not None:
            identities = identities[:max_identities]

        for idx, identity_dir in enumerate(identities):
            self.class_to_idx[identity_dir.name] = idx
            for img_path in identity_dir.glob("*.jpg"):
                self.samples.append((str(img_path), idx))
            for img_path in identity_dir.glob("*.png"):
                self.samples.append((str(img_path), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img, label


class LivenessDataset(Dataset):
    """
    Liveness detection dataset.

    Expects directory structure:
        root/
            real/
                img_001.jpg
            print/
                img_001.jpg
            screen/
                img_001.jpg
    """

    LABEL_MAP = {"real": 0, "print": 1, "screen": 2}

    def __init__(self, root: str, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

        self._load_samples()

    def _load_samples(self) -> None:
        if not self.root.exists():
            return

        for label_name, label_idx in self.LABEL_MAP.items():
            label_dir = self.root / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), label_idx))
                    self.class_counts[label_idx] = self.class_counts.get(label_idx, 0) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        total = sum(self.class_counts.values())
        weights = []
        for i in range(3):
            count = self.class_counts.get(i, 1)
            weights.append(total / (3 * count))
        return torch.tensor(weights, dtype=torch.float32)


def create_recognition_loaders(
    train_root: str,
    val_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    max_identities: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders for face recognition."""
    train_dataset = FaceRecognitionDataset(train_root, max_identities=max_identities)
    val_dataset = FaceRecognitionDataset(val_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_liveness_loaders(
    train_root: str,
    val_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders for liveness detection."""
    train_dataset = LivenessDataset(train_root)
    val_dataset = LivenessDataset(val_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
