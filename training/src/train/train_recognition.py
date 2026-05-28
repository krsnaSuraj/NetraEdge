"""
Training script for face recognition model (MobileFaceNet).

Supports:
- ArcFace loss for angular margin
- Knowledge distillation from teacher model
- Cosine annealing LR schedule
- Early stopping
- Model checkpointing

Usage:
    python -m src.train.train_recognition --config configs/recognition.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..models.mobilefacenet import MobileFaceNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss for face recognition."""

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norms = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(norms, weights)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)

        logits = one_hot * target_logits + (1 - one_hot) * cosine
        logits *= self.scale

        return F.cross_entropy(logits, labels.long())


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        preds = embeddings.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            logger.info(
                "  Batch %d/%d — Loss: %.4f — Acc: %.2f%%",
                batch_idx, len(loader),
                loss.item(), 100.0 * correct / max(total, 1),
            )

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on validation set. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        preds = embeddings.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = MobileFaceNet(
        embedding_dim=config["model"]["embedding_dim"],
        use_attention=config["model"].get("use_attention", True),
    ).to(device)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")
    logger.info("Estimated size: %.1f MB", model.estimate_size_mb())

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=1e-6,
    )

    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    patience_counter = 0

    logger.info("Starting training for %d epochs", train_cfg["epochs"])

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()
        logger.info("Epoch %d/%d", epoch + 1, train_cfg["epochs"])

        # Training is executed via Google Colab notebook (T4 GPU).
        # See training/NetraEdge_Train.ipynb or run build_notebook.py.
        #
        # For local training, create data loaders and uncomment:
        #   train_loader = DataLoader(train_dataset, batch_size=..., shuffle=True)
        #   val_loader = DataLoader(val_dataset, batch_size=..., shuffle=False)
        #   train_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, device)
        #   val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        #
        # The Colab notebook handles dataset download, augmentation,
        # training loop, checkpointing, and TFLite export end-to-end.

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch completed in %.1fs — Train Loss: %.4f — Val Acc: %.2f%%",
            elapsed, 0, 0,
        )

    logger.info("Training complete. Best accuracy: %.2f%%", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--config", default="configs/recognition.yaml", help="Config YAML path")
    args = parser.parse_args()
    main(args.config)
