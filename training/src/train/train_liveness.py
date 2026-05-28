"""
Training script for liveness detection model (LivenessCNN).

Usage:
    python -m src.train.train_liveness --config configs/liveness.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ..models.liveness_cnn import LivenessCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

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

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
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

    model = LivenessCNN(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"].get("dropout_rate", 0.3),
    ).to(device)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    train_cfg = config["training"]

    class_weights = config["loss"].get("class_weights", [1.0, 1.0, 1.0])
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config["loss"].get("label_smoothing", 0.0))

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

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch completed in %.1fs — Train Loss: %.4f — Val Acc: %.2f%%",
            elapsed, 0, 0,
        )

    logger.info("Training complete. Best accuracy: %.2f%%", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train liveness detection model")
    parser.add_argument("--config", default="configs/liveness.yaml", help="Config YAML path")
    args = parser.parse_args()
    main(args.config)
