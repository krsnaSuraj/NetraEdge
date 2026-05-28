"""
NetraEdge Liveness Training Notebook for Google Colab

Train the LivenessCNN model for face anti-spoofing.
Classifies faces as: real, print_attack, screen_attack.

Runtime: GPU (T4)
Estimated time: ~1 hour
"""

# %% [markdown]
# # NetraEdge Liveness Detection Training
# ## Anti-Spoofing: Real vs Print vs Screen
#
# **Classes:**
# - 0: Real (live person)
# - 1: Print attack (photo held to camera)
# - 2: Screen attack (phone/tablet showing photo)
#
# **Runtime:** GPU (T4)

# %%
# Cell 1: Setup
import subprocess, sys

for pkg in ["torch", "torchvision", "numpy", "opencv-python", "pillow",
            "scikit-learn", "tqdm", "matplotlib"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

print(f"PyTorch: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# %%
# Cell 2: LivenessCNN Architecture
class LivenessBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class LivenessCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            LivenessBlock(32, 64), LivenessBlock(64, 128, 2),
            LivenessBlock(128, 256, 2), LivenessBlock(256, 256, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5), nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


model = LivenessCNN(3)
print(f"LivenessCNN: {model.count_params():,} params")
dummy = torch.randn(2, 3, 112, 112)
out = model(dummy)
print(f"Output shape: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Architecture verified.")

# %%
# Cell 3: Synthetic Liveness Dataset Generator
# Since real spoof datasets require special access, we generate synthetic data
# that simulates print/screen attacks using image transformations

import random

class SyntheticLivenessDataset(Dataset):
    """
    Generates synthetic liveness training data.

    Real faces: Normal images from LFW
    Print attacks: Photos of photos (re-captured with noise + blur)
    Screen attacks: Photos of screens (scan lines + moire pattern + brightness)
    """

    LABEL_NAMES = {0: "real", 1: "print", 2: "screen"}

    def __init__(self, lfw_dir: str, num_samples: int = 3000, transform=None) -> None:
        self.lfw_dir = Path(lfw_dir)
        self.num_samples = num_samples
        self.transform = transform
        self.all_images: list[str] = []

        for img_path in self.lfw_dir.rglob("*.jpg"):
            self.all_images.append(str(img_path))

        if not self.all_images:
            raise RuntimeError(f"No images found in {lfw_dir}")

        print(f"Found {len(self.all_images)} source images for synthetic generation")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        source_path = random.choice(self.all_images)
        label = idx % 3  # 0: real, 1: print, 2: screen

        img = Image.open(source_path).convert("RGB").resize((112, 112), Image.BILINEAR)

        if label == 0:
            # Real face — standard augmentation
            pass
        elif label == 1:
            # Print attack — simulate re-photographing
            img = self._simulate_print(img)
        else:
            # Screen attack — simulate screen capture
            img = self._simulate_screen(img)

        arr = np.array(img).astype(np.float32) / 255.0

        # Add noise
        arr = np.clip(arr + np.random.normal(0, 0.02, arr.shape), 0, 1)

        # Random flip
        if random.random() > 0.5:
            arr = np.flip(arr, axis=1).copy()

        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std, label

    def _simulate_print(self, img: Image.Image) -> Image.Image:
        """Simulate print attack by re-photographing."""
        arr = np.array(img)

        # Slight blur (simulating camera focus)
        kernel = np.ones((3, 3)) / 9
        arr = np.stack([
            np.convolve(arr[:, :, c].flatten(), kernel.flatten(), mode="same").reshape(112, 112)
            for c in range(3)
        ], axis=-1).astype(np.uint8)

        # Increase brightness slightly (flash)
        arr = np.clip(arr.astype(np.float32) * 1.1 + 10, 0, 255).astype(np.uint8)

        # Add JPEG compression artifacts
        img = Image.fromarray(arr)
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        return Image.open(buf)

    def _simulate_screen(self, img: Image.Image) -> Image.Image:
        """Simulate screen attack."""
        arr = np.array(img).astype(np.float32)

        # Scan lines (horizontal lines every 2 pixels)
        for y in range(0, 112, 2):
            arr[y, :, :] *= 0.85

        # Moire pattern
        for y in range(112):
            for x in range(112):
                arr[y, x, :] *= 1.0 + 0.05 * np.sin(y * 2 * np.pi / 4)

        # Blue tint (screen color temperature)
        arr[:, :, 2] *= 1.1

        # Reduce contrast
        arr = arr * 0.8 + 30

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# %%
# Cell 4: Create dataset and loaders
LFW_DIR = "data/lfw"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
LR = 5e-4

train_dataset = SyntheticLivenessDataset(LFW_DIR, num_samples=3000)
val_dataset = SyntheticLivenessDataset(LFW_DIR, num_samples=600)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")

# %%
# Cell 5: Training loop
model = LivenessCNN(3).to(DEVICE)

# Class weights for imbalanced data
weights = torch.tensor([1.0, 1.5, 1.5]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    t0 = time.time()

    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item()
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            for i in range(3):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += ((preds == labels) & mask).sum().item()

    train_acc = 100.0 * train_correct / train_total
    val_acc = 100.0 * val_correct / val_total
    elapsed = time.time() - t0
    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS} — {elapsed:.0f}s")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f} — Train Acc: {train_acc:.1f}%")
    print(f"  Val Acc: {val_acc:.1f}%")
    for i, name in enumerate(["Real", "Print", "Screen"]):
        acc = 100.0 * class_correct[i] / max(class_total[i], 1)
        print(f"  {name}: {acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "epoch": epoch + 1,
        }, "checkpoints/liveness_best.pt")
        print(f"  Saved best model (val_acc={val_acc:.1f}%)")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")

# %%
# Cell 6: Export to ONNX
print("Exporting liveness model to ONNX...")

model.cpu().eval()
torch.onnx.export(
    model, torch.randn(1, 3, 112, 112),
    "checkpoints/liveness_detector.onnx",
    input_names=["input"], output_names=["output"], opset_version=13,
)
print("ONNX export complete: checkpoints/liveness_detector.onnx")

# %% [markdown]
# # Download Trained Models
#
# Run this cell to download both .onnx files:
# - `face_recognition.onnx`
# - `liveness_detector.onnx`
#
# Convert to TFLite:
# ```bash
# pip install onnx2tf
# onnx2tf -i face_recognition.onnx -o models/
# onnx2tf -i liveness_detector.onnx -o models/
# ```
