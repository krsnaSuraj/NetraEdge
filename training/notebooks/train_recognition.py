"""
NetraEdge — Training Notebook for Google Colab

This notebook trains both models:
1. MobileFaceNet (face recognition)
2. LivenessCNN (liveness detection)

Upload this to Google Colab and run all cells.
Models will be saved and ready for TFLite export.

Runtime: GPU (T4 recommended)
Estimated time: ~2 hours
"""

# %% [markdown]
# # NetraEdge Model Training
# ## Face Recognition + Liveness Detection
#
# **Runtime:** GPU (T4)
# **Estimated time:** ~2 hours
# **Output:** Trained PyTorch checkpoints

# %%
# Cell 1: Setup and installation
import subprocess
import sys

def install_package(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install dependencies
for pkg in ["torch", "torchvision", "onnx", "numpy", "opencv-python", "pillow",
            "scikit-learn", "pyyaml", "tqdm", "albumentations", "matplotlib"]:
    install_package(pkg)

print("Dependencies installed.")

# %%
# Cell 2: Clone NetraEdge repo (or upload training/ directory)
import os

# Option A: Clone from GitHub
if not os.path.exists("netraedge"):
    os.system("git clone https://github.com/krsnaSuraj/NetraEdge.git")
    print("Cloned NetraEdge repository.")
else:
    print("NetraEdge directory already exists.")

# Move to training directory
os.chdir("NetraEdge/training")
print(f"Working directory: {os.getcwd()}")

# %%
# Cell 3: Download LFW dataset
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

lfw_path = data_dir / "lfw-funneled.tgz"
lfw_dir = data_dir / "lfw"

if not lfw_dir.exists():
    print("Downloading LFW dataset (~170MB)...")
    urlretrieve(
        "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
        str(lfw_path),
    )
    print("Extracting...")
    with tarfile.open(str(lfw_path), "r:gz") as tar:
        tar.extractall(str(data_dir))
    print("LFW dataset ready.")
else:
    print("LFW dataset already exists.")

# Count identities
identities = [d for d in lfw_dir.iterdir() if d.is_dir()] if lfw_dir.exists() else []
print(f"LFW: {len(identities)} identities")

# %%
# Cell 4: Model architecture — MobileFaceNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(ch // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


class MobileBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        mid = in_ch * 2
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.dw = DepthwiseSeparableConv(mid, out_ch, stride)
        self.se = SEBlock(out_ch)
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.dw(self.expand(x)))
        return out + x if self.residual else out


class MobileFaceNet(nn.Module):
    def __init__(self, emb_dim: int = 128) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            MobileBlock(64, 64), MobileBlock(64, 128, 2), MobileBlock(128, 128),
            MobileBlock(128, 256, 2), MobileBlock(256, 256), MobileBlock(256, 256),
            MobileBlock(256, 512, 2), MobileBlock(512, 512), MobileBlock(512, 512),
        )
        self.final = nn.Sequential(
            nn.Conv2d(512, 512, 3, groups=512, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, emb_dim, 1, bias=False), nn.BatchNorm2d(emb_dim),
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
        x = self.final(self.blocks(self.stem(x)))
        return F.normalize(x.view(x.size(0), -1), p=2, dim=1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Verify model
model = MobileFaceNet(128)
dummy = torch.randn(2, 3, 112, 112)
out = model(dummy)
print(f"MobileFaceNet: {model.count_params():,} params, output shape: {out.shape}")
assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"
print("Model architecture verified.")

# %%
# Cell 5: Model architecture — LivenessCNN
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
            LivenessBlock(32, 64), LivenessBlock(64, 128, 2), LivenessBlock(128, 256, 2), LivenessBlock(256, 256, 2),
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


liveness_model = LivenessCNN(3)
out = liveness_model(dummy)
print(f"LivenessCNN: {liveness_model.count_params():,} params, output shape: {out.shape}")
assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"
print("Liveness model verified.")

# %%
# Cell 6: Data augmentation
from PIL import Image
import numpy as np

def augment_face(img: Image.Image) -> torch.Tensor:
    """Apply augmentations simulating outdoor conditions."""
    arr = np.array(img).astype(np.float32) / 255.0

    # Random brightness (-30 to +30 in 0-255 range)
    if np.random.random() > 0.5:
        arr = np.clip(arr + np.random.uniform(-0.12, 0.12), 0, 1)

    # Random contrast
    if np.random.random() > 0.5:
        mean = arr.mean()
        arr = np.clip((arr - mean) * np.random.uniform(0.7, 1.3) + mean, 0, 1)

    # Random horizontal flip
    if np.random.random() > 0.5:
        arr = np.flip(arr, axis=1).copy()

    # Convert to tensor
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def load_lfw_image(path: str) -> torch.Tensor:
    """Load and preprocess an LFW image."""
    img = Image.open(path).convert("RGB")
    img = img.resize((112, 112), Image.BILINEAR)
    return augment_face(img)


print("Augmentation pipeline ready.")

# %%
# Cell 7: Training loop — face recognition
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import time


class LWFDataset(Dataset):
    def __init__(self, root: str, max_identities: int = 100) -> None:
        self.root = Path(root)
        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}

        identities = sorted([d for d in self.root.iterdir() if d.is_dir()])[:max_identities]
        for idx, identity in enumerate(identities):
            self.class_to_idx[identity.name] = idx
            for img_path in identity.glob("*.jpg"):
                self.samples.append((str(img_path), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((112, 112), Image.BILINEAR)
        return augment_face(img), label


# Training config
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Load dataset
train_dataset = LWFDataset("data/lfw", max_identities=80)
val_dataset = LWFDataset("data/lfw", max_identities=100)

print(f"Train: {len(train_dataset)} images, {len(train_dataset.class_to_idx)} identities")
print(f"Val: {len(val_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Initialize model
recognition_model = MobileFaceNet(EMBEDDING_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(recognition_model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Training
best_val_acc = 0.0
history: list[dict] = []

for epoch in range(EPOCHS):
    t0 = time.time()

    # Train
    recognition_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        embeddings = recognition_model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(recognition_model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item()
        preds = embeddings.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # Validate
    recognition_model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            embeddings = recognition_model(images)
            preds = embeddings.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    train_acc = 100.0 * train_correct / train_total
    val_acc = 100.0 * val_correct / val_total
    elapsed = time.time() - t0

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} — {elapsed:.0f}s — "
          f"Loss: {train_loss/len(train_loader):.4f} — "
          f"Train Acc: {train_acc:.1f}% — Val Acc: {val_acc:.1f}%")

    history.append({"epoch": epoch+1, "train_acc": train_acc, "val_acc": val_acc, "time": elapsed})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": recognition_model.state_dict(),
            "val_acc": val_acc,
            "epoch": epoch + 1,
        }, "checkpoints/recognition_best.pt")
        print(f"  New best model saved (val_acc={val_acc:.1f}%)")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")

# %%
# Cell 8: Export to ONNX
print("Exporting face recognition model to ONNX...")

recognition_model.eval()
dummy_input = torch.randn(1, 3, 112, 112).to(DEVICE)

torch.onnx.export(
    recognition_model.cpu(),
    torch.randn(1, 3, 112, 112),
    "checkpoints/face_recognition.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)
print("ONNX export complete: checkpoints/face_recognition.onnx")

# %% [markdown]
# # Next Steps
#
# 1. Download `face_recognition.onnx` from Colab
# 2. Convert to TFLite with INT8 quantization:
#    ```bash
#    pip install onnx2tf
#    onnx2tf -i face_recognition.onnx -o models/
#    ```
# 3. Repeat for LivenessCNN (same pipeline)
# 4. Place `.tflite` files in `models/` directory
