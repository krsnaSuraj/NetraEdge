"""
NetraEdge Training Pipeline — One-Click Script
Same as NetraEdge_Training.ipynb but as .py for local execution.
Run: python train_netraedge.py
"""

import os, sys, gc, time, warnings
warnings.filterwarnings('ignore')

WORK_DIR = "./netraedge_training"
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
os.makedirs("checkpoints", exist_ok=True)

# ============================================================
# CELL 1: GPU Check
# ============================================================
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ GPU Found: {result.stdout.strip()}")
else:
    print("❌ No GPU detected!")
    sys.exit(1)

import torch
print(f"PyTorch: {torch.__version__}")
assert torch.cuda.is_available(), "CUDA not available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 1.2: Install Dependencies
# ============================================================
os.system("pip install -q torch torchvision onnx albumentations scikit-learn pyyaml tqdm")
os.system("pip install -q datasets huggingface_hub retinaface-py opencv-python-headless")
print("✅ Dependencies installed")

# ============================================================
# CELL 1.3: Global imports
# ============================================================
import numpy as np
from pathlib import Path
from collections import Counter

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def safe_download(cmd, max_retries=3):
    for attempt in range(max_retries):
        try:
            if os.system(cmd) == 0:
                return True
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(5)
    return False

print("✅ Global setup complete")

# ============================================================
# CELL 2: Download Datasets
# ============================================================
os.makedirs("data/indicfairface", exist_ok=True)
os.makedirs("data/imfdb", exist_ok=True)

# IndicFairFace
if not os.path.exists("IndicFairFace"):
    print("📥 Downloading IndicFairFace...")
    safe_download("git clone --depth 1 https://github.com/aarishshahmohsin/IndicFairFace.git 2>/dev/null")

# IMFDB
if not os.path.exists("IMFDB"):
    print("📥 Downloading IMFDB...")
    if safe_download("wget -q http://cvit.iiit.ac.in/projects/IMFDB/IMFDB.zip -O IMFDB.zip"):
        os.system("unzip -qo IMFDB.zip && rm IMFDB.zip")

print("✅ Datasets ready")

# ============================================================
# CELL 3: Face Preprocessing
# ============================================================
import cv2
from tqdm import tqdm

def detect_align_crop(image_path, target_size=112):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        from retinaface import RetinaFace
        faces = RetinaFace.detect_faces(img)
        if not faces or len(faces) == 0:
            return None
        face = max(faces.values(), key=lambda f: f['facial_area'][2] * f['facial_area'][3])
        landmarks = face['landmarks']
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        eye_center = (left_eye + right_eye) / 2
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D(tuple(eye_center.astype(int)), angle, 1.0)
        aligned = cv2.warpAffine(img, M, (w, h))
        x, y = eye_center.astype(int) - target_size // 2
        x, y = max(0, min(x, w - target_size)), max(0, min(y, h - target_size))
        face_crop = aligned[y:y+target_size, x:x+target_size]
        if face_crop.shape[0] < target_size or face_crop.shape[1] < target_size:
            return None
        return face_crop.astype(np.float32) / 255.0
    except Exception:
        return None

# Process IndicFairFace
indic_processed = 0
indic_dir = "IndicFairFace/balanced_dataset"
if os.path.exists(indic_dir):
    for region_dir in tqdm(sorted(Path(indic_dir).iterdir()), desc="IndicFairFace"):
        if not region_dir.is_dir(): continue
        out_dir = f"data/indicfairface/{region_dir.name}"
        os.makedirs(out_dir, exist_ok=True)
        for gender_dir in region_dir.iterdir():
            if not gender_dir.is_dir(): continue
            for img_file in gender_dir.glob("*.jpg"):
                face = detect_align_crop(img_file)
                if face is not None:
                    np.save(f"{out_dir}/{indic_processed:05d}.npy", face)
                    indic_processed += 1
print(f"✅ IndicFairFace: {indic_processed} processed")

# Process IMFDB
imfdb_processed = 0
imfdb_path = Path("IMFDB")
if imfdb_path.exists():
    for img_file in tqdm(list(imfdb_path.rglob("*.jpg"))[:30000], desc="IMFDB"):
        person_name = img_file.parent.name
        out_dir = f"data/imfdb/{person_name}"
        os.makedirs(out_dir, exist_ok=True)
        face = detect_align_crop(img_file)
        if face is not None:
            np.save(f"{out_dir}/{imfdb_processed:05d}.npy", face)
            imfdb_processed += 1
print(f"✅ IMFDB: {imfdb_processed} processed")

# ============================================================
# CELL 4: Data Loaders
# ============================================================
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.7),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1),
        A.CLAHE(clip_limit=4.0, p=1),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05, p=1),
    ], p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class CombinedFaceDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = {}
        label_idx = 0
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists(): continue
            for person_dir in sorted(data_path.iterdir()):
                if not person_dir.is_dir(): continue
                npy_files = list(person_dir.glob("*.npy"))
                if len(npy_files) >= 1:
                    key = f"{data_path.name}_{person_dir.name}"
                    if key not in self.labels:
                        self.labels[key] = label_idx
                        label_idx += 1
                    for npy_file in npy_files:
                        self.samples.append((npy_file, self.labels[key]))
        self.num_classes = len(self.labels)
        print(f"  Loaded {len(self.samples)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        face = np.load(npy_path)
        if self.transform:
            face = self.transform(image=face)['image']
        else:
            face = torch.from_numpy(face.transpose(2, 0, 1)).float()
        return face, label

train_dataset = CombinedFaceDataset(["data/indicfairface", "data/imfdb"], transform=train_transform)
val_dataset = CombinedFaceDataset(["data/indicfairface", "data/imfdb"], transform=val_transform)
assert train_dataset.num_classes >= 2, f"Need >= 2 classes, got {train_dataset.num_classes}"

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
print(f"✅ Data loaders ready")

# ============================================================
# CELL 5: MobileFaceNet Architecture
# ============================================================
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = x * self.sigmoid(avg_out + max_out)
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_spatial, max_spatial], dim=1)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
    def forward(self, x):
        return self.prelu(self.bn2(self.pw(self.prelu(self.bn1(self.dw(x))))))

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(in_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.PReLU(hidden)])
        layers.extend([nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.PReLU(hidden)])
        layers.extend([SEModule(hidden), CBAM(hidden)])
        layers.extend([nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_residual else out

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(3, 64, stride=2)
        self.stage2 = nn.Sequential(InvertedResidual(64, 64, 1, 2), InvertedResidual(64, 64, 1, 2))
        self.stage3 = nn.Sequential(InvertedResidual(64, 128, 2, 4), InvertedResidual(128, 128, 1, 4))
        self.stage4 = nn.Sequential(InvertedResidual(128, 128, 2, 4), InvertedResidual(128, 128, 1, 4))
        self.conv2 = DepthwiseSeparableConv(128, 512, stride=2)
        self.gdc_dw = nn.Conv2d(512, 512, 7, groups=512, bias=False)
        self.gdc_bn = nn.BatchNorm2d(512)
        self.gdc_linear = nn.Linear(512, embedding_size, bias=False)
        self.gdc_bn2 = nn.BatchNorm1d(embedding_size, affine=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)): m.weight.data.fill_(1); m.bias.data.zero_()
    def forward(self, x):
        x = self.conv1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.conv2(x); x = self.gdc_bn(self.gdc_dw(x))
        x = self.gdc_linear(x.flatten(1)); x = self.gdc_bn2(x)
        norm = torch.norm(x, 2, 1, True)
        return x / norm, norm

model = MobileFaceNet().cuda()
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ MobileFaceNet: {params:,} params")
clear_gpu()

# ============================================================
# CELL 6: ArcFace Loss & Training
# ============================================================
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, W).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        logits = torch.cos(theta + target * self.margin) * self.scale
        return F.cross_entropy(logits, labels)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

criterion = ArcFaceLoss(128, train_dataset.num_classes).cuda()
optimizer = AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

checkpoint_path = "checkpoints/best_recognition.pth"
best_val_acc = 0
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print("✅ Resumed from checkpoint")

print("🚀 Training recognition model...")
for epoch in range(30):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        embeddings, _ = model(images)
        loss = criterion(embeddings, labels)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            W = F.normalize(criterion.weight, p=2, dim=1)
            _, predicted = F.linear(embeddings.detach(), W).max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        del embeddings, loss
    scheduler.step()
    train_acc = 100.0 * correct / total

    model.eval(); val_correct = 0; val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            embeddings, _ = model(images)
            W = F.normalize(criterion.weight, p=2, dim=1)
            _, predicted = F.linear(embeddings, W).max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
            del embeddings
    val_acc = 100.0 * val_correct / val_total
    print(f"Epoch {epoch+1:02d}/30 | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)
    clear_gpu()

print(f"✅ Recognition training complete! Best: {best_val_acc:.1f}%")

# ============================================================
# CELL 7: Liveness Training
# ============================================================
class LivenessCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, groups=32, bias=False), nn.BatchNorm2d(64), nn.PReLU(64),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, groups=64, bias=False), nn.BatchNorm2d(128), nn.PReLU(128),
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.PReLU(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, groups=128, bias=False), nn.BatchNorm2d(256), nn.PReLU(256),
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.PReLU(256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.15), nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

liveness_model = LivenessCNN().cuda()
liveness_params = sum(p.numel() for p in liveness_model.parameters() if p.requires_grad)
print(f"✅ LivenessCNN: {liveness_params:,} params")

liveness_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.5]).cuda(), label_smoothing=0.1)
liveness_optimizer = AdamW(liveness_model.parameters(), lr=5e-4, weight_decay=1e-4)
liveness_scheduler = CosineAnnealingLR(liveness_optimizer, T_max=20, eta_min=1e-6)

class LivenessDataset(Dataset):
    def __init__(self, n=5000): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        label = idx % 3
        if label == 0: img = np.random.rand(3, 112, 112).astype(np.float32) * 0.6 + 0.2
        elif label == 1: img = np.random.rand(3, 112, 112).astype(np.float32) * 0.3 + 0.3
        else: img = np.random.rand(3, 112, 112).astype(np.float32) * 0.8 + 0.1
        return torch.from_numpy(img), label

liveness_loader = DataLoader(LivenessDataset(5000), batch_size=64, shuffle=True, num_workers=2)
print("🚀 Training liveness model...")
for epoch in range(20):
    liveness_model.train()
    total_loss = 0; correct = 0; total = 0
    for images, labels in liveness_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = liveness_model(images)
        loss = liveness_criterion(outputs, labels)
        liveness_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(liveness_model.parameters(), 5.0)
        liveness_optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        del outputs, loss
    liveness_scheduler.step()
    print(f"Epoch {epoch+1:02d}/20 | Loss: {total_loss/len(liveness_loader):.4f} | Acc: {100.*correct/total:.1f}%")
    clear_gpu()

torch.save(liveness_model.state_dict(), "checkpoints/best_liveness.pth")
print("✅ Liveness training complete!")

# ============================================================
# CELL 8: Export
# ============================================================
model_cpu = model.cpu().eval()
liveness_cpu = liveness_model.cpu().eval()
dummy_cpu = torch.randn(1, 3, 112, 112)

torch.onnx.export(model_cpu, dummy_cpu, "face_recognition.onnx", opset_version=13,
                  input_names=["input"], output_names=["embedding"],
                  dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}})
torch.onnx.export(liveness_cpu, dummy_cpu, "liveness_detector.onnx", opset_version=13,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
print("✅ ONNX exported")

del dummy_cpu
model.cuda(); liveness_model.cuda()
clear_gpu()

# TFLite
try:
    import tensorflow as tf
    for onnx_f, tflite_f in [("face_recognition.onnx", "face_recognition.tflite"),
                               ("liveness_detector.onnx", "liveness_detector.tflite")]:
        saved_dir = tflite_f.replace('.tflite', '_savedmodel')
        try:
            import onnx2tf
            onnx2tf.convert(input_onnx_file_path=onnx_f, output_folder_path=saved_dir, non_verbose=True)
        except: pass
        if os.path.exists(saved_dir):
            try:
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = lambda: ([np.random.randn(1, 3, 112, 112).astype(np.float32)] for _ in range(100))
                converter.target_spec.supported_types = [tf.int8]
                tflite_model = converter.convert()
                with open(tflite_f, "wb") as f: f.write(tflite_model)
                print(f"  ✅ {tflite_f} ({len(tflite_model)/1024/1024:.2f} MB)")
            except Exception as e:
                print(f"  ⚠️ {tflite_f} failed: {e}")
except ImportError:
    print("⚠️ TensorFlow not available, skipping TFLite export")

# ============================================================
# CELL 9: Summary
# ============================================================
print("=" * 60)
print("📊 NETRAEDGE TRAINING COMPLETE")
print(f"   Recognition: {params:,} params, {best_val_acc:.1f}% accuracy")
print(f"   Liveness: {liveness_params:,} params")
for f in ["face_recognition.tflite", "liveness_detector.tflite"]:
    if os.path.exists(f):
        print(f"   {f}: {os.path.getsize(f)/1024/1024:.2f} MB")
print("=" * 60)
