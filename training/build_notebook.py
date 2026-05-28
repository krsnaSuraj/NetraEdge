import json

cells = []

# Cell 0: Markdown header
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "# NetraEdge Training\n",
    "1. Runtime > Change runtime type > **GPU**\n",
    "2. Runtime > Run all\n",
    "3. Download .onnx files from output"
]})

# Cell 1: Setup
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "!pip install -q onnx onnxscript tqdm\n",
    "import torch, torch.nn as nn, torch.nn.functional as F\n",
    "import os, time, random\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print('Device:', DEVICE)\n",
    "if DEVICE.type == 'cuda':\n",
    "    props = torch.cuda.get_device_properties(0)\n",
    "    print('GPU:', props.name)\n",
    "    vram_gb = round(props.total_memory / 1e9, 1)\n",
    "    print('VRAM:', vram_gb, 'GB')"
], "outputs": [], "execution_count": None})

# Cell 2: MobileFaceNet
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "class DWSep(nn.Module):\n",
    "    def __init__(s, ic, oc, st=1):\n",
    "        super().__init__()\n",
    "        s.dw = nn.Conv2d(ic, ic, 3, st, 1, groups=ic, bias=False)\n",
    "        s.b1 = nn.BatchNorm2d(ic)\n",
    "        s.pw = nn.Conv2d(ic, oc, 1, bias=False)\n",
    "        s.b2 = nn.BatchNorm2d(oc)\n",
    "    def forward(s, x):\n",
    "        return F.relu(s.b2(s.pw(F.relu(s.b1(s.dw(x))))))\n",
    "\n",
    "class SE(nn.Module):\n",
    "    def __init__(s, ch, r=4):\n",
    "        super().__init__()\n",
    "        m = max(ch // r, 8)\n",
    "        s.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n",
    "            nn.Linear(ch, m), nn.ReLU(True), nn.Linear(m, ch), nn.Sigmoid())\n",
    "    def forward(s, x):\n",
    "        return x * s.se(x).unsqueeze(-1).unsqueeze(-1)\n",
    "\n",
    "class MB(nn.Module):\n",
    "    def __init__(s, ic, oc, st=1):\n",
    "        super().__init__()\n",
    "        m = ic * 2\n",
    "        s.ex = nn.Sequential(nn.Conv2d(ic, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))\n",
    "        s.dw = DWSep(m, oc, st)\n",
    "        s.se = SE(oc)\n",
    "        s.res = (st == 1 and ic == oc)\n",
    "    def forward(s, x):\n",
    "        o = s.se(s.dw(s.ex(x)))\n",
    "        return o + x if s.res else o\n",
    "\n",
    "class MobileFaceNet(nn.Module):\n",
    "    def __init__(s, ed=128):\n",
    "        super().__init__()\n",
    "        s.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))\n",
    "        s.blk = nn.Sequential(\n",
    "            MB(64, 64), MB(64, 128, 2), MB(128, 128),\n",
    "            MB(128, 256, 2), MB(256, 256), MB(256, 256),\n",
    "            MB(256, 512, 2), MB(512, 512), MB(512, 512))\n",
    "        s.fin = nn.Sequential(\n",
    "            nn.Conv2d(512, 512, 3, groups=512, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),\n",
    "            nn.Conv2d(512, ed, 1, bias=False), nn.BatchNorm2d(ed))\n",
    "        for m in s.modules():\n",
    "            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')\n",
    "            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)\n",
    "    def forward(s, x):\n",
    "        x = s.fin(s.blk(s.stem(x)))\n",
    "        return F.normalize(x.view(x.size(0), -1), p=2, dim=1)\n",
    "\n",
    "m = MobileFaceNet(128)\n",
    "print('MobileFaceNet:', sum(p.numel() for p in m.parameters()), 'params')"
], "outputs": [], "execution_count": None})

# Cell 3: Dataset
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "class FaceDataset(Dataset):\n",
    "    def __init__(s, n_id=50, n_img=20, aug=True):\n",
    "        s.samples = []\n",
    "        s.aug = aug\n",
    "        for i in range(n_id):\n",
    "            rng = np.random.RandomState(i * 1000)\n",
    "            base = rng.rand(112, 112, 3).astype(np.float32) * 0.3 + 0.35\n",
    "            base[35:50, 35:55, :] *= 0.7\n",
    "            base[35:50, 60:80, :] *= 0.7\n",
    "            base[45:65, 50:65, :] += 0.05\n",
    "            base[70:85, 40:75, 0] += 0.1\n",
    "            for j in range(n_img):\n",
    "                face = base + rng.randn(112, 112, 3).astype(np.float32) * 0.05\n",
    "                s.samples.append((np.clip(face, 0, 1), i))\n",
    "        print(len(s.samples), 'images,', n_id, 'identities')\n",
    "    def __len__(s): return len(s.samples)\n",
    "    def __getitem__(s, idx):\n",
    "        face, label = s.samples[idx]\n",
    "        arr = face.copy()\n",
    "        if s.aug:\n",
    "            if random.random() > 0.5: arr = np.clip(arr + np.random.uniform(-0.08, 0.08), 0, 1)\n",
    "            if random.random() > 0.5: arr = np.flip(arr, axis=1).copy()\n",
    "        t = torch.from_numpy(arr).permute(2, 0, 1).float()\n",
    "        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)\n",
    "        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)\n",
    "        return (t - mean) / std, label\n",
    "\n",
    "tr = FaceDataset(50, 20, True)\n",
    "va = FaceDataset(50, 20, False)\n",
    "tr_dl = DataLoader(tr, 32, shuffle=True, num_workers=2, pin_memory=True)\n",
    "va_dl = DataLoader(va, 32, shuffle=False, num_workers=2)\n",
    "print('Train:', len(tr), 'Val:', len(va))"
], "outputs": [], "execution_count": None})

# Cell 4: Train recognition
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "from tqdm import tqdm\n",
    "\n",
    "model = MobileFaceNet(128).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n",
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)\n",
    "best = 0\n",
    "\n",
    "for ep in range(10):\n",
    "    t0 = time.time()\n",
    "    model.train()\n",
    "    tc = tt = 0\n",
    "    for x, y in tqdm(tr_dl, desc='Ep ' + str(ep+1) + '/10'):\n",
    "        x, y = x.to(DEVICE), y.to(DEVICE)\n",
    "        optimizer.zero_grad()\n",
    "        out = model(x)\n",
    "        loss = criterion(out, y)\n",
    "        loss.backward()\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n",
    "        optimizer.step()\n",
    "        tc += (out.argmax(1) == y).sum().item()\n",
    "        tt += y.size(0)\n",
    "    model.eval()\n",
    "    vc = vt = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in va_dl:\n",
    "            x, y = x.to(DEVICE), y.to(DEVICE)\n",
    "            vc += (model(x).argmax(1) == y).sum().item()\n",
    "            vt += y.size(0)\n",
    "    ta = 100*tc/tt\n",
    "    va_acc = 100*vc/vt\n",
    "    elapsed = int(time.time()-t0)\n",
    "    scheduler.step()\n",
    "    print('Ep ' + str(ep+1) + ': Train ' + str(round(ta,1)) + '% Val ' + str(round(va_acc,1)) + '% ' + str(elapsed) + 's')\n",
    "    if va_acc > best:\n",
    "        best = va_acc\n",
    "        torch.save(model.state_dict(), 'rec_best.pt')\n",
    "print('Best:', round(best,1), '%')"
], "outputs": [], "execution_count": None})

# Cell 5: Export recognition
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "model.cpu().eval()\n",
    "torch.onnx.export(model, torch.randn(1, 3, 112, 112),\n",
    "    'face_recognition.onnx', input_names=['input'],\n",
    "    output_names=['output'], opset_version=13, dynamo=False)\n",
    "sz = os.path.getsize('face_recognition.onnx') / 1e6\n",
    "print('face_recognition.onnx:', round(sz, 1), 'MB')\n",
    "from google.colab import files\n",
    "files.download('face_recognition.onnx')"
], "outputs": [], "execution_count": None})

# Cell 6: LivenessCNN
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "class LivBlock(nn.Module):\n",
    "    def __init__(s, ic, oc, st=1):\n",
    "        super().__init__()\n",
    "        s.dw = nn.Conv2d(ic, ic, 3, st, 1, groups=ic, bias=False)\n",
    "        s.b1 = nn.BatchNorm2d(ic)\n",
    "        s.pw = nn.Conv2d(ic, oc, 1, bias=False)\n",
    "        s.b2 = nn.BatchNorm2d(oc)\n",
    "    def forward(s, x):\n",
    "        return F.relu(s.b2(s.pw(F.relu(s.b1(s.dw(x))))))\n",
    "\n",
    "class LivenessCNN(nn.Module):\n",
    "    def __init__(s, nc=3, dp=0.3):\n",
    "        super().__init__()\n",
    "        s.feat = nn.Sequential(\n",
    "            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),\n",
    "            LivBlock(32, 64), LivBlock(64, 128, 2), LivBlock(128, 256, 2), LivBlock(256, 256, 2))\n",
    "        s.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n",
    "            nn.Dropout(dp), nn.Linear(256, 64), nn.ReLU(True),\n",
    "            nn.Dropout(dp * 0.5), nn.Linear(64, nc))\n",
    "        for m in s.modules():\n",
    "            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')\n",
    "            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)\n",
    "    def forward(s, x):\n",
    "        return s.cls(s.feat(x))\n",
    "\n",
    "class LivDataset(Dataset):\n",
    "    def __init__(s, n=3000):\n",
    "        s.n = n\n",
    "        s.bases = []\n",
    "        for i in range(100):\n",
    "            rng = np.random.RandomState(i * 777)\n",
    "            face = rng.rand(112, 112, 3).astype(np.float32) * 0.4 + 0.3\n",
    "            face[35:50, 35:55, :] *= 0.7\n",
    "            face[35:50, 60:80, :] *= 0.7\n",
    "            face[70:85, 40:75, 0] += 0.1\n",
    "            s.bases.append(face)\n",
    "    def __len__(s): return s.n\n",
    "    def __getitem__(s, idx):\n",
    "        label = idx % 3\n",
    "        base = s.bases[idx % len(s.bases)]\n",
    "        arr = base.copy()\n",
    "        rng = np.random.RandomState(idx)\n",
    "        if label == 1:\n",
    "            arr = np.clip(arr * 1.05 + 0.02, 0, 1)\n",
    "            k = np.ones(5) / 5\n",
    "            arr = np.stack([np.convolve(arr[:,:,c].flatten(), k, mode='same').reshape(112,112) for c in range(3)], -1)\n",
    "        elif label == 2:\n",
    "            for y in range(0, 112, 2): arr[y,:,:] *= 0.88\n",
    "            arr[:,:,2] *= 1.05\n",
    "            arr = arr * 0.85 + 0.08\n",
    "        arr = np.clip(arr + rng.randn(112, 112, 3).astype(np.float32) * 0.02, 0, 1)\n",
    "        if random.random() > 0.5: arr = np.flip(arr, axis=1).copy()\n",
    "        t = torch.from_numpy(arr).permute(2, 0, 1).float()\n",
    "        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)\n",
    "        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)\n",
    "        return (t - mean) / std, label\n",
    "\n",
    "print('LivenessCNN:', sum(p.numel() for p in LivenessCNN(3).parameters()), 'params')"
], "outputs": [], "execution_count": None})

# Cell 7: Train liveness
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "tr2 = LivDataset(3000)\n",
    "va2 = LivDataset(600)\n",
    "tr2_dl = DataLoader(tr2, 64, shuffle=True, num_workers=2, pin_memory=True)\n",
    "va2_dl = DataLoader(va2, 64, shuffle=False, num_workers=2)\n",
    "\n",
    "livmodel = LivenessCNN(3).to(DEVICE)\n",
    "w = torch.tensor([1.0, 1.5, 1.5]).to(DEVICE)\n",
    "criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)\n",
    "optimizer = torch.optim.AdamW(livmodel.parameters(), lr=5e-4, weight_decay=1e-4)\n",
    "scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)\n",
    "best = 0\n",
    "\n",
    "for ep in range(10):\n",
    "    t0 = time.time()\n",
    "    livmodel.train()\n",
    "    tc = tt = 0\n",
    "    for x, y in tqdm(tr2_dl, desc='Ep ' + str(ep+1) + '/10'):\n",
    "        x, y = x.to(DEVICE), y.to(DEVICE)\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(livmodel(x), y)\n",
    "        loss.backward()\n",
    "        torch.nn.utils.clip_grad_norm_(livmodel.parameters(), 5.0)\n",
    "        optimizer.step()\n",
    "        tc += (livmodel(x).argmax(1) == y).sum().item()\n",
    "        tt += y.size(0)\n",
    "    livmodel.eval()\n",
    "    vc = vt = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in va2_dl:\n",
    "            x, y = x.to(DEVICE), y.to(DEVICE)\n",
    "            vc += (livmodel(x).argmax(1) == y).sum().item()\n",
    "            vt += y.size(0)\n",
    "    ta = 100*tc/tt\n",
    "    va_acc = 100*vc/vt\n",
    "    elapsed = int(time.time()-t0)\n",
    "    scheduler.step()\n",
    "    print('Ep ' + str(ep+1) + ': Train ' + str(round(ta,1)) + '% Val ' + str(round(va_acc,1)) + '% ' + str(elapsed) + 's')\n",
    "    if va_acc > best:\n",
    "        best = va_acc\n",
    "        torch.save(livmodel.state_dict(), 'liv_best.pt')\n",
    "print('Best:', round(best,1), '%')"
], "outputs": [], "execution_count": None})

# Cell 8: Export liveness
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "livmodel.cpu().eval()\n",
    "torch.onnx.export(livmodel, torch.randn(1, 3, 112, 112),\n",
    "    'liveness_detector.onnx', input_names=['input'],\n",
    "    output_names=['output'], opset_version=13, dynamo=False)\n",
    "sz = os.path.getsize('liveness_detector.onnx') / 1e6\n",
    "print('liveness_detector.onnx:', round(sz, 1), 'MB')\n",
    "from google.colab import files\n",
    "files.download('liveness_detector.onnx')"
], "outputs": [], "execution_count": None})

# Cell 9: Summary
cells.append({"cell_type": "code", "metadata": {}, "source": [
    "r = os.path.getsize('face_recognition.onnx') / 1e6\n",
    "l = os.path.getsize('liveness_detector.onnx') / 1e6\n",
    "print('face_recognition.onnx:', round(r, 1), 'MB')\n",
    "print('liveness_detector.onnx:', round(l, 1), 'MB')\n",
    "print('Total:', round(r + l, 1), 'MB')\n",
    "print('Place in F:/PROJECTS/NetraEdge/models/')"
], "outputs": [], "execution_count": None})

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "cells": cells
}

path = r'F:\PROJECTS\NetraEdge\training\NetraEdge_Train.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Verify
with open(path) as f:
    data = f.read()
    nb_check = json.loads(data)

# Final verification
errors = []
for i, cell in enumerate(nb_check['cells']):
    src = ''.join(cell.get('source', []))
    if 'total_mem' in src and 'total_memory' not in src:
        errors.append(f'Cell {i}: has total_mem')
    if 'total_memoryory' in src:
        errors.append(f'Cell {i}: has total_memoryory')

if errors:
    print('ERRORS:', errors)
else:
    print('NetraEdge_Train.ipynb created - NO ERRORS')
    print('Cells:', len(nb_check['cells']))
