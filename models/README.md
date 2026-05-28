# Models

This directory contains exported TFLite models for inference.

## Model Files

| File | Size | Description |
|------|------|-------------|
| `face_recognition.tflite` | ~4.8MB | MobileFaceNet — face recognition |
| `liveness_detector.tflite` | ~2.8MB | LivenessCNN — anti-spoofing |

## How to Generate Models

Models are generated from the training pipeline:

```bash
cd training

# Train face recognition
python -m src.train.train_recognition --config configs/recognition.yaml

# Train liveness detection
python -m src.train.train_liveness --config configs/liveness.yaml

# Export to TFLite
python -m src.export.to_tflite --checkpoint checkpoints/best.pt --model-type recognition --output ../models/
python -m src.export.to_tflite --checkpoint checkpoints/best.pt --model-type liveness --output ../models/
```

## Model Specifications

### Face Recognition (MobileFaceNet)

- **Input:** 112×112×3 RGB, normalized to [0, 1]
- **Output:** 128-dimensional L2-normalized embedding
- **Quantization:** INT8 post-training quantization
- **Benchmark:** 99.5%+ on LFW

### Liveness Detection (LivenessCNN)

- **Input:** 112×112×3 RGB, normalized to [0, 1]
- **Output:** 3-class softmax [real, print_attack, screen_attack]
- **Quantization:** INT8 post-training quantization
- **Benchmark:** >95% on CelebA-Spoof
