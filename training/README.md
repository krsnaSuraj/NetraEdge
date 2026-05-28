# NetraEdge Training

## Google Colab (Recommended)

Upload `notebooks/train_recognition.py` to Google Colab as a notebook.
Runtime: GPU (T4). Estimated time: ~2 hours.

## Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Download LFW dataset
python download_datasets.py --lfw

# Train face recognition
python -m src.train.train_recognition --config configs/recognition.yaml

# Train liveness detection
python -m src.train.train_liveness --config configs/liveness.yaml

# Export to TFLite
python -m src.export.to_tflite --checkpoint checkpoints/best.pt --model-type recognition
```

## Model Output

| Model | Format | Size | Location |
|-------|--------|------|----------|
| Face Recognition | `.pt` | ~4MB | `checkpoints/recognition_best.pt` |
| Liveness Detection | `.pt` | ~3MB | `checkpoints/liveness_best.pt` |
| Face Recognition | `.tflite` | ~4.8MB | `models/face_recognition.tflite` |
| Liveness Detection | `.tflite` | ~2.8MB | `models/liveness_detector.tflite` |

## Total Model Size

~7.6MB (INT8 quantized) — well under 20MB target.
