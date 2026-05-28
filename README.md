# NetraEdge

**Offline Facial Recognition & Liveness Detection for Zero-Network Environments**

A lightweight, secure, and entirely offline facial recognition system with multi-modal liveness detection, designed for integration into NHAI Datalake 3.0.

## Features

- **Offline-first** — 100% on-device inference, no internet required
- **Multi-modal liveness** — Blink detection + texture analysis + depth estimation
- **Ultra-lightweight** — Target <20MB total model size (INT8 quantized)
- **Fast** — Target <1s total inference on mid-range devices
- **Cross-platform** — React Native (Android + iOS)
- **Indian demographics** — Architecture designed for diverse Indian face datasets
- **Open source** — All MIT licensed, no proprietary dependencies

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile | React Native 0.76+ + Vision Camera v5 |
| Face Detection | Google ML Kit (BlazeFace) |
| Face Recognition | MobileFaceNet (TFLite, INT8) |
| Liveness Detection | Custom CNN + EAR Blink + Depth Estimation |
| Core Logic | Pure TypeScript (no native deps) |
| Storage | In-memory + SQLite-ready |
| Sync | Queue-based offline-to-online with REST transport |

## Quick Start

```bash
# Clone
git clone https://github.com/krsnaSuraj/NetraEdge.git
cd NetraEdge

# Install
npm install

# Train models (requires Google Colab with T4 GPU)
# Upload training/NetraEdge_Train.ipynb to Colab and run all cells
# Download the generated .tflite files

# Copy models to Android assets
cp models/*.tflite packages/app/android/app/src/main/assets/

# Run
cd packages/app
npx react-native run-android
```

## Project Structure

```
NetraEdge/
├── packages/
│   ├── core/                 # Pure TypeScript — types, pipeline, sync
│   ├── react-native/         # RN hooks, components, native bridge (Kotlin + Swift)
│   └── app/                  # Demo application
├── training/                 # Python ML pipeline (PyTorch → ONNX → TFLite)
├── models/                   # Exported model artifacts
├── docs/                     # Architecture, API, Integration, Benchmarks
└── presentation/             # Hackathon presentation outline
```

## Target Performance

| Metric | Target | Basis |
|--------|--------|-------|
| Model Size | <20MB | INT8 quantization (MobileFaceNet ~4.8MB + LivenessCNN ~2.8MB) |
| Inference Time | <1s | On-device TFLite inference, no network round-trip |
| Recognition Accuracy | >95% | MobileFaceNet on LFW benchmark (literature: 99.5%) |
| Liveness Accuracy | >95% | Multi-modal (blink + texture + depth) reduces spoof risk |

*Benchmarks will be measured after model training. See [BENCHMARKS.md](docs/BENCHMARKS.md) for methodology.*

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design with Mermaid diagrams
- [API Reference](docs/API.md) — Complete TypeScript API docs
- [Integration Guide](docs/INTEGRATION.md) — Step-by-step Datalake 3.0 integration
- [Benchmarks](docs/BENCHMARKS.md) — Performance testing methodology

## License

MIT License — see [LICENSE](LICENSE) for details.
