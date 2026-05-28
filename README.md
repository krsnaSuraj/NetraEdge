# NetraEdge

**Offline Facial Recognition & Liveness Detection for Zero-Network Environments**

A lightweight, secure, and entirely offline facial recognition system with multi-modal liveness detection, designed for integration into NHAI Datalake 3.0.

## Features

- **Offline-first** — 100% on-device inference, no internet required
- **Multi-modal liveness** — Blink detection + texture analysis + depth estimation
- **Ultra-lightweight** — 7.6MB total model size (4× under 20MB target)
- **Fast** — <400ms total inference time on mid-range devices
- **Cross-platform** — React Native (Android + iOS)
- **Indian demographics** — Trained on diverse Indian face datasets
- **Open source** — All MIT/Apache licensed, no proprietary dependencies

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Mobile | React Native v5 + Vision Camera v5 |
| Face Detection | Google ML Kit (BlazeFace) |
| Face Recognition | MobileFaceNet (TFLite, INT8) |
| Liveness Detection | Custom CNN + MediaPipe Face Mesh |
| Core Logic | Pure TypeScript (no native deps) |
| Storage | SQLite (local) |
| Sync | AWS SDK (offline-to-online) |

## Quick Start

```bash
# Clone
git clone https://github.com/krsnaSuraj/netraedge.git
cd netraedge

# Install
npm install

# Build
npm run build

# Run
cd packages/app
npx react-native run-android
```

## Project Structure

```
netraedge/
├── packages/
│   ├── core/                 # Pure TypeScript logic
│   ├── react-native/         # React Native bridge
│   └── app/                  # Demo application
├── training/                 # Python ML pipeline
├── models/                   # Exported TFLite models
├── docs/                     # Documentation
└── presentation/             # Hackathon presentation
```

## Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Model Size | <20MB | **7.6MB** |
| Inference Time | <1s | **400ms** |
| Recognition Accuracy | >95% | **99.5%** |
| Liveness Accuracy | >95% | **96%** |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Integration Guide](docs/INTEGRATION.md)
- [Benchmarks](docs/BENCHMARKS.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
