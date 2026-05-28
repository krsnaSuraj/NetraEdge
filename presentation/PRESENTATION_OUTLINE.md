# NetraEdge — Hackathon 7.0 Presentation

## Slide 1: Title
**NetraEdge**
Offline Facial Recognition & Liveness Detection
for Zero-Network Environments

NHAI Hackathon 7.0 | Datalake 3.0 Integration
Team: krsnaSuraj

---

## Slide 2: The Problem

**Field personnel in remote locations need secure authentication — with zero internet.**

- Datalake 3.0 operates in areas with no cellular/WiFi coverage
- Current methods: manual attendance, easily spoofed
- Challenge: accurate face recognition + anti-spoofing on mid-range phones
- Constraints: <20MB model, <1s inference, React Native, Android 8+ & iOS 12+

---

## Slide 3: Our Solution — NetraEdge

**Multi-modal offline face verification with 3-layer liveness defense**

| Feature | Implementation |
|---------|---------------|
| Face Recognition | MobileFaceNet (TFLite INT8) |
| Liveness | Blink + Texture CNN + Depth Estimation |
| Storage | SQLite (local, persists offline) |
| Sync | Queue-based offline→online with AWS |
| Platform | React Native (Android + iOS) |

---

## Slide 4: System Architecture

```
┌─────────────────────────────────────────────────┐
│                   React Native App               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │EnrollScreen│  │VerifyScreen│  │  AppProvider │  │
│  └─────┬────┘  └─────┬────┘  └──────┬───────┘  │
│        │              │              │           │
│  ┌─────┴──────────────┴──────────────┴───────┐  │
│  │            FacePipeline (Core)             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐  │  │
│  │  │ Encoder  │ │ Liveness │ │   Store   │  │  │
│  │  │(MobileFN)│ │Orchestr. │ │ (SQLite)  │  │  │
│  │  └──────────┘ └──────────┘ └───────────┘  │  │
│  └────────────────────┬──────────────────────┘  │
│                       │                          │
│  ┌────────────────────┴──────────────────────┐  │
│  │  SyncManager → Queue → AWS Transport      │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Slide 5: Model Architecture

**Two lightweight models, one inference pipeline**

| Model | Size (INT8) | Input | Output | Purpose |
|-------|-------------|-------|--------|---------|
| MobileFaceNet | ~4.8MB | 112×112×3 | 128-d vector | Face encoding |
| LivenessCNN | ~2.8MB | 112×112×3 | [real, print, screen] | Spoof detection |
| **Total** | **~7.6MB** | | | **4× under 20MB target** |

Compression: FP32 → INT8 post-training quantization (4× reduction, minimal accuracy loss)

---

## Slide 6: Multi-Modal Liveness Detection

**3 independent checks — 2 must pass for LIVE verdict**

1. **Blink Detection** (Temporal)
   - Eye Aspect Ratio (EAR) using 12 MediaPipe face mesh landmarks
   - EAR < 0.21 for 100ms = blink detected
   - Requires natural eye movement — photos/screens can't blink

2. **Texture Analysis** (Per-frame CNN)
   - LivenessCNN classifies: real skin vs printed photo vs screen
   - Real skin has pores, specular highlights, micro-wrinkles
   - Screens have Moiré patterns, uniform backlight

3. **Depth Estimation** (3D Mesh)
   - Variance of z-coordinates across 478 face mesh points
   - Variance > 0.25 = 3D face (vs flat photo/screen)

---

## Slide 7: Architecture Quality

**Production-grade TypeScript monorepo**

- **Functional error handling** — `Result<T, E>` pattern (no thrown exceptions)
- **Dependency injection** — Interfaces for encoder, store, liveness (testable)
- **Clean separation** — Core (pure TS) → React Native (bridge) → App (UI)
- **Type safety** — TypeScript strict mode, zero `any` types
- **Test coverage** — 86 unit tests across 12 test suites (all passing)
- **CI/CD** — GitHub Actions (lint + typecheck + test)

---

## Slide 8: React Native Integration

**Seamless Datalake 3.0 integration**

```typescript
// 3 lines to integrate into existing app
import { AppProvider, EnrollScreen, VerifyScreen } from '@netraedge/react-native';

<AppProvider>
  <EnrollScreen onBack={handleBack} />
</AppProvider>
```

| Layer | Package | Purpose |
|-------|---------|---------|
| `@netraedge/core` | Pure TypeScript | Business logic, zero native deps |
| `@netraedge/react-native` | RN bridge | Hooks, components, native module |
| Native (Kotlin/Swift) | TFLite | On-device inference |

---

## Slide 9: Sync & Purge Mechanism

**Offline-first with automatic cloud sync**

```
Offline Mode:
  Enroll → Encode → Store (SQLite) → Queue (pending sync)

Online Mode:
  Queue → NetworkMonitor detects connection
       → SyncManager uploads batch to AWS
       → DataPurge removes synced local data
       → Event emitted (sync_complete)
```

- Queue-based: FIFO ordering, retry tracking (max 3 retries)
- Network-aware: auto-syncs when connectivity restored
- Purge policy: configurable (after sync, max age, max records)
- Transport: REST API (compatible with API Gateway + Lambda)

---

## Slide 10: Target Performance

| Metric | Requirement | Target | Basis |
|--------|-------------|--------|-------|
| Model Size | <20MB | ~7.6MB | INT8 quantization |
| Inference | <1s | <400ms | TFLite on-device |
| Recognition | >95% | >99% | MobileFaceNet (LFW benchmark) |
| Liveness | >95% | >95% | Multi-modal 2/3 voting |
| Min RAM | 3GB | 3GB | Lightweight pipeline |

*Benchmarks to be measured after model training. Device-specific numbers pending.*

---

## Slide 11: Innovation Highlights

1. **Multi-modal liveness** — Not just blink. 3 independent signals (temporal + visual + spatial)
2. **INT8 edge compression** — 4× size reduction, runs on any mid-range phone
3. **Functional architecture** — Result pattern eliminates runtime crashes
4. **Offline-first sync** — Queue + purge ensures no data loss, no duplication
5. **Indian demographics focus** — Training pipeline designed for diverse skin tones and outdoor lighting

---

## Slide 12: Open Source & Compliance

| Requirement | Status |
|-------------|--------|
| Open source only | MIT License, all dependencies |
| React Native | Cross-platform (Android + iOS) |
| No proprietary models | MobileFaceNet + custom LivenessCNN |
| Source code | GitHub repository |
| Documentation | Architecture, API, Integration guides |

---

## Slide 13: Demo Flow

1. **Enroll** — Enter user ID → Camera captures 10 face frames → Encode → Store
2. **Verify** — Camera detects face → Blink challenge → Liveness passes → Match against enrolled → Result
3. **Sync** — When network available → Queue drains to server → Local data purged

---

## Slide 14: Future Scope

- Federated learning for model improvement across devices
- Additional liveness challenges (smile, head turn, speech)
- Model fine-tuning per deployment region
- Biometric template encryption at rest
- Integration with Datalake 3.0 attendance module

---

## Slide 15: Thank You

**NetraEdge** — Secure. Offline. Lightweight.

GitHub: github.com/krsnaSuraj/NetraEdge
NHAI Hackathon 7.0 | Datalake 3.0 Integration
