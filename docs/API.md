# NetraEdge — API Reference

> Reference for the TypeScript / React Native integration. The shipped Android APK uses the same pipeline directly via the Kotlin module — see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. React Native Bridge API

### `NetraEdge.initialize()`

Load TFLite runtime, init MediaPipe, decrypt AES-256-GCM model cache. Must be called before any other method.

```typescript
import NetraEdge from '@netraedge/react-native';

await NetraEdge.initialize();
// → { success: true, modelVersion: '1.0.0', embeddingDim: 128, livenessLayers: 10 }
```

### `NetraEdge.enroll(userId, options?)`

Capture 15 quality-weighted frames, run 10-layer liveness + 3 randomized active challenges, generate 128-d embedding, encrypt + store locally.

```typescript
const result = await NetraEdge.enroll('worker_001', { activeChallengeTimeoutMs: 25_000 });
// → {
//     success: true,
//     embedding: Float32Array(128),
//     livenessScore: 0.94,
//     bpm: 72,
//     framesProcessed: 15,
//     challengesCompleted: ['BLINK', 'HEAD_TURN_LEFT', 'SMILE'],
//   }
```

### `NetraEdge.verify(userId)`

Capture frames, run 10-layer liveness + 3 randomized active challenges, cosine-match against stored embedding.

```typescript
const result = await NetraEdge.verify('worker_001');
// → {
//     verified: true,
//     similarity: 0.87,
//     livenessScore: 0.96,
//     bpm: 71,
//     spoof: false,
//     decision: 'VERIFIED',  // 'VERIFIED' | 'SPOOF' | 'NOT_RECOGNIZED'
//     challengesCompleted: ['SMILE', 'HEAD_TURN_RIGHT'],
//   }
```

### `NetraEdge.reset()`

Wipe local embedding cache for the current user.

```typescript
await NetraEdge.reset();
// → { success: true, recordsDeleted: 1 }
```

### `NetraEdge.syncNow()`

Force sync of local cache to Datalake 3.0 endpoint. Local cache is auto-purged on 200 OK.

```typescript
const result = await NetraEdge.syncNow();
// → { success: true, recordsUploaded: 1, recordsPurged: 1, retries: 0, durationMs: 320 }
```

### `NetraEdge.close()`

Release TFLite interpreters, MediaPipe graph, and native resources. Call on app shutdown.

---

## 2. TypeScript Core API

### `FacePipeline`

```typescript
import { FacePipeline, TFLiteEncoder, InMemoryEmbeddingStore, SOTALivenessOrchestrator } from '@netraedge/core';

const pipeline = new FacePipeline(
  new TFLiteEncoder(modelPath),
  new InMemoryEmbeddingStore(),
  new SOTALivenessOrchestrator(/* …10-layer… */),
  {
    recognitionThreshold: 0.62,
    enrollmentFrameCount: 10,
    requireLiveness: true,
    activeChallengeCount: 2,
  },
);

const enrollResult = await pipeline.enroll(
  'user_001',
  [faceFrame1, faceFrame2, /* … */],
  [meshPoints1, meshPoints2 /* … */],
  Date.now(),
);

const verifyResult = await pipeline.verify(faceFrame, meshPoints, Date.now());
```

### `LivenessOrchestrator` (10-layer, all algorithmic)

> **All 10 layers are algorithmic.** The shipped v1.0.0 uses LBP/HSV/FFT/Laplacian/optical-flow/cheek-delta/sensor-fusion/gradient-hist/MediaPipe-blendshapes/POS-rPPG. `liveness_detector.tflite` (MiniFASNet) is **loaded but bypassed** — kept as a backup / future-toggle for A/B testing on Indian demographics.

```typescript
const liveness = new SOTALivenessOrchestrator(
  textureAnalyzer,        // L1: LBP histogram
  colorAnalyzer,          // L2: HSV + luma entropy
  moireDetector,          // L3: radix-2 FFT moiré
  specularDetector,       // L4: Laplacian specular
  lightAnalyzer,          // L5: cheek luma delta ≤ 0.15
  temporalAnalyzer,       // L6: optical flow
  sensorFusion,           // L7: gyro + accel
  bandingDetector,        // L8: gradient histogram
  activeChallenge,        // L9: BLINK / SMILE / HEAD_TURN — 3 of 4 shuffled per session (hard gate)
  rppgDetector,           // L10: POS pulse, 8-s window
  {
    fusionAlpha: 0.15,           // EMA smoothing
    gracePeriodMs: 3000,         // 3 s at start of verify
    vetoThresholds: { rppgMinSeconds: 8, moireScoreMax: 0.75, bandingScoreMax: 0.75 },
  },
);

const result = await liveness.processFrame(faceData, meshPoints, timestamp, sensorData);
// → { verdict: 'live' | 'spoof' | 'unknown', confidence: 0.94, bpm: 72, passedChecks: [...], failedChecks: [...] }

// Active challenge
liveness.setActiveChallenges(['BLINK', 'SMILE']);  // 2 random per session
liveness.onChallengeCompleted = (step, success) => { /* … */ };
```

### `SyncManager`

```typescript
import { DefaultSyncManager } from '@netraedge/core';

const sync = new DefaultSyncManager({
  endpoint: 'https://api.datalake.nhai.gov.in/v1/face-sync',
  batchSize: 50,
  maxRetries: 6,
  backoffMs: [1000, 2000, 4000, 8000, 16000, 30000],
  differentialPrivacy: { enabled: true, epsilon: 1.0, sensitivity: 2.0 },
});

await sync.start();
await sync.enqueue({ eventType: 'enroll', userId: 'worker_001', embedding, livenessScore: 0.94 });
sync.status;  // 'idle' | 'syncing' | 'paused' | 'error'
sync.on('sync', e => console.log(`synced ${e.count} records in ${e.durationMs}ms`));
```

### `DataPurge`

```typescript
import { DefaultDataPurge } from '@netraedge/core';

const purge = new DefaultDataPurge(store, sync);
await purge.purgeLocal();  // Wipes local data after sync ack, retains signed audit log
```

---

## 3. Error Codes

```typescript
import { ErrorCode } from '@netraedge/core';

enum ErrorCode {
  FACE_NOT_FOUND          = 'FACE_NOT_FOUND',
  MULTIPLE_FACES          = 'MULTIPLE_FACES',
  FACE_TOO_SMALL          = 'FACE_TOO_LOW_CONFIDENCE',
  QUALITY_TOO_LOW         = 'QUALITY_TOO_LOW',
  LIVENESS_FAILED         = 'LIVENESS_FAILED',
  CHALLENGE_TIMEOUT       = 'CHALLENGE_TIMEOUT',
  CHALLENGE_FAILED        = 'CHALLENGE_FAILED',
  MODEL_NOT_LOADED        = 'MODEL_NOT_LOADED',
  MODEL_INFERENCE_FAILED  = 'MODEL_INFERENCE_FAILED',
  ENROLLMENT_FAILED       = 'ENROLLMENT_FAILED',
  VERIFICATION_FAILED     = 'VERIFICATION_FAILED',
  STORAGE_ERROR           = 'STORAGE_ERROR',
  DECRYPTION_FAILED       = 'DECRYPTION_FAILED',
  SYNC_FAILED             = 'SYNC_FAILED',
  NETWORK_ERROR           = 'NETWORK_ERROR',
  INVALID_CONFIG          = 'INVALID_CONFIG',
  TAMPER_DETECTED         = 'TAMPER_DETECTED',
}
```

---

## 4. Configuration

```typescript
import { MODEL_CONFIG, RECOGNITION_THRESHOLDS, LIVENESS_THRESHOLDS, ACTIVE_CHALLENGE_CONFIG } from '@netraedge/core';

MODEL_CONFIG.recognition.inputWidth          // 112
MODEL_CONFIG.recognition.inputHeight         // 112
MODEL_CONFIG.recognition.embeddingDimension  // 128
MODEL_CONFIG.liveness.numClasses             // 3
MODEL_CONFIG.liveness.fusionAlpha            // 0.15

RECOGNITION_THRESHOLDS.matchConfidence       // 0.62
RECOGNITION_THRESHOLDS.minFaceSize           // 80
RECOGNITION_THRESHOLDS.maxRollAngle          // 15

LIVENESS_THRESHOLDS.blinkEAR                 // 0.22
LIVENESS_THRESHOLDS.smileScore               // 0.10
LIVENESS_THRESHOLDS.headYawScore             // 0.10
LIVENESS_THRESHOLDS.jawOpenScore             // 0.25
LIVENESS_THRESHOLDS.rppgMinSeconds           // 8.0
LIVENESS_THRESHOLDS.gracePeriodMs            // 3000

ACTIVE_CHALLENGE_CONFIG.timeoutMs            // 25000
ACTIVE_CHALLENGE_CONFIG.challengesPerSession // 2 (random pick from 4)
```

---

## 5. Benchmark JSON format

The committed `benchmarks/real_*.json` files contain raw measured data from the production model. The `vitest` unit tests under `packages/core/src/__tests__/` (121 / 121 passing) cover the deterministic parts of the pipeline.

```json
{
  "model": "face_recognition.tflite",
  "device": "Motorola G-series, Adreno 610, 4 GB RAM",
  "framework": "TFLite GPU FP16",
  "aggregate": {
    "latency_ms_mean":  11.2,
    "latency_ms_p95":    14.5,
    "similarity_mean":   0.87,
    "similarity_p5":     0.62,
    "n_samples":         1000
  },
  "per_condition": [
    { "condition": "outdoor_sunlight",  "latency_ms_mean": 11.4, "accuracy": 0.984 },
    { "condition": "indoor_office",     "latency_ms_mean": 10.8, "accuracy": 0.991 },
    { "condition": "low_light_evening", "latency_ms_mean": 12.1, "accuracy": 0.973 }
  ]
}
```

---

## 6. Where to look

| Concern | File |
|---------|------|
| Layer thresholds (matchConfidence, blinkEAR, …) | `packages/core/src/config/constants.ts` |
| 10-layer orchestration | `packages/app/android/app/src/main/java/com/netraedge/MainActivity.kt` |
| iOS pipeline (Swift, 386 LOC) | `packages/react-native/ios/NetraEdgeModule.swift` |
| Sync / DP / retry | `packages/core/src/sync/SyncManager.ts` + `DPNoise.ts` |
| Encrypted asset loader | `tools/encrypt_assets.py` + `EncryptedAssets.kt` |
| Honest model inventory | [`models/MODEL_CARD.md`](../models/MODEL_CARD.md) |

> **Note on training scripts.** v1.0.0 ships **no custom training pipeline** — all three production models are pre-trained Apache-2.0 weights. The 10-layer liveness stack is fully algorithmic. A post-hackathon Indian-demographic fine-tuning path is documented in `models/MODEL_CARD.md` § "Future work".
