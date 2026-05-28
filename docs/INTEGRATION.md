# NetraEdge — Integration Guide

## Quick Start

```bash
# Install dependencies
npm install

# Build core package
npm run build --workspace=@netraedge/core

# Run on Android
cd packages/app
npx react-native run-android
```

## Integration into Datalake 3.0

### Step 1: Install Package

```bash
npm install @netraedge/core @netraedge/react-native
```

### Step 2: Copy Models

Copy the TFLite models to your app's assets:

```bash
cp models/face_recognition.tflite android/app/src/main/assets/
cp models/liveness_detector.tflite android/app/src/main/assets/
```

### Step 3: Initialize Pipeline

```typescript
import { FacePipeline, TFLiteEncoder, InMemoryEmbeddingStore, LivenessOrchestrator, StubTextureAnalyzer } from '@netraedge/core';

// Initialize components
const encoder = new TFLiteEncoder(nativeModule);
const store = new InMemoryEmbeddingStore(); // or SQLite-backed
const textureAnalyzer = new StubTextureAnalyzer(); // replace with TFLite
const liveness = new LivenessOrchestrator(textureAnalyzer);

// Create pipeline
const pipeline = new FacePipeline(encoder, store, liveness);
```

### Step 4: Add Camera

```typescript
import { FaceCamera, useFaceDetection, useLivenessCheck } from '@netraedge/react-native';

function AttendanceScreen() {
  const { state: detectionState, onFacesDetected } = useFaceDetection();
  const { state: livenessState, processFrame } = useLivenessCheck();

  return (
    <FaceCamera
      onFaceDetected={(faces) => {
        onFacesDetected(faces);
        if (faces.length > 0) {
          processFrame(faces[0].landmarks);
        }
      }}
      isActive={true}
    />
  );
}
```

### Step 5: Enroll & Verify

```typescript
// Enroll
await pipeline.enroll('field_agent_001', faceFrames, meshPoints, Date.now());

// Verify
const result = await pipeline.verify(faceData, meshPoints, Date.now());
if (result.ok && result.value.matched) {
  // Access granted
}
```

### Step 6: Sync Configuration

```typescript
import { DefaultSyncManager } from '@netraedge/core';

const syncManager = new DefaultSyncManager(transport, networkMonitor);
syncManager.start(); // Begins periodic sync checks
```

## API Reference

See [API.md](./API.md) for complete API documentation.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for system design details.

## Benchmarks

See [BENCHMARKS.md](./BENCHMARKS.md) for performance data.
