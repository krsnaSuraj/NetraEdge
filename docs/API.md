# NetraEdge — API Reference

## Core Package (`@netraedge/core`)

### FacePipeline

Main orchestrator for face processing.

```typescript
class FacePipeline {
  constructor(
    encoder: Encoder,
    store: EmbeddingStore,
    liveness?: LivenessOrchestrator,
    config?: Partial<PipelineConfig>,
  );

  enroll(
    userId: string,
    faceFrames: Uint8Array[],
    meshPoints: Point3D[][],
    timestampMs: number,
  ): Promise<Result<EnrollmentResult>>;

  verify(
    faceData: Uint8Array,
    meshPoints: Point3D[],
    timestampMs: number,
    skipLiveness?: boolean,
  ): Promise<Result<VerificationResult>>;

  listUsers(): Promise<string[]>;
  removeUser(userId: string): Promise<boolean>;
  enrollmentCount(): Promise<number>;
  clearEnrollments(): Promise<void>;
  dispose(): void;
}
```

### Encoder

```typescript
interface Encoder {
  encode(faceData: Uint8Array): Float32Array | null;
  readonly isLoaded: boolean;
  dispose(): void;
}
```

### EmbeddingStore

```typescript
interface EmbeddingStore {
  enroll(userId: string, embeddings: Float32Array[]): Promise<boolean>;
  identify(probe: Float32Array, threshold: number): Promise<IdentificationResult | null>;
  listUsers(): Promise<string[]>;
  remove(userId: string): Promise<boolean>;
  count(): Promise<number>;
  clear(): Promise<void>;
  dispose(): void;
}
```

### LivenessOrchestrator

```typescript
class LivenessOrchestrator {
  constructor(
    textureAnalyzer: TextureAnalyzer,
    blinkDetector?: BlinkDetector,
    depthEstimator?: DepthEstimator,
    requiredChecks?: number,
  );

  processFrame(
    faceData: Uint8Array,
    meshPoints: Point3D[],
    timestampMs: number,
  ): LivenessResult;

  reset(): void;
  dispose(): void;
}
```

### BlinkDetector

```typescript
class BlinkDetector {
  constructor(
    maxHistory?: number,
    threshold?: number,
    minDuration?: number,
  );

  processFrame(
    meshPoints: Array<{ x: number; y: number }>,
    timestampMs: number,
  ): BlinkResult;

  readonly blinkCount: number;
  reset(): void;
}
```

### SyncManager

```typescript
interface SyncManager {
  start(): void;
  stop(): void;
  syncNow(): Promise<boolean>;
  readonly status: SyncStatus;
  onEvent(listener: SyncListener): () => void;
  dispose(): void;
}
```

---

## React Native Package (`@netraedge/react-native`)

### Hooks

```typescript
// Face detection
useFaceDetection(): {
  state: FaceDetectionState;
  onFacesDetected: (faces: Face[]) => void;
  reset: () => void;
};

// Face recognition
useFaceRecognition(pipeline: FacePipeline): {
  state: RecognitionState;
  verify: (faceData: Uint8Array, meshPoints: unknown[]) => Promise<VerificationResult | null>;
  enroll: (userId: string, frames: Uint8Array[]) => Promise<boolean>;
  refreshUsers: () => Promise<void>;
  reset: () => void;
};

// Liveness check
useLivenessCheck(): {
  state: LivenessState;
  processFrame: (meshPoints: unknown[], faceData?: Uint8Array) => void;
  reset: () => void;
};
```

### Components

```typescript
// Camera with face detection
<FaceCamera
  onFaceDetected={(faces) => {}}
  isActive={true}
  style={{}}
/>

// Liveness challenge overlay
<LivenessPrompt
  prompt="Blink your eyes"
  blinkCount={0}
  isLive={false}
/>

// Verification result card
<ResultCard
  matched={true}
  userId="user_001"
  confidence={0.92}
  livenessPassed={true}
/>
```

---

## Types

```typescript
// Result pattern
type Result<T, E> =
  | { ok: true; value: T }
  | { ok: false; error: E };

// Face detection
interface FaceDetection {
  boundingBox: BoundingBox;
  landmarks: FaceLandmarks;
  rollAngle: number;
  yawAngle: number;
  confidence: number;
}

// Liveness
interface LivenessResult {
  verdict: LivenessVerdict; // 'live' | 'spoof' | 'unknown'
  confidence: number;
  blink: BlinkResult;
  texture: TextureResult;
  depth: DepthResult;
  passedChecks: LivenessCheck[];
  failedChecks: LivenessCheck[];
}
```
