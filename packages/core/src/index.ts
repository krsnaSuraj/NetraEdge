/**
 * @netraedge/core
 *
 * Core face recognition and liveness detection logic.
 * Pure TypeScript — no React Native or native dependencies.
 *
 * @example
 * import { FacePipeline, StubEncoder, InMemoryEmbeddingStore } from '@netraedge/core';
 *
 * const encoder = new StubEncoder();
 * const store = new InMemoryEmbeddingStore();
 * const pipeline = new FacePipeline(encoder, store);
 *
 * await pipeline.enroll('user1', [faceFrame], [meshPoints], Date.now());
 * const result = await pipeline.verify(faceFrame, meshPoints, Date.now());
 */

// Types
export type { FaceDetection, BoundingBox, FaceLandmarks, Point2D, FaceMesh, Point3D } from './types/Face';
export {
  LivenessCheck,
  LivenessVerdict,
  type BlinkResult,
  type TextureResult,
  type DepthResult,
  type LivenessResult,
} from './types/Liveness';
export {
  type Result,
  ok,
  err,
  unwrap,
  NetraEdgeError,
  ErrorCode,
} from './types/Result';

// Config
export {
  MODEL_CONFIG,
  RECOGNITION_THRESHOLDS,
  LIVENESS_THRESHOLDS,
  QUALITY_CONFIG,
  SYNC_CONFIG,
  CAMERA_CONFIG,
} from './config/constants';
export { type AppConfig, LivenessMode, validateConfig } from './config/AppConfig';

// Embedding
export { cosineSimilarity, l2Normalize } from './embedding/CosineSimilarity';
export { type Encoder, TFLiteEncoder, StubEncoder } from './embedding/Encoder';
export {
  type Enrollment,
  type EmbeddingStore,
  type IdentificationResult,
  InMemoryEmbeddingStore,
} from './embedding/EmbeddingStore';

// Liveness
export { BlinkDetector } from './liveness/BlinkDetector';
export {
  type TextureAnalyzer,
  CNNTextureAnalyzer,
  StubTextureAnalyzer,
} from './liveness/TextureAnalyzer';
export { DepthEstimator } from './liveness/DepthEstimator';
export { LivenessOrchestrator } from './liveness/LivenessOrchestrator';

// Pipeline
export {
  FacePipeline,
  type EnrollmentResult,
  type VerificationResult,
  type PipelineConfig,
} from './pipeline/FacePipeline';

// Sync
export {
  SyncStatus,
  type SyncEvent,
  type SyncListener,
  type SyncManager,
  type SyncTransport,
  type NetworkMonitor,
  DefaultSyncManager,
} from './sync/SyncManager';
