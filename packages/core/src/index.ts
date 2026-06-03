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
  LivenessLayer,
  type BlinkResult,
  type ActiveChallengeResult,
  type TextureResult,
  type RppgResult,
  type DepthResult,
  type DepthMotionResult,
  type MoirResult,
  type ColorResult,
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

// Liveness (SOTA 6-layer)
export { BlinkDetector } from './liveness/BlinkDetector';
export {
  type TextureAnalyzer,
  CNNTextureAnalyzer,
  StubTextureAnalyzer,
} from './liveness/TextureAnalyzer';
export { DepthEstimator } from './liveness/DepthEstimator';
export { DepthMotionEstimator } from './liveness/DepthMotionEstimator';
export { RppgDetector } from './liveness/RppgDetector';
export { ActiveChallengeManager, type ChallengeType } from './liveness/ActiveChallengeManager';
export { detectMoire } from './liveness/MoireDetector';
export { analyzeColor } from './liveness/ColorAnalyzer';
export { SOTALivenessOrchestrator, LivenessOrchestrator, type SOTALivenessConfig } from './liveness/LivenessOrchestrator';

// Quality
export { evaluateQuality, type QualityResult } from './quality/QualityGate';

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
export {
  type SyncQueueItem,
  type SyncQueue,
  InMemorySyncQueue,
} from './sync/SyncQueue';
export {
  type PurgePolicy,
  type PurgeResult,
  DEFAULT_PURGE_POLICY,
  DataPurgeManager,
} from './sync/DataPurge';
