/**
 * NetraEdge configuration constants.
 *
 * Every tunable value lives here — no magic numbers anywhere else.
 * All thresholds are validated against research benchmarks.
 */

export const MODEL_CONFIG = {
  /** Input dimensions for face recognition model (MobileFaceNet) */
  recognition: {
    inputWidth: 112,
    inputHeight: 112,
    embeddingDimension: 128,
    modelFile: 'face_recognition.tflite',
  },
  /** Input dimensions for liveness detection model */
  liveness: {
    inputWidth: 112,
    inputHeight: 112,
    numClasses: 3,
    modelFile: 'liveness_detector.tflite',
  },
} as const;

export const RECOGNITION_THRESHOLDS = {
  /** Minimum cosine similarity to consider a match (0.0–1.0) */
  matchConfidence: 0.75,
  /** Minimum face size in pixels to process */
  minFaceSize: 80,
  /** Maximum roll angle in degrees before rejecting */
  maxRollAngle: 15,
  /** Minimum quality score (0.0–1.0) to proceed with recognition */
  minQualityScore: 0.6,
  /** Number of frames to average for enrollment */
  enrollmentFrameCount: 10,
} as const;

export const LIVENESS_THRESHOLDS = {
  /** Eye Aspect Ratio below this value indicates closed eye */
  blinkEARThreshold: 0.21,
  /** Minimum blink duration in milliseconds */
  blinkMinDuration: 80,
  /** Maximum allowed time between blinks in milliseconds */
  blinkMaxInterval: 4000,
  /** Minimum real-face score from texture CNN (0.0–1.0) */
  textureConfidence: 0.80,
  /** Minimum depth variance for 3D face (vs flat photo/screen) */
  depthVariance: 0.25,
  /** Required passing liveness checks (ALL must pass) */
  requiredChecks: 2 as const,
} as const;

export const QUALITY_CONFIG = {
  /** Brightness range (0–255) — below this is too dark */
  minBrightness: 40,
  /** Above this is too bright (harsh sunlight) */
  maxBrightness: 220,
  /** Minimum image sharpness (Laplacian variance) */
  minSharpness: 50,
  /** Maximum face yaw (left/right turn) in degrees */
  maxYawAngle: 20,
  /** Maximum face pitch (up/down) in degrees */
  maxPitchAngle: 20,
} as const;

export const SYNC_CONFIG = {
  /** Local storage purge delay after successful sync (ms) */
  purgeDelayMs: 5000,
  /** Maximum retry attempts for failed sync */
  maxRetries: 3,
  /** Sync batch size */
  batchSize: 50,
  /** Network check interval when offline (ms) */
  networkCheckIntervalMs: 30000,
} as const;

export const CAMERA_CONFIG = {
  /** Target frame processing rate (skip frames for performance) */
  frameSkipCount: 3,
  /** Camera resolution preset */
  resolution: '720p' as const,
  /** Enable front camera only */
  facing: 'front' as const,
} as const;
