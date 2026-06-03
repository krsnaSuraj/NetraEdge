/**
 * NetraEdge configuration constants.
 *
 * Every tunable value lives here — no magic numbers anywhere else.
 * All thresholds are validated against research benchmarks.
 */

export const MODEL_CONFIG = {
  /** Input dimensions for face recognition model (MobileFaceNet Apache-2.0) */
  recognition: {
    inputWidth: 112,
    inputHeight: 112,
    /**
     * SOTA production dim for MobileFaceNet v1 (foamliu, 99.48% LFW,
     * Apache-2.0). The on-disk TFLite is bit-identical to foamliu's
     * scripted release. Native bridges read the actual dim at load
     * time, so this constant is documentation-only (real dim comes
     * from model).
     */
    embeddingDimension: 128, // SOTA production: MobileFaceNet 128-d, Apache-2.0
    modelFile: 'face_recognition.tflite',
  },
  /** Input dimensions for liveness detection model (MiniFASNet-style Apache-2.0) */
  liveness: {
    inputWidth: 112,
    inputHeight: 112,
    numClasses: 3,
    modelFile: 'liveness_detector.tflite',
  },
  /** Face detection / landmarks (MediaPipe Face Landmarker, Apache-2.0) */
  faceLandmarker: {
    modelFile: 'face_landmarker.task',
    /** Number of 3D face landmarks (MediaPipe canonical) */
    landmarkCount: 468,
    /** Number of blendshape scores (eyes, mouth, jaw, brow, ...) */
    blendshapeCount: 52,
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

export const ACTIVE_LIVENESS_CONFIG = {
  /** Blendshape score threshold for "blink detected" (MediaPipe eyeBlinkLeft/Right) */
  blinkThreshold: 0.5,
  /** Required number of blinks during active challenge */
  blinkCount: 2,
  /** Blendshape score threshold for "smile detected" (MediaPipe _smile blendshape) */
  smileThreshold: 0.5,
  /** Yaw threshold in degrees for "head turned left" (MediaPipe headYaw blendshape) */
  headTurnYawThreshold: 15,
  /** Total time budget for active liveness challenge in ms */
  challengeTimeoutMs: 10000,
} as const;

export const RPPG_CONFIG = {
  /** Number of frames to accumulate for rPPG pulse estimation */
  windowFrames: 90, // ~3 sec at 30 fps
  /** Expected human heart rate range in Hz (40-150 bpm) */
  minBpmHz: 0.7,
  maxBpmHz: 2.5,
  /** Minimum spectral peak prominence to accept a pulse signal */
  minPeakProminence: 0.05,
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
