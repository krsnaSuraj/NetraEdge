/**
 * FacePipeline — the main orchestrator for face processing.
 *
 * Coordinates detection → alignment → recognition → liveness
 * into a single call. This is the public API that the React Native
 * layer calls.
 *
 * Usage:
 *   const pipeline = new FacePipeline(encoder, store, liveness);
 *   const result = await pipeline.verify(frame);
 */

import { RECOGNITION_THRESHOLDS } from '../config/constants';
import type { Encoder } from '../embedding/Encoder';
import type { EmbeddingStore, IdentificationResult } from '../embedding/EmbeddingStore';
import type { LivenessOrchestrator } from '../liveness/LivenessOrchestrator';
import type { FaceDetection, Point3D } from '../types/Face';
import type { LivenessResult } from '../types/Liveness';
import { LivenessVerdict } from '../types/Liveness';
import { type Result, ok, err, ErrorCode } from '../types/Result';

export interface EnrollmentResult {
  readonly userId: string;
  readonly frameCount: number;
  readonly averageQuality: number;
}

export interface VerificationResult {
  readonly matched: boolean;
  readonly userId: string | null;
  readonly confidence: number;
  readonly liveness: LivenessResult;
}

export interface PipelineConfig {
  readonly recognitionThreshold: number;
  readonly enrollmentFrameCount: number;
  readonly requireLiveness: boolean;
}

const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  recognitionThreshold: RECOGNITION_THRESHOLDS.matchConfidence,
  enrollmentFrameCount: RECOGNITION_THRESHOLDS.enrollmentFrameCount,
  requireLiveness: true,
};

export class FacePipeline {
  private readonly _encoder: Encoder;
  private readonly _store: EmbeddingStore;
  private readonly _liveness: LivenessOrchestrator | null;
  private readonly _config: PipelineConfig;

  constructor(
    encoder: Encoder,
    store: EmbeddingStore,
    liveness: LivenessOrchestrator | null = null,
    config?: Partial<PipelineConfig>,
  ) {
    this._encoder = encoder;
    this._store = store;
    this._liveness = liveness;
    this._config = { ...DEFAULT_PIPELINE_CONFIG, ...config };
  }

  /**
   * Enroll a new user by encoding multiple face frames.
   *
   * @param userId - Unique identifier for the user
   * @param faceFrames - Array of face crop data (112×112×3 RGB)
   * @param meshPoints - Corresponding face mesh points
   * @param timestampMs - Frame timestamp
   * @returns Enrollment result or error
   */
  async enroll(
    userId: string,
    faceFrames: readonly Uint8Array[],
    meshPoints: readonly Point3D[][],
    timestampMs: number,
  ): Promise<Result<EnrollmentResult>> {
    if (faceFrames.length === 0) {
      return err({
        code: ErrorCode.ENROLLMENT_FAILED,
        message: 'No face frames provided',
        timestamp: Date.now(),
      });
    }

    if (!this._encoder.isLoaded) {
      return err({
        code: ErrorCode.MODEL_NOT_LOADED,
        message: 'Encoder model not loaded',
        timestamp: Date.now(),
      });
    }

    const embeddings: Float32Array[] = [];
    let totalQuality = 0;

    for (let i = 0; i < faceFrames.length; i++) {
      const frame = faceFrames[i];
      const points = meshPoints[i];

      if (!frame || !points) continue;

      const embedding = this._encoder.encode(frame);
      if (embedding) {
        embeddings.push(embedding);
        totalQuality += 1.0;
      }
    }

    if (embeddings.length === 0) {
      return err({
        code: ErrorCode.ENROLLMENT_FAILED,
        message: 'Could not encode any face frames',
        timestamp: Date.now(),
      });
    }

    const success = await this._store.enroll(userId, embeddings);
    if (!success) {
      return err({
        code: ErrorCode.ENROLLMENT_FAILED,
        message: 'Failed to store enrollment',
        timestamp: Date.now(),
      });
    }

    return ok({
      userId,
      frameCount: embeddings.length,
      averageQuality: totalQuality / faceFrames.length,
    });
  }

  /**
   * Verify a face frame against enrolled users.
   *
   * @param faceData - Face crop (112×112×3 RGB)
   * @param meshPoints - 478 face mesh landmarks
   * @param timestampMs - Frame timestamp
   * @param skipLiveness - Skip liveness check (for testing)
   * @returns Verification result
   */
  async verify(
    faceData: Uint8Array,
    meshPoints: Point3D[],
    timestampMs: number,
    skipLiveness = false,
  ): Promise<Result<VerificationResult>> {
    if (!this._encoder.isLoaded) {
      return err({
        code: ErrorCode.MODEL_NOT_LOADED,
        message: 'Encoder model not loaded',
        timestamp: Date.now(),
      });
    }

    let livenessResult: LivenessResult = {
      verdict: LivenessVerdict.UNKNOWN,
      confidence: 0,
      blink: { detected: false, earValue: 0, duration: 0 },
      texture: { realScore: 0, printScore: 0, screenScore: 0 },
      depth: { variance: 0, isThreeDimensional: false },
      passedChecks: [],
      failedChecks: [],
    };

    if (this._liveness && !skipLiveness) {
      livenessResult = this._liveness.processFrame(faceData, meshPoints, timestampMs);

      if (livenessResult.verdict === LivenessVerdict.SPOOF) {
        return ok({
          matched: false,
          userId: null,
          confidence: 0,
          liveness: livenessResult,
        });
      }
    }

    const embedding = this._encoder.encode(faceData);
    if (!embedding) {
      return err({
        code: ErrorCode.MODEL_INFERENCE_FAILED,
        message: 'Failed to generate embedding',
        timestamp: Date.now(),
      });
    }

    const match = await this._store.identify(
      embedding,
      this._config.recognitionThreshold,
    );

    return ok({
      matched: match !== null,
      userId: match?.userId ?? null,
      confidence: match?.confidence ?? 0,
      liveness: livenessResult,
    });
  }

  /** List all enrolled user IDs */
  async listUsers(): Promise<readonly string[]> {
    return this._store.listUsers();
  }

  /** Remove a user's enrollment */
  async removeUser(userId: string): Promise<boolean> {
    return this._store.remove(userId);
  }

  /** Get enrollment count */
  async enrollmentCount(): Promise<number> {
    return this._store.count();
  }

  /** Clear all enrollments */
  async clearEnrollments(): Promise<void> {
    return this._store.clear();
  }

  /** Release all resources */
  dispose(): void {
    this._encoder.dispose();
    this._store.dispose();
    this._liveness?.dispose();
  }
}
