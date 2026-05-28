/**
 * Liveness orchestrator — combines blink, texture, and depth checks
 * into a single liveness verdict.
 *
 * This is the main entry point for liveness detection. It runs all
 * available checks and requires a minimum number to pass.
 *
 * Design:
 * - Blink check: temporal (requires video frames)
 * - Texture check: per-frame (CNN inference)
 * - Depth check: per-frame (face mesh 3D data)
 *
 * All checks must pass for a LIVE verdict. If any critical check
 * fails, the result is SPOOF.
 */

import { LIVENESS_THRESHOLDS } from '../config/constants';
import { BlinkDetector } from './BlinkDetector';
import { DepthEstimator } from './DepthEstimator';
import type { TextureAnalyzer } from './TextureAnalyzer';
import type { Point3D } from '../types/Face';
import type { LivenessResult, BlinkResult, TextureResult, DepthResult } from '../types/Liveness';
import { LivenessCheck, LivenessVerdict } from '../types/Liveness';

export class LivenessOrchestrator {
  private readonly _blinkDetector: BlinkDetector;
  private readonly _textureAnalyzer: TextureAnalyzer;
  private readonly _depthEstimator: DepthEstimator;
  private readonly _requiredChecks: number;

  constructor(
    textureAnalyzer: TextureAnalyzer,
    blinkDetector?: BlinkDetector,
    depthEstimator?: DepthEstimator,
    requiredChecks = LIVENESS_THRESHOLDS.requiredChecks,
  ) {
    this._blinkDetector = blinkDetector ?? new BlinkDetector();
    this._textureAnalyzer = textureAnalyzer;
    this._depthEstimator = depthEstimator ?? new DepthEstimator();
    this._requiredChecks = requiredChecks;
  }

  /**
   * Run a single-frame liveness check (blink + texture + depth).
   *
   * For blink detection, call processBlinkFrame across multiple frames.
   *
   * @param faceData - 112×112×3 RGB face crop (normalized floats)
   * @param meshPoints - 478 face mesh points with z coordinates
   * @param timestampMs - Frame timestamp
   * @returns Liveness result (blink may be UNKNOWN on first frame)
   */
  async processFrame(
    faceData: Float32Array,
    meshPoints: readonly Point3D[],
    timestampMs: number,
  ): Promise<LivenessResult> {
    const blink = this._blinkDetector.processFrame(meshPoints, timestampMs);
    const texture = await this._textureAnalyzer.analyze(faceData);
    const depth = this._depthEstimator.analyze(meshPoints);

    return this.combineResults(blink, texture, depth);
  }

  /**
   * Process a frame specifically for blink detection.
   * Call this repeatedly to accumulate blink history.
   */
  processBlinkFrame(
    meshPoints: ReadonlyArray<{ readonly x: number; readonly y: number }>,
    timestampMs: number,
  ): BlinkResult {
    return this._blinkDetector.processFrame(meshPoints, timestampMs);
  }

  get blinkCount(): number {
    return this._blinkDetector.blinkCount;
  }

  reset(): void {
    this._blinkDetector.reset();
  }

  dispose(): void {
    this._textureAnalyzer.dispose();
    this._blinkDetector.reset();
  }

  private combineResults(
    blink: BlinkResult,
    texture: TextureResult,
    depth: DepthResult,
  ): LivenessResult {
    const passedChecks: LivenessCheck[] = [];
    const failedChecks: LivenessCheck[] = [];

    if (blink.detected) {
      passedChecks.push(LivenessCheck.BLINK);
    } else {
      failedChecks.push(LivenessCheck.BLINK);
    }

    if (texture.realScore >= LIVENESS_THRESHOLDS.textureConfidence) {
      passedChecks.push(LivenessCheck.TEXTURE);
    } else {
      failedChecks.push(LivenessCheck.TEXTURE);
    }

    if (depth.isThreeDimensional) {
      passedChecks.push(LivenessCheck.DEPTH);
    } else {
      failedChecks.push(LivenessCheck.DEPTH);
    }

    const passedCount = passedChecks.length;
    const verdict =
      passedCount >= this._requiredChecks
        ? LivenessVerdict.LIVE
        : LivenessVerdict.SPOOF;

    const confidence = passedCount / 3;

    return {
      verdict,
      confidence,
      blink,
      texture,
      depth,
      passedChecks,
      failedChecks,
    };
  }
}
