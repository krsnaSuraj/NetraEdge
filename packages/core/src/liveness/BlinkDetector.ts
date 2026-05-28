/**
 * Blink detection using Eye Aspect Ratio (EAR).
 *
 * Based on the paper "Real-Time Eye Blink Detection using Facial Landmarks"
 * by Soukupová and Čech (2016).
 *
 * EAR formula:
 *   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
 *
 * Where p1..p6 are the 6 eye landmarks:
 *   p1 = outer corner, p2-p6 = upper/lower lid points, p4 = inner corner
 *
 * When the eye is open, EAR is high (~0.3).
 * When closed, EAR drops (~0.1 or lower).
 */

import { LIVENESS_THRESHOLDS } from '../config/constants';
import type { Point2D } from '../types/Face';
import type { BlinkResult } from '../types/Liveness';

/**
 * MediaPipe Face Mesh eye landmark indices.
 *
 * Left eye:  33, 160, 158, 133, 153, 144
 * Right eye: 362, 385, 387, 263, 373, 380
 */
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] as const;
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380] as const;

export class BlinkDetector {
  private readonly _earHistory: number[] = [];
  private readonly _maxHistory: number;
  private _blinkStartTime: number | null = null;
  private _blinkCount = 0;
  private readonly _threshold: number;
  private readonly _minDuration: number;

  constructor(
    maxHistory = 30,
    threshold = LIVENESS_THRESHOLDS.blinkEARThreshold,
    minDuration = LIVENESS_THRESHOLDS.blinkMinDuration,
  ) {
    this._maxHistory = maxHistory;
    this._threshold = threshold;
    this._minDuration = minDuration;
  }

  /**
   * Process a frame's face mesh points and detect blinks.
   *
   * @param meshPoints - 478 face mesh landmarks (normalized coords)
   * @param timestampMs - Frame timestamp in milliseconds
   * @returns Blink detection result
   */
  processFrame(
    meshPoints: ReadonlyArray<{ readonly x: number; readonly y: number }>,
    timestampMs: number,
  ): BlinkResult {
    const leftEAR = this.calculateEAR(meshPoints, LEFT_EYE_INDICES);
    const rightEAR = this.calculateEAR(meshPoints, RIGHT_EYE_INDICES);
    const avgEAR = (leftEAR + rightEAR) / 2;

    this._earHistory.push(avgEAR);
    if (this._earHistory.length > this._maxHistory) {
      this._earHistory.shift();
    }

    const blinkDetected = this.detectBlinkTransition(timestampMs);

    return {
      detected: blinkDetected,
      earValue: avgEAR,
      duration: this._blinkStartTime !== null
        ? timestampMs - this._blinkStartTime
        : 0,
    };
  }

  get blinkCount(): number {
    return this._blinkCount;
  }

  reset(): void {
    this._earHistory.length = 0;
    this._blinkStartTime = null;
    this._blinkCount = 0;
  }

  private calculateEAR(
    points: ReadonlyArray<{ readonly x: number; readonly y: number }>,
    indices: readonly number[],
  ): number {
    const p1 = points[indices[0]];
    const p2 = points[indices[1]];
    const p3 = points[indices[2]];
    const p4 = points[indices[3]];
    const p5 = points[indices[4]];
    const p6 = points[indices[5]];

    if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return 0.3;

    const vertical1 = this.euclideanDistance(p2, p6);
    const vertical2 = this.euclideanDistance(p3, p5);
    const horizontal = this.euclideanDistance(p1, p4);

    if (horizontal === 0) return 0.3;

    return (vertical1 + vertical2) / (2 * horizontal);
  }

  private euclideanDistance(
    a: { readonly x: number; readonly y: number },
    b: { readonly x: number; readonly y: number },
  ): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private detectBlinkTransition(timestampMs: number): boolean {
    if (this._earHistory.length < 3) return false;

    const current = this._earHistory[this._earHistory.length - 1] ?? 0.3;
    const prev = this._earHistory[this._earHistory.length - 2] ?? 0.3;

    const isClosed = current < this._threshold;
    const wasOpen = prev >= this._threshold;

    if (isClosed && wasOpen) {
      this._blinkStartTime = timestampMs;
      return false;
    }

    if (!isClosed && this._blinkStartTime !== null) {
      const duration = timestampMs - this._blinkStartTime;
      this._blinkStartTime = null;

      if (duration >= this._minDuration) {
        this._blinkCount++;
        return true;
      }
    }

    return false;
  }
}
