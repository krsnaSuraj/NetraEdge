/**
 * Depth estimation from MediaPipe face mesh 3D landmarks.
 *
 * Real human faces have significant 3D structure — the nose protrudes,
 * cheeks curve, eye sockets are recessed. Printed photos and screens
 * are flat (near-zero depth variance).
 *
 * We compute depth variance from the z-coordinates of the face mesh.
 * A real face will have variance > 0.25, while a flat spoof will
 * have variance < 0.10.
 */

import { LIVENESS_THRESHOLDS } from '../config/constants';
import type { DepthResult } from '../types/Liveness';
import type { Point3D } from '../types/Face';

export class DepthEstimator {
  private readonly _threshold: number;

  constructor(threshold = LIVENESS_THRESHOLDS.depthVariance) {
    this._threshold = threshold;
  }

  /**
   * Estimate depth variance from face mesh z-coordinates.
   *
   * @param meshPoints - 478 face mesh landmarks with z values
   * @returns Depth analysis result
   */
  analyze(meshPoints: readonly Point3D[]): DepthResult {
    if (meshPoints.length === 0) {
      return { variance: 0, isThreeDimensional: false };
    }

    const zValues = meshPoints.map((p) => p.z);
    const variance = this.calculateVariance(zValues);

    return {
      variance,
      isThreeDimensional: variance >= this._threshold,
    };
  }

  private calculateVariance(values: readonly number[]): number {
    const n = values.length;
    if (n === 0) return 0;

    let sum = 0;
    for (const v of values) {
      sum += v;
    }
    const mean = sum / n;

    let sumSquaredDiffs = 0;
    for (const v of values) {
      const diff = v - mean;
      sumSquaredDiffs += diff * diff;
    }

    return sumSquaredDiffs / n;
  }
}
