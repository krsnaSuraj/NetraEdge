/**
 * Liveness detection result types.
 *
 * Liveness is determined by combining multiple signals:
 * blink detection, texture analysis, and depth estimation.
 */

export enum LivenessCheck {
  BLINK = 'blink',
  TEXTURE = 'texture',
  DEPTH = 'depth',
}

export enum LivenessVerdict {
  LIVE = 'live',
  SPOOF = 'spoof',
  UNKNOWN = 'unknown',
}

export interface BlinkResult {
  readonly detected: boolean;
  readonly earValue: number;
  readonly duration: number;
}

export interface TextureResult {
  readonly realScore: number;
  readonly printScore: number;
  readonly screenScore: number;
}

export interface DepthResult {
  readonly variance: number;
  readonly isThreeDimensional: boolean;
}

export interface LivenessResult {
  readonly verdict: LivenessVerdict;
  readonly confidence: number;
  readonly blink: BlinkResult;
  readonly texture: TextureResult;
  readonly depth: DepthResult;
  readonly passedChecks: readonly LivenessCheck[];
  readonly failedChecks: readonly LivenessCheck[];
}
