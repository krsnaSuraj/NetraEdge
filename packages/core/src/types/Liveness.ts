/**
 * Liveness detection result types.
 *
 * SOTA 6-layer liveness detection (NetraEdge — 2026):
 *   - ACTIVE        : random user-action challenge (blink, smile, head turn)
 *   - TEXTURE       : CDCN-Lite central-difference convolution texture analysis
 *   - RPPG          : remote photoplethysmography (pulse from face pixels) — kills 2D spoofs
 *   - DEPTH         : 3D mesh variance (nose protrusion, cheek curvature)
 *   - DEPTH_MOTION  : parallax-based depth from face movement (catches curved prints)
 *   - MOIRÉ         : FFT-based screen replay detection
 *   - COLOR         : chrominance distribution + illuminant consistency
 *
 * Voting: 3 of 7 must pass for LIVE verdict (configurable; default 2 of 5 for legacy).
 */

export enum LivenessCheck {
  BLINK = 'blink',
  ACTIVE = 'active',
  TEXTURE = 'texture',
  RPPG = 'rppg',
  DEPTH = 'depth',
  DEPTH_MOTION = 'depth_motion',
  MOIRE = 'moire',
  COLOR = 'color',
}

export enum LivenessVerdict {
  LIVE = 'live',
  SPOOF = 'spoof',
  UNKNOWN = 'unknown',
}

export enum LivenessLayer {
  ACTIVE_CHALLENGE = 'active_challenge',
  PASSIVE_TEXTURE = 'passive_texture',
  PASSIVE_RPPG = 'passive_rppg',
  PASSIVE_DEPTH = 'passive_depth',
  PASSIVE_DEPTH_MOTION = 'passive_depth_motion',
  PASSIVE_MOIRE = 'passive_moire',
  PASSIVE_COLOR = 'passive_color',
}

export interface BlinkResult {
  readonly detected: boolean;
  readonly earValue: number;
  readonly duration: number;
}

export interface ActiveChallengeResult {
  readonly challengeType: 'blink' | 'smile' | 'head_turn_left' | 'head_turn_right' | 'none';
  readonly completed: boolean;
  readonly confidence: number;
  readonly durationMs: number;
}

export interface TextureResult {
  readonly realScore: number;
  readonly printScore: number;
  readonly screenScore: number;
}

export interface RppgResult {
  readonly pulseDetected: boolean;
  readonly heartRateBpm: number;
  readonly signalQuality: number;
  readonly dominantPower: number;
  readonly isLive: boolean;
}

export interface DepthResult {
  readonly variance: number;
  readonly isThreeDimensional: boolean;
}

export interface DepthMotionResult {
  readonly parallaxScore: number;
  readonly motionDepth: number;
  readonly isConsistent: boolean;
}

export interface MoirResult {
  readonly moireScore: number;
  readonly peaks: readonly number[];
  readonly periodicity: number;
  readonly detected: boolean;
}

export interface ColorResult {
  readonly realScore: number;
  readonly skinConsistency: number;
  readonly illuminantConsistency: number;
  readonly isReal: boolean;
}

export interface LivenessResult {
  readonly verdict: LivenessVerdict;
  readonly confidence: number;
  readonly blink: BlinkResult;
  readonly active: ActiveChallengeResult;
  readonly texture: TextureResult;
  readonly rppg: RppgResult;
  readonly depth: DepthResult;
  readonly depthMotion: DepthMotionResult;
  readonly moire: MoirResult;
  readonly color: ColorResult;
  readonly passedChecks: readonly LivenessCheck[];
  readonly failedChecks: readonly LivenessCheck[];
  readonly layerWeights: Readonly<Record<LivenessLayer, number>>;
}
