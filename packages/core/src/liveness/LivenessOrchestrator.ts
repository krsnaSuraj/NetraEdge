/**
 * SOTA 6-layer liveness orchestrator.
 *
 * Combines 6 independent liveness signals into a single verdict with
 * attention-weighted fusion. SOTA design (NetraEdge — 2026).
 *
 *   L1: Active challenge (blink/smile/head turn) — kills video replays
 *   L2: CDCN-Lite texture CNN — kills print/screen
 *   L3: rPPG pulse detection — kills ANY 2D spoof (blood-flow)
 *   L4: Depth mesh variance — kills flat 2D
 *   L5: Depth-from-motion parallax — kills curved prints
 *   L6: Moiré FFT — kills screen replays
 *   L7: Chrominance + illuminant — kills tinted prints
 *
 * Voting: configurable minimum passes; default 3-of-7 for live verdict.
 * Weighted confidence: attention-style weights computed from per-check reliability.
 */

import { LIVENESS_THRESHOLDS } from '../config/constants';
import { BlinkDetector } from './BlinkDetector';
import { DepthEstimator } from './DepthEstimator';
import { detectMoire } from './MoireDetector';
import { analyzeColor } from './ColorAnalyzer';
import type { TextureAnalyzer } from './TextureAnalyzer';
import { RppgDetector } from './RppgDetector';
import { ActiveChallengeManager } from './ActiveChallengeManager';
import { DepthMotionEstimator } from './DepthMotionEstimator';
import type { Point3D } from '../types/Face';
import type {
  LivenessResult,
  BlinkResult,
  TextureResult,
  DepthResult,
  RppgResult,
  ActiveChallengeResult,
  DepthMotionResult,
  MoirResult,
  ColorResult,
  LivenessCheck as LivenessCheckType,
} from '../types/Liveness';
import { LivenessCheck, LivenessVerdict, LivenessLayer } from '../types/Liveness';

export interface SOTALivenessConfig {
  readonly minPasses: number;
  readonly enableActive: boolean;
  readonly enableRppg: boolean;
  readonly enableDepthMotion: boolean;
}

const DEFAULT_CONFIG: SOTALivenessConfig = {
  minPasses: 3,
  enableActive: true,
  enableRppg: true,
  enableDepthMotion: true,
};

export class SOTALivenessOrchestrator {
  private readonly _blinkDetector: BlinkDetector;
  private readonly _textureAnalyzer: TextureAnalyzer;
  private readonly _depthEstimator: DepthEstimator;
  private readonly _depthMotionEstimator: DepthMotionEstimator;
  private readonly _rppgDetector: RppgDetector;
  private readonly _activeChallenge: ActiveChallengeManager;
  private readonly _config: SOTALivenessConfig;

  constructor(
    textureAnalyzer: TextureAnalyzer,
    blinkDetector?: BlinkDetector,
    depthEstimator?: DepthEstimator,
    depthMotionEstimator?: DepthMotionEstimator,
    rppgDetector?: RppgDetector,
    activeChallenge?: ActiveChallengeManager,
    config?: Partial<SOTALivenessConfig>,
  ) {
    this._blinkDetector = blinkDetector ?? new BlinkDetector();
    this._textureAnalyzer = textureAnalyzer;
    this._depthEstimator = depthEstimator ?? new DepthEstimator();
    this._depthMotionEstimator = depthMotionEstimator ?? new DepthMotionEstimator();
    this._rppgDetector = rppgDetector ?? new RppgDetector();
    this._activeChallenge = activeChallenge ?? new ActiveChallengeManager();
    this._config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Run all 6+ liveness checks on a single frame.
   */
  async processFrame(
    faceData: Float32Array,
    meshPoints: readonly Point3D[],
    timestampMs: number,
  ): Promise<LivenessResult> {
    const blink = this._blinkDetector.processFrame(meshPoints, timestampMs);
    const texture = await this._textureAnalyzer.analyze(faceData);
    const depth = this._depthEstimator.analyze(meshPoints);
    const depthMotion = this._depthMotionEstimator.analyze(meshPoints, timestampMs);
    const rppg = this._rppgDetector.feed(faceData, timestampMs);
    const active = this._activeChallenge.tick(timestampMs);
    const moire = detectMoire(faceData);
    const color = analyzeColor(faceData);

    return this.combineResults(blink, active, texture, rppg, depth, depthMotion, moire, color);
  }

  /** Push frame to rPPG (call every frame for ~30 frames before verdict). */
  feedRppg(faceData: Float32Array, timestampMs: number): RppgResult {
    return this._rppgDetector.feed(faceData, timestampMs);
  }

  /** Get current pulse estimate from rPPG buffer. */
  getRppgResult(): RppgResult {
    return this._rppgDetector.getResult();
  }

  /** Set the active challenge (e.g. "smile" or "turn_left"). */
  setActiveChallenge(
    type: 'blink' | 'smile' | 'head_turn_left' | 'head_turn_right',
  ): void {
    this._activeChallenge.setChallenge(type);
  }

  get activeCompleted(): boolean {
    return this._activeChallenge.completed;
  }

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
    this._activeChallenge.reset();
    this._rppgDetector.reset();
  }

  dispose(): void {
    this._textureAnalyzer.dispose();
    this._blinkDetector.reset();
    this._rppgDetector.reset();
    this._activeChallenge.reset();
  }

  private combineResults(
    blink: BlinkResult,
    active: ActiveChallengeResult,
    texture: TextureResult,
    rppg: RppgResult,
    depth: DepthResult,
    depthMotion: DepthMotionResult,
    moire: MoirResult,
    color: ColorResult,
  ): LivenessResult {
    const passedChecks: LivenessCheckType[] = [];
    const failedChecks: LivenessCheckType[] = [];

    // L1: Active challenge (only enabled in verify mode)
    if (this._config.enableActive) {
      if (active.completed && active.confidence > 0.6) {
        passedChecks.push(LivenessCheck.ACTIVE);
      } else if (this._activeChallenge.isActive) {
        failedChecks.push(LivenessCheck.ACTIVE);
      }
    }

    // L2: Blink (legacy/temporal)
    if (blink.detected) {
      passedChecks.push(LivenessCheck.BLINK);
    } else if (this._blinkDetector.blinkCount > 0) {
      passedChecks.push(LivenessCheck.BLINK);
    } else {
      failedChecks.push(LivenessCheck.BLINK);
    }

    // L3: Texture (CDCN-Lite)
    if (texture.realScore >= LIVENESS_THRESHOLDS.textureConfidence) {
      passedChecks.push(LivenessCheck.TEXTURE);
    } else {
      failedChecks.push(LivenessCheck.TEXTURE);
    }

    // L4: rPPG pulse (SOTA differentiator)
    if (this._config.enableRppg) {
      if (rppg.isLive && rppg.signalQuality > 0.4) {
        passedChecks.push(LivenessCheck.RPPG);
      } else if (this._rppgDetector.hasEnoughSamples) {
        failedChecks.push(LivenessCheck.RPPG);
      }
    }

    // L5: Depth
    if (depth.isThreeDimensional) {
      passedChecks.push(LivenessCheck.DEPTH);
    } else {
      failedChecks.push(LivenessCheck.DEPTH);
    }

    // L6: Depth-from-motion
    if (this._config.enableDepthMotion) {
      if (depthMotion.isConsistent && depthMotion.parallaxScore > 0.5) {
        passedChecks.push(LivenessCheck.DEPTH_MOTION);
      } else if (this._depthMotionEstimator.hasMotion) {
        failedChecks.push(LivenessCheck.DEPTH_MOTION);
      }
    }

    // L7: Moiré (screen detection) — fixed routing
    if (!moire.detected) {
      passedChecks.push(LivenessCheck.MOIRE);
    } else {
      failedChecks.push(LivenessCheck.MOIRE);
    }

    // L8: Color (print/screen detection) — fixed routing
    if (color.isReal) {
      passedChecks.push(LivenessCheck.COLOR);
    } else {
      failedChecks.push(LivenessCheck.COLOR);
    }

    // Deduplicate
    const passedSet = new Set<LivenessCheckType>(passedChecks);
    const failedSet = new Set<LivenessCheckType>(failedChecks);

    // Verdict
    const passedCount = passedSet.size;
    const verdict: LivenessVerdict =
      passedCount >= this._config.minPasses
        ? LivenessVerdict.LIVE
        : LivenessVerdict.SPOOF;

    // Attention-weighted confidence
    const weights = {
      [LivenessLayer.ACTIVE_CHALLENGE]: 0.20,
      [LivenessLayer.PASSIVE_TEXTURE]: 0.20,
      [LivenessLayer.PASSIVE_RPPG]: 0.25,
      [LivenessLayer.PASSIVE_DEPTH]: 0.10,
      [LivenessLayer.PASSIVE_DEPTH_MOTION]: 0.10,
      [LivenessLayer.PASSIVE_MOIRE]: 0.10,
      [LivenessLayer.PASSIVE_COLOR]: 0.05,
    };

    const confidence =
      weights[LivenessLayer.ACTIVE_CHALLENGE] * (active.completed ? active.confidence : 0) +
      weights[LivenessLayer.PASSIVE_TEXTURE] * texture.realScore +
      weights[LivenessLayer.PASSIVE_RPPG] * (rppg.isLive ? rppg.signalQuality : 0) +
      weights[LivenessLayer.PASSIVE_DEPTH] * (depth.isThreeDimensional ? 1 : 0) +
      weights[LivenessLayer.PASSIVE_DEPTH_MOTION] * depthMotion.parallaxScore +
      weights[LivenessLayer.PASSIVE_MOIRE] * (1 - moire.moireScore) +
      weights[LivenessLayer.PASSIVE_COLOR] * color.realScore;

    return {
      verdict,
      confidence: Math.min(1, Math.max(0, confidence)),
      blink,
      active,
      texture,
      rppg,
      depth,
      depthMotion,
      moire,
      color,
      passedChecks: Array.from(passedSet),
      failedChecks: Array.from(failedSet),
      layerWeights: weights,
    };
  }
}

// Backward-compat alias
export { SOTALivenessOrchestrator as LivenessOrchestrator };
export type { MoirResult, ColorResult };
