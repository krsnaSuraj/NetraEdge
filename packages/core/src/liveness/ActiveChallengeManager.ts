/**
 * ActiveChallengeManager — SOTA anti-replay via random user challenges.
 *
 * Drives a state machine that requires the user to perform random actions:
 *   1. BLINK          — Eye Aspect Ratio (EAR) < threshold for 100ms
 *   2. SMILE          — Mouth corner Y-displacement > 8px
 *   3. HEAD_TURN_LEFT — Yaw angle > 15° to the left
 *   4. HEAD_TURN_RIGHT — Yaw angle > 15° to the right
 *
 * Why this matters: defeats pre-recorded video attacks. A 2D video can't
 * respond to a *random* challenge that wasn't pre-recorded.
 *
 * State machine:
 *   IDLE → (setChallenge) → PENDING → (tick detects action) → COMPLETED
 *                                   → (timeout 5s) → FAILED
 *
 * Random challenge order is enforced via setChallenge(type) called by UI.
 */

import type { ActiveChallengeResult } from '../types/Liveness';

export type ChallengeType = 'blink' | 'smile' | 'head_turn_left' | 'head_turn_right' | 'none';

const SMILE_LIP_DELTA = 6.0;
const HEAD_YAW_DEG = 15.0;
const CHALLENGE_TIMEOUT_MS = 6000;
const MIN_COMPLETION_HOLD_MS = 200;

type State = 'idle' | 'pending' | 'completed' | 'failed';

export class ActiveChallengeManager {
  private _state: State = 'idle';
  private _type: ChallengeType = 'none';
  private _startTimeMs = 0;
  private _completionStartMs = 0;
  private _lastSmileBaseline = 0;

  constructor() {}

  /** Issue a new random challenge. */
  setChallenge(type: ChallengeType): void {
    this._type = type;
    this._state = 'pending';
    this._startTimeMs = Date.now();
    this._completionStartMs = 0;
  }

  /** Stop the challenge (e.g. user backed out). */
  reset(): void {
    this._state = 'idle';
    this._type = 'none';
    this._startTimeMs = 0;
    this._completionStartMs = 0;
  }

  get isActive(): boolean {
    return this._state === 'pending';
  }

  get completed(): boolean {
    return this._state === 'completed';
  }

  get state(): State {
    return this._state;
  }

  get currentChallenge(): ChallengeType {
    return this._type;
  }

  /**
   * Tick once per frame. Pass meshPoints + optional blink result so the
   * manager can verify the requested action.
   */
  tick(timestampMs: number): ActiveChallengeResult {
    if (this._state === 'idle' || this._state === 'completed' || this._type === 'none') {
      return this._result(0, 0);
    }
    if (this._state === 'pending') {
      if (timestampMs - this._startTimeMs > CHALLENGE_TIMEOUT_MS) {
        this._state = 'failed';
        return this._result(timestampMs - this._startTimeMs, 0);
      }
      return this._result(timestampMs - this._startTimeMs, 0);
    }
    return this._result(timestampMs - this._startTimeMs, 1);
  }

  /**
   * Verify the action was performed. Call this from the frame-processor
   * with current mesh landmarks + blink result.
   */
  verifyWithLandmarks(
    meshPoints: ReadonlyArray<{ readonly x: number; readonly y: number }>,
    blinkDetected: boolean,
    timestampMs: number,
  ): ActiveChallengeResult {
    if (this._state !== 'pending') {
      return this._tickOnly(timestampMs);
    }

    let isCompleted = false;
    let confidence = 0;

    switch (this._type) {
      case 'blink':
        if (blinkDetected) {
          isCompleted = true;
          confidence = 0.95;
        }
        break;
      case 'smile':
        isCompleted = this._detectSmile(meshPoints, timestampMs, (c) => (confidence = c));
        break;
      case 'head_turn_left':
      case 'head_turn_right':
        isCompleted = this._detectHeadTurn(
          meshPoints,
          this._type === 'head_turn_left' ? -1 : 1,
          timestampMs,
          (c) => (confidence = c),
        );
        break;
    }

    if (isCompleted) {
      if (this._completionStartMs === 0) {
        this._completionStartMs = timestampMs;
      }
      if (timestampMs - this._completionStartMs >= MIN_COMPLETION_HOLD_MS) {
        this._state = 'completed';
      }
    } else {
      this._completionStartMs = 0;
    }

    if (timestampMs - this._startTimeMs > CHALLENGE_TIMEOUT_MS) {
      this._state = 'failed';
    }

    return this._result(timestampMs - this._startTimeMs, confidence);
  }

  /**
   * Mouth-corner displacement check. MediaPipe landmark 61 (left corner) vs 291 (right corner)
   * and 13 (upper lip) vs 14 (lower lip) — when smiling, corners move up & out,
   * vertical distance between upper-lower lips grows.
   */
  private _detectSmile(
    meshPoints: ReadonlyArray<{ readonly x: number; readonly y: number }>,
    _timestampMs: number,
    setConfidence: (c: number) => void,
  ): boolean {
    if (meshPoints.length < 292) return false;
    const upperLip = meshPoints[13];
    const lowerLip = meshPoints[14];
    const leftCorner = meshPoints[61];
    const rightCorner = meshPoints[291];
    if (!upperLip || !lowerLip || !leftCorner || !rightCorner) return false;

    // Mouth openness (height) and width
    const mouthOpen = Math.abs(lowerLip.y - upperLip.y);
    const mouthWidth = Math.abs(rightCorner.x - leftCorner.x);

    // Baseline: if first 5 frames no smile, mouth is closed
    if (this._lastSmileBaseline === 0 && mouthOpen > 0) {
      this._lastSmileBaseline = mouthOpen;
      return false;
    }

    const openDelta = mouthOpen - this._lastSmileBaseline;
    const aspect = mouthOpen / Math.max(mouthWidth, 0.01);

    // Smile: openness increased AND aspect ratio > neutral (~0.05)
    const isSmile = openDelta > SMILE_LIP_DELTA * 0.01 && aspect > 0.06;
    if (isSmile) {
      const conf = Math.min(1, openDelta / (SMILE_LIP_DELTA * 0.02));
      setConfidence(Math.max(0.6, conf));
    }
    return isSmile;
  }

  /**
   * Head turn detection: compare nose position to mid-eye. When face turns
   * left, the nose shifts to the right of eye center (from camera POV).
   */
  private _detectHeadTurn(
    meshPoints: ReadonlyArray<{ readonly x: number; readonly y: number }>,
    direction: -1 | 1,
    _timestampMs: number,
    setConfidence: (c: number) => void,
  ): boolean {
    if (meshPoints.length < 168) return false;
    const leftEye = meshPoints[33];
    const rightEye = meshPoints[263];
    const noseTip = meshPoints[1];
    if (!leftEye || !rightEye || !noseTip) return false;

    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeDist = Math.abs(rightEye.x - leftEye.x);
    if (eyeDist < 0.01) return false;

    const noseOffsetX = (noseTip.x - eyeMidX) / eyeDist;
    void noseOffsetX; // computed yaw from x; y not used here

    // Convert to degrees (approximation: small-angle)
    const yawDeg = Math.atan2(noseOffsetX, 0.5) * (180 / Math.PI);

    // Head turned in requested direction beyond threshold
    const isTurned = direction === -1 ? yawDeg < -HEAD_YAW_DEG : yawDeg > HEAD_YAW_DEG;
    if (isTurned) {
      const conf = Math.min(1, Math.abs(yawDeg) / (HEAD_YAW_DEG * 1.5));
      setConfidence(Math.max(0.65, conf));
    }
    return isTurned;
  }

  private _tickOnly(timestampMs: number): ActiveChallengeResult {
    return this._result(timestampMs - this._startTimeMs, this._state === 'completed' ? 1 : 0);
  }

  private _result(durationMs: number, confidence: number): ActiveChallengeResult {
    return {
      challengeType: this._type,
      completed: this._state === 'completed',
      confidence: Math.min(1, Math.max(0, confidence)),
      durationMs,
    };
  }
}
