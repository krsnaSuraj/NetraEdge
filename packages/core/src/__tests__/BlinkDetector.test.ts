import { describe, it, expect, beforeEach } from 'vitest';
import { BlinkDetector } from '../liveness/BlinkDetector';

function makeMeshPoint(x: number, y: number): { x: number; y: number } {
  return { x, y };
}

function makeEyePoints(
  outerX: number,
  innerX: number,
  upperY: number,
  lowerY: number,
): { x: number; y: number }[] {
  const midY = (upperY + lowerY) / 2;
  return [
    makeMeshPoint(outerX, midY),
    makeMeshPoint(outerX + 0.02, upperY),
    makeMeshPoint(outerX + 0.04, upperY + 0.005),
    makeMeshPoint(innerX, midY),
    makeMeshPoint(outerX + 0.04, lowerY - 0.005),
    makeMeshPoint(outerX + 0.02, lowerY),
  ];
}

describe('BlinkDetector', () => {
  let detector: BlinkDetector;

  beforeEach(() => {
    detector = new BlinkDetector(30, 0.21, 80);
  });

  it('detects no blink with constant open eyes', () => {
    const openEye = makeEyePoints(0.3, 0.4, 0.35, 0.45);
    const allPoints = new Array(468).fill(null).map((_, i) => {
      if (i === 33) return openEye[0]!;
      if (i === 160) return openEye[1]!;
      if (i === 158) return openEye[2]!;
      if (i === 133) return openEye[3]!;
      if (i === 153) return openEye[4]!;
      if (i === 144) return openEye[5]!;
      if (i === 362) return openEye[0]!;
      if (i === 385) return openEye[1]!;
      if (i === 387) return openEye[2]!;
      if (i === 263) return openEye[3]!;
      if (i === 373) return openEye[4]!;
      if (i === 380) return openEye[5]!;
      return makeMeshPoint(0.5, 0.5);
    });

    for (let i = 0; i < 10; i++) {
      const result = detector.processFrame(allPoints, i * 33);
      expect(result.detected).toBe(false);
    }
  });

  it('tracks blink count', () => {
    expect(detector.blinkCount).toBe(0);
    detector.reset();
    expect(detector.blinkCount).toBe(0);
  });

  it('resets state correctly', () => {
    detector.reset();
    expect(detector.blinkCount).toBe(0);
  });
});
