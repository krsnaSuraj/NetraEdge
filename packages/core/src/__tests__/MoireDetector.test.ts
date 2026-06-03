import { describe, it, expect } from 'vitest';
import { detectMoire } from '../liveness/MoireDetector';

describe('MoireDetector', () => {
  it('returns valid result structure', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = Math.random();
    const result = detectMoire(faceData);
    expect(result).toHaveProperty('moireScore');
    expect(result).toHaveProperty('peaks');
    expect(result).toHaveProperty('periodicity');
    expect(result).toHaveProperty('detected');
    expect(typeof result.moireScore).toBe('number');
    expect(typeof result.detected).toBe('boolean');
  });

  it('score is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = Math.random();
    const result = detectMoire(faceData);
    expect(result.moireScore).toBeGreaterThanOrEqual(0);
    expect(result.moireScore).toBeLessThanOrEqual(1);
  });

  it('periodicity is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = Math.random();
    const result = detectMoire(faceData);
    expect(result.periodicity).toBeGreaterThanOrEqual(0);
    expect(result.periodicity).toBeLessThanOrEqual(1);
  });

  it('flat image has no moire', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const result = detectMoire(faceData);
    expect(result.detected).toBe(false);
    expect(result.moireScore).toBeLessThan(0.4);
  });

  it('accepts number array input', () => {
    const faceData: number[] = [];
    for (let i = 0; i < 112 * 112 * 3; i++) faceData.push(Math.random());
    const result = detectMoire(faceData);
    expect(result.moireScore).toBeGreaterThanOrEqual(0);
  });
});
