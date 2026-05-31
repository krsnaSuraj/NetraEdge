import { describe, it, expect } from 'vitest';
import { detectMoiré } from '../liveness/MoiréDetector';

describe('MoiréDetector', () => {
  it('returns valid result structure', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = Math.random();
    const result = detectMoiré(faceData);
    expect(result).toHaveProperty('moiréScore');
    expect(result).toHaveProperty('peaks');
    expect(result).toHaveProperty('periodicity');
    expect(result).toHaveProperty('detected');
    expect(typeof result.moiréScore).toBe('number');
    expect(typeof result.detected).toBe('boolean');
  });

  it('score is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = Math.random();
    const result = detectMoiré(faceData);
    expect(result.moiréScore).toBeGreaterThanOrEqual(0);
    expect(result.moiréScore).toBeLessThanOrEqual(1);
  });

  it('periodicity is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = Math.random();
    const result = detectMoiré(faceData);
    expect(result.periodicity).toBeGreaterThanOrEqual(0);
    expect(result.periodicity).toBeLessThanOrEqual(1);
  });

  it('flat image has no moiré', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const result = detectMoiré(faceData);
    expect(result.detected).toBe(false);
    expect(result.peaks.length).toBe(0);
  });

  it('accepts number array input', () => {
    const faceData: number[] = [];
    for (let i = 0; i < 112 * 112 * 3; i++) faceData.push(Math.random());
    const result = detectMoiré(faceData);
    expect(result.moiréScore).toBeGreaterThanOrEqual(0);
  });
});
