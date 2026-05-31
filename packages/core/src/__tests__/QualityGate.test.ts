import { describe, it, expect } from 'vitest';
import { evaluateQuality } from '../quality/QualityGate';

describe('QualityGate', () => {
  it('returns valid result structure', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = 0.5;
    const result = evaluateQuality(faceData);
    expect(result).toHaveProperty('score');
    expect(result).toHaveProperty('passed');
    expect(result).toHaveProperty('checks');
    expect(result).toHaveProperty('failureReason');
    expect(result.checks).toHaveProperty('brightness');
    expect(result.checks).toHaveProperty('sharpness');
    expect(result.checks).toHaveProperty('faceSize');
    expect(result.checks).toHaveProperty('poseAngle');
  });

  it('score is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = 0.5;
    const result = evaluateQuality(faceData);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });

  it('too dark image fails brightness check', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.05);
    const result = evaluateQuality(faceData);
    expect(result.checks.brightness.passed).toBe(false);
    expect(result.failureReason).toContain('dark');
  });

  it('too bright image fails brightness check', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.95);
    const result = evaluateQuality(faceData);
    expect(result.checks.brightness.passed).toBe(false);
    expect(result.failureReason).toContain('bright');
  });

  it('normal brightness passes', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const result = evaluateQuality(faceData);
    expect(result.checks.brightness.passed).toBe(true);
  });

  it('small face fails size check', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const bounds = { width: 50, height: 50 };
    const result = evaluateQuality(faceData, bounds);
    expect(result.checks.faceSize.passed).toBe(false);
  });

  it('large face passes size check', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const bounds = { width: 150, height: 150 };
    const result = evaluateQuality(faceData, bounds);
    expect(result.checks.faceSize.passed).toBe(true);
  });

  it('extreme yaw angle fails pose check', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const landmarks = {
      leftEye: { x: 20, y: 50 },
      rightEye: { x: 90, y: 50 },
      nose: { x: 80, y: 60 },
    };
    const result = evaluateQuality(faceData, undefined, landmarks);
    expect(result.checks.poseAngle.passed).toBe(false);
  });

  it('centered face passes pose check', () => {
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const landmarks = {
      leftEye: { x: 35, y: 50 },
      rightEye: { x: 75, y: 50 },
      nose: { x: 55, y: 55 },
    };
    const result = evaluateQuality(faceData, undefined, landmarks);
    expect(result.checks.poseAngle.passed).toBe(true);
  });
});
