import { describe, it, expect } from 'vitest';
import { DepthEstimator } from '../liveness/DepthEstimator';
import type { Point3D } from '../types/Face';

describe('DepthEstimator', () => {
  const estimator = new DepthEstimator(0.25);

  it('returns false for flat points (all same z)', () => {
    const points: Point3D[] = Array.from({ length: 100 }, (_, i) => ({
      x: i * 0.01,
      y: i * 0.01,
      z: 0.5,
    }));
    const result = estimator.analyze(points);
    expect(result.isThreeDimensional).toBe(false);
    expect(result.variance).toBeCloseTo(0, 5);
  });

  it('returns true for 3D face points (varying z)', () => {
    const points: Point3D[] = Array.from({ length: 100 }, (_, i) => ({
      x: i * 0.01,
      y: i * 0.01,
      z: Math.sin(i * 0.3) * 2.0,
    }));
    const result = estimator.analyze(points);
    expect(result.isThreeDimensional).toBe(true);
    expect(result.variance).toBeGreaterThan(0.25);
  });

  it('returns false for empty points', () => {
    const result = estimator.analyze([]);
    expect(result.isThreeDimensional).toBe(false);
    expect(result.variance).toBe(0);
  });

  it('calculates variance correctly', () => {
    const points: Point3D[] = [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
      { x: 1, y: 1, z: 0 },
    ];
    const result = estimator.analyze(points);
    expect(result.variance).toBeCloseTo(0, 5);
  });
});
