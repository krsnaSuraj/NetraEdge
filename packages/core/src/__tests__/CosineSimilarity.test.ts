import { describe, it, expect } from 'vitest';
import { cosineSimilarity, l2Normalize } from '../embedding/CosineSimilarity';

describe('cosineSimilarity', () => {
  it('returns 1.0 for identical vectors', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
  });

  it('returns 0.0 for orthogonal vectors', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([0, 1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
  });

  it('returns -1.0 for opposite vectors', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([-1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 5);
  });

  it('clamps result to [-1, 1]', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([1, 0]);
    const result = cosineSimilarity(a, b);
    expect(result).toBeGreaterThanOrEqual(-1);
    expect(result).toBeLessThanOrEqual(1);
  });

  it('throws on dimension mismatch', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([1, 0]);
    expect(() => cosineSimilarity(a, b)).toThrow('Embedding dimension mismatch');
  });

  it('handles 128-d embeddings', () => {
    const a = new Float32Array(128).fill(0.1);
    const b = new Float32Array(128).fill(0.1);
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
  });

  it('handles negative values', () => {
    const a = new Float32Array([0.5, -0.5, 0.7]);
    const b = new Float32Array([0.5, -0.5, 0.7]);
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThan(0.98);
  });
});

describe('l2Normalize', () => {
  it('returns unit vector', () => {
    const vec = new Float32Array([3, 4]);
    const normalized = l2Normalize(vec);
    const magnitude = Math.sqrt(
      normalized[0]! * normalized[0]! + normalized[1]! * normalized[1]!,
    );
    expect(magnitude).toBeCloseTo(1.0, 5);
  });

  it('preserves direction', () => {
    const vec = new Float32Array([3, 4]);
    const normalized = l2Normalize(vec);
    expect(normalized[0]!).toBeCloseTo(0.6, 5);
    expect(normalized[1]!).toBeCloseTo(0.8, 5);
  });

  it('handles zero vector', () => {
    const vec = new Float32Array([0, 0, 0]);
    const normalized = l2Normalize(vec);
    expect(normalized[0]).toBe(0);
    expect(normalized[1]).toBe(0);
    expect(normalized[2]).toBe(0);
  });

  it('handles 128-d embedding', () => {
    const vec = new Float32Array(128).fill(0.1);
    const normalized = l2Normalize(vec);
    let magnitude = 0;
    for (let i = 0; i < 128; i++) {
      magnitude += normalized[i]! * normalized[i]!;
    }
    expect(Math.sqrt(magnitude)).toBeCloseTo(1.0, 5);
  });
});
