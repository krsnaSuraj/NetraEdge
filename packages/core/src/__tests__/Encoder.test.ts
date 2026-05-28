import { describe, it, expect } from 'vitest';
import { StubEncoder } from '../embedding/Encoder';
import { l2Normalize } from '../embedding/CosineSimilarity';

describe('StubEncoder', () => {
  it('reports as loaded', () => {
    const encoder = new StubEncoder();
    expect(encoder.isLoaded).toBe(true);
  });

  it('returns 128-d embedding by default', () => {
    const encoder = new StubEncoder();
    const faceData = new Uint8Array(112 * 112 * 3).fill(128);
    const embedding = encoder.encode(faceData);
    expect(embedding).not.toBeNull();
    expect(embedding!.length).toBe(128);
  });

  it('returns deterministic embeddings', () => {
    const encoder = new StubEncoder();
    const faceData = new Uint8Array(112 * 112 * 3).fill(128);
    const emb1 = encoder.encode(faceData);
    const emb2 = encoder.encode(faceData);
    expect(emb1).toEqual(emb2);
  });

  it('returns L2-normalized embedding', () => {
    const encoder = new StubEncoder();
    const faceData = new Uint8Array(112 * 112 * 3).fill(128);
    const embedding = encoder.encode(faceData)!;

    let magnitude = 0;
    for (let i = 0; i < embedding.length; i++) {
      magnitude += embedding[i]! * embedding[i]!;
    }
    expect(Math.sqrt(magnitude)).toBeCloseTo(1.0, 3);
  });

  it('can be disposed', () => {
    const encoder = new StubEncoder();
    expect(encoder.isLoaded).toBe(true);
    encoder.dispose();
    expect(encoder.isLoaded).toBe(false);
  });

  it('supports custom dimensions', () => {
    const encoder = new StubEncoder(64);
    const faceData = new Uint8Array(112 * 112 * 3).fill(128);
    const embedding = encoder.encode(faceData);
    expect(embedding!.length).toBe(64);
  });
});
