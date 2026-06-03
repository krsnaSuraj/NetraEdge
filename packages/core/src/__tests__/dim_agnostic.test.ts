import { describe, it, expect } from 'vitest';
import { StubEncoder, SOTA_EMBEDDING_DIM, flipFaceHorizontal } from '../embedding/Encoder';
import { InMemoryEmbeddingStore } from '../embedding/EmbeddingStore';

describe('Dimension-agnostic encoder + store + TTA (Path A+)', () => {
  it('SOTA_EMBEDDING_DIM is 512 (Path A+: EdgeFace-XS target)', () => {
    expect(SOTA_EMBEDDING_DIM).toBe(512);
  });

  it('StubEncoder reports its dim and produces that-length embeddings', async () => {
    for (const dim of [64, 128, 192, 256, 384, 512]) {
      const enc = new StubEncoder(dim);
      expect(enc.embeddingDim).toBe(dim);
      expect(enc.isLoaded).toBe(true);
      const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
      const emb = await enc.encode(faceData);
      expect(emb).not.toBeNull();
      expect(emb!.length).toBe(dim);
    }
  });

  it('StubEncoder reports -1 when disposed', () => {
    const enc = new StubEncoder(512);
    enc.dispose();
    expect(enc.isLoaded).toBe(false);
    expect(enc.embeddingDim).toBe(-1);
  });

  it('InMemoryEmbeddingStore works with 128-d (current pre-trained model)', async () => {
    const store = new InMemoryEmbeddingStore();
    const a = new Float32Array(128).fill(0.1);
    const b = new Float32Array(128).fill(0.1);
    await store.enroll('alice', [a, b]);
    const result = await store.identify(a, 0.9);
    expect(result).not.toBeNull();
    expect(result!.userId).toBe('alice');
  });

  it('InMemoryEmbeddingStore works with 192-d (legacy GhostFaceNet-W1 dim)', async () => {
    const store = new InMemoryEmbeddingStore();
    const a = new Float32Array(192).fill(0.1);
    const b = new Float32Array(192).fill(0.1);
    await store.enroll('bob', [a, b]);
    const result = await store.identify(a, 0.9);
    expect(result).not.toBeNull();
    expect(result!.userId).toBe('bob');
  });

  it('InMemoryEmbeddingStore works with 512-d (Path A+ EdgeFace-XS target)', async () => {
    const store = new InMemoryEmbeddingStore();
    const a = new Float32Array(512).fill(0.1);
    const b = new Float32Array(512).fill(0.1);
    await store.enroll('charlie', [a, b]);
    const result = await store.identify(a, 0.9);
    expect(result).not.toBeNull();
    expect(result!.userId).toBe('charlie');
  });

  it('128-d probe does NOT match a 192-d enrollment (different dims)', async () => {
    const store = new InMemoryEmbeddingStore();
    const enroll = new Float32Array(192).fill(0.1);
    await store.enroll('diana', [enroll]);
    const probe128 = new Float32Array(128).fill(0.1);
    // The store now rejects dim mismatches — no spurious partial match.
    const result = await store.identify(probe128, 0.9);
    expect(result).toBeNull();
  });

  it('128-d probe does NOT match a 512-d enrollment (different dims)', async () => {
    const store = new InMemoryEmbeddingStore();
    const enroll = new Float32Array(512).fill(0.1);
    await store.enroll('evan', [enroll]);
    const probe128 = new Float32Array(128).fill(0.1);
    const result = await store.identify(probe128, 0.9);
    expect(result).toBeNull();
  });

  it('Mixed-dim enrollment batch throws on enroll', async () => {
    const store = new InMemoryEmbeddingStore();
    const a = new Float32Array(128).fill(0.1);
    const b = new Float32Array(192).fill(0.1);
    await expect(store.enroll('fred', [a, b])).rejects.toThrow(/dim mismatch/i);
  });

  it('TTA defaults to true (Path A+ recommendation)', () => {
    const enc = new StubEncoder(512);
    expect(enc.useTTA).toBe(true);
  });

  it('setUseTTA toggles the flag', () => {
    const enc = new StubEncoder(512);
    expect(enc.useTTA).toBe(true);
    enc.setUseTTA(false);
    expect(enc.useTTA).toBe(false);
    enc.setUseTTA(true);
    expect(enc.useTTA).toBe(true);
  });

  it('TTA disabled produces a single L2-normalized embedding', async () => {
    const enc = new StubEncoder(512);
    enc.setUseTTA(false);
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const e1 = await enc.encode(faceData);
    const e2 = await enc.encode(faceData);
    // Deterministic encoder + TTA off + same input → identical embeddings
    expect(e1).not.toBeNull();
    expect(e1!.length).toBe(512);
    let norm = 0;
    for (const v of e1!) norm += v * v;
    expect(Math.abs(Math.sqrt(norm) - 1.0)).toBeLessThan(1e-5);
    for (let i = 0; i < e1!.length; i++) expect(e1![i]).toBeCloseTo(e2![i]!, 6);
  });

  it('TTA enabled averages the original and flipped embeddings (norm preserved)', async () => {
    const enc = new StubEncoder(512);
    enc.setUseTTA(true);
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const e = await enc.encode(faceData);
    expect(e).not.toBeNull();
    expect(e!.length).toBe(512);
    let norm = 0;
    for (const v of e) norm += v * v;
    // After L2 normalize, the norm must be 1
    expect(Math.abs(Math.sqrt(norm) - 1.0)).toBeLessThan(1e-5);
  });

  it('flipFaceHorizontal reverses pixels within each row, keeps channels grouped', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    // Pixel (y=0, x=0) = R=1, G=2, B=3
    faceData[0] = 1; faceData[1] = 2; faceData[2] = 3;
    // Pixel (y=0, x=1) = R=4, G=5, B=6
    faceData[3] = 4; faceData[4] = 5; faceData[5] = 6;
    // Pixel (y=0, x=111) = R=7, G=8, B=9
    const lastIdx = (111) * 3;
    faceData[lastIdx] = 7; faceData[lastIdx + 1] = 8; faceData[lastIdx + 2] = 9;

    const flipped = flipFaceHorizontal(faceData);
    // After flip, pixel (y=0, x=0) should be the OLD pixel (y=0, x=111) = (7,8,9)
    expect(flipped[0]).toBe(7);
    expect(flipped[1]).toBe(8);
    expect(flipped[2]).toBe(9);
    // And pixel (y=0, x=111) should be the OLD pixel (y=0, x=0) = (1,2,3)
    expect(flipped[lastIdx]).toBe(1);
    expect(flipped[lastIdx + 1]).toBe(2);
    expect(flipped[lastIdx + 2]).toBe(3);
    // Middle pixel (y=0, x=1) should now hold the OLD (y=0, x=110) value
    // which is 0 (unchanged test data) — just verify the function runs end-to-end.
    expect(flipped.length).toBe(faceData.length);
  });
});
