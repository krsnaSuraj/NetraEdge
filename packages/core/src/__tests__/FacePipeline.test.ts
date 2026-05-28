import { describe, it, expect } from 'vitest';
import { StubEncoder } from '../embedding/Encoder';
import { InMemoryEmbeddingStore } from '../embedding/EmbeddingStore';
import { FacePipeline } from '../pipeline/FacePipeline';
import { StubTextureAnalyzer } from '../liveness/TextureAnalyzer';
import { LivenessOrchestrator } from '../liveness/LivenessOrchestrator';

describe('FacePipeline', () => {
  function createPipeline() {
    const encoder = new StubEncoder(128);
    const store = new InMemoryEmbeddingStore();
    const textureAnalyzer = new StubTextureAnalyzer();
    const liveness = new LivenessOrchestrator(textureAnalyzer);

    return new FacePipeline(encoder, store, liveness, {
      recognitionThreshold: 0.75,
      enrollmentFrameCount: 10,
      requireLiveness: true,
    });
  }

  it('enrolls a user successfully', async () => {
    const pipeline = createPipeline();
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const meshPoints = [{ x: 0.5, y: 0.5, z: 0.5 }];

    const result = await pipeline.enroll(
      'user1',
      [faceData, faceData, faceData],
      [meshPoints, meshPoints, meshPoints],
      Date.now(),
    );

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.userId).toBe('user1');
      expect(result.value.frameCount).toBe(3);
    }
  });

  it('rejects enrollment with no frames', async () => {
    const pipeline = createPipeline();
    const result = await pipeline.enroll('user1', [], [], Date.now());
    expect(result.ok).toBe(false);
  });

  it('verifies a face against enrolled user', async () => {
    const pipeline = createPipeline();
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const meshPoints = [{ x: 0.5, y: 0.5, z: 0.5 }];

    await pipeline.enroll(
      'user1',
      [faceData, faceData, faceData],
      [meshPoints, meshPoints, meshPoints],
      Date.now(),
    );

    const verifyResult = await pipeline.verify(
      faceData,
      meshPoints,
      Date.now(),
      true,
    );

    expect(verifyResult.ok).toBe(true);
    if (verifyResult.ok) {
      expect(verifyResult.value.matched).toBe(true);
      expect(verifyResult.value.userId).toBe('user1');
    }
  });

  it('lists enrolled users', async () => {
    const pipeline = createPipeline();
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const meshPoints = [{ x: 0.5, y: 0.5, z: 0.5 }];

    await pipeline.enroll('user1', [faceData], [meshPoints], Date.now());
    await pipeline.enroll('user2', [faceData], [meshPoints], Date.now());

    const users = await pipeline.listUsers();
    expect(users).toContain('user1');
    expect(users).toContain('user2');
  });

  it('removes a user', async () => {
    const pipeline = createPipeline();
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const meshPoints = [{ x: 0.5, y: 0.5, z: 0.5 }];

    await pipeline.enroll('user1', [faceData], [meshPoints], Date.now());
    const removed = await pipeline.removeUser('user1');
    expect(removed).toBe(true);

    const users = await pipeline.listUsers();
    expect(users).not.toContain('user1');
  });

  it('clears all enrollments', async () => {
    const pipeline = createPipeline();
    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const meshPoints = [{ x: 0.5, y: 0.5, z: 0.5 }];

    await pipeline.enroll('user1', [faceData], [meshPoints], Date.now());
    await pipeline.enroll('user2', [faceData], [meshPoints], Date.now());

    await pipeline.clearEnrollments();
    expect(await pipeline.enrollmentCount()).toBe(0);
  });

  it('reports enrollment count', async () => {
    const pipeline = createPipeline();
    expect(await pipeline.enrollmentCount()).toBe(0);

    const faceData = new Float32Array(112 * 112 * 3).fill(0.5);
    const meshPoints = [{ x: 0.5, y: 0.5, z: 0.5 }];

    await pipeline.enroll('user1', [faceData], [meshPoints], Date.now());
    expect(await pipeline.enrollmentCount()).toBe(1);
  });
});
