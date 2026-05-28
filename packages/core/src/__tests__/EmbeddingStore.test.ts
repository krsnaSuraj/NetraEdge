import { describe, it, expect, beforeEach } from 'vitest';
import { InMemoryEmbeddingStore } from '../embedding/EmbeddingStore';

describe('InMemoryEmbeddingStore', () => {
  let store: InMemoryEmbeddingStore;

  beforeEach(() => {
    store = new InMemoryEmbeddingStore();
  });

  it('enrolls a user and stores embedding', async () => {
    const embedding = new Float32Array([1, 0, 0]);
    const result = await store.enroll('user1', [embedding]);
    expect(result).toBe(true);
  });

  it('rejects empty embeddings', async () => {
    const result = await store.enroll('user1', []);
    expect(result).toBe(false);
  });

  it('identifies enrolled user', async () => {
    const embedding = new Float32Array([1, 0, 0]);
    await store.enroll('user1', [embedding]);

    const probe = new Float32Array([1, 0, 0]);
    const result = await store.identify(probe, 0.75);
    expect(result).not.toBeNull();
    expect(result!.userId).toBe('user1');
    expect(result!.confidence).toBeCloseTo(1.0, 2);
  });

  it('returns null for no match', async () => {
    const embedding = new Float32Array([1, 0, 0]);
    await store.enroll('user1', [embedding]);

    const probe = new Float32Array([0, 0, 1]);
    const result = await store.identify(probe, 0.75);
    expect(result).toBeNull();
  });

  it('returns best match among multiple users', async () => {
    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    await store.enroll('user2', [new Float32Array([0, 1, 0])]);
    await store.enroll('user3', [new Float32Array([0, 0, 1])]);

    const probe = new Float32Array([0.9, 0.1, 0]);
    const result = await store.identify(probe, 0.5);
    expect(result).not.toBeNull();
    expect(result!.userId).toBe('user1');
  });

  it('lists enrolled users', async () => {
    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    await store.enroll('user2', [new Float32Array([0, 1, 0])]);

    const users = await store.listUsers();
    expect(users).toContain('user1');
    expect(users).toContain('user2');
  });

  it('removes a user', async () => {
    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    const removed = await store.remove('user1');
    expect(removed).toBe(true);

    const users = await store.listUsers();
    expect(users).not.toContain('user1');
  });

  it('returns correct count', async () => {
    expect(await store.count()).toBe(0);

    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    await store.enroll('user2', [new Float32Array([0, 1, 0])]);
    expect(await store.count()).toBe(2);
  });

  it('clears all enrollments', async () => {
    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    await store.enroll('user2', [new Float32Array([0, 1, 0])]);
    await store.clear();
    expect(await store.count()).toBe(0);
  });

  it('averages multiple embeddings', async () => {
    const emb1 = new Float32Array([1, 0, 0]);
    const emb2 = new Float32Array([0.9, 0.1, 0]);
    await store.enroll('user1', [emb1, emb2]);

    const probe = new Float32Array([1, 0, 0]);
    const result = await store.identify(probe, 0.5);
    expect(result).not.toBeNull();
    expect(result!.userId).toBe('user1');
  });
});
