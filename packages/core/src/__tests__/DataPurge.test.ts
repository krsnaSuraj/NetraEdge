import { describe, it, expect, beforeEach } from 'vitest';
import { DataPurgeManager } from '../sync/DataPurge';
import { InMemoryEmbeddingStore } from '../embedding/EmbeddingStore';
import { InMemorySyncQueue } from '../sync/SyncQueue';

describe('DataPurgeManager', () => {
  let store: InMemoryEmbeddingStore;
  let queue: InMemorySyncQueue;
  let purge: DataPurgeManager;

  beforeEach(() => {
    store = new InMemoryEmbeddingStore();
    queue = new InMemorySyncQueue();
    purge = new DataPurgeManager(store, queue, {
      afterSuccessfulSync: true,
    });
  });

  it('purges synced embeddings from store', async () => {
    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    await store.enroll('user2', [new Float32Array([0, 1, 0])]);

    const result = await purge.purgeAfterSync(['user1']);
    expect(result.purgedEmbeddings).toBe(1);
    expect(await store.count()).toBe(1);
    expect(await store.listUsers()).toContain('user2');
  });

  it('purges synced items from queue', async () => {
    queue.enqueue({
      userId: 'user1',
      embedding: new Float32Array([1, 0, 0]),
      metadata: {},
      createdAt: Date.now(),
      retryCount: 0,
    });

    await purge.purgeAfterSync(['user1']);
    expect(queue.count()).toBe(0);
  });

  it('skips purge when afterSuccessfulSync is false', async () => {
    const noPurge = new DataPurgeManager(store, queue, {
      afterSuccessfulSync: false,
    });

    await store.enroll('user1', [new Float32Array([1, 0, 0])]);
    const result = await noPurge.purgeAfterSync(['user1']);
    expect(result.purgedEmbeddings).toBe(0);
    expect(await store.count()).toBe(1);
  });

  it('purges retries exceeding max', () => {
    queue.enqueue({
      userId: 'user1',
      embedding: new Float32Array([1, 0, 0]),
      metadata: {},
      createdAt: Date.now(),
      retryCount: 0,
    });
    queue.enqueue({
      userId: 'user2',
      embedding: new Float32Array([0, 1, 0]),
      metadata: {},
      createdAt: Date.now(),
      retryCount: 0,
    });

    queue.markRetried(['user1']);
    queue.markRetried(['user1']);
    queue.markRetried(['user1']);

    const expired = purge.purgeRetries(3);
    expect(expired).toContain('user1');
    expect(queue.count()).toBe(1);
  });

  it('returns default purge policy', () => {
    expect(purge.policy.afterSuccessfulSync).toBe(true);
    expect(purge.policy.maxAgeMs).toBeGreaterThan(0);
    expect(purge.policy.maxRecords).toBeGreaterThan(0);
  });
});
