import { describe, it, expect, beforeEach } from 'vitest';
import { InMemorySyncQueue } from '../sync/SyncQueue';
import type { SyncQueueItem } from '../sync/SyncQueue';

function makeItem(userId: string, createdAt?: number): SyncQueueItem {
  return {
    userId,
    embedding: new Float32Array([1, 0, 0]),
    metadata: { test: true },
    createdAt: createdAt ?? Date.now(),
    retryCount: 0,
  };
}

describe('InMemorySyncQueue', () => {
  let queue: InMemorySyncQueue;

  beforeEach(() => {
    queue = new InMemorySyncQueue();
  });

  it('starts empty', () => {
    expect(queue.count()).toBe(0);
  });

  it('enqueues items', () => {
    queue.enqueue(makeItem('user1'));
    queue.enqueue(makeItem('user2'));
    expect(queue.count()).toBe(2);
  });

  it('peeks without removing', () => {
    queue.enqueue(makeItem('user1', 100));
    queue.enqueue(makeItem('user2', 200));

    const batch = queue.peek(10);
    expect(batch.length).toBe(2);
    expect(queue.count()).toBe(2);
  });

  it('peeks in FIFO order', () => {
    queue.enqueue(makeItem('user2', 200));
    queue.enqueue(makeItem('user1', 100));
    queue.enqueue(makeItem('user3', 300));

    const batch = queue.peek(10);
    expect(batch[0]!.userId).toBe('user1');
    expect(batch[1]!.userId).toBe('user2');
    expect(batch[2]!.userId).toBe('user3');
  });

  it('respects peek limit', () => {
    queue.enqueue(makeItem('user1', 100));
    queue.enqueue(makeItem('user2', 200));
    queue.enqueue(makeItem('user3', 300));

    const batch = queue.peek(2);
    expect(batch.length).toBe(2);
    expect(batch[0]!.userId).toBe('user1');
    expect(batch[1]!.userId).toBe('user2');
  });

  it('removes items by userId', () => {
    queue.enqueue(makeItem('user1'));
    queue.enqueue(makeItem('user2'));
    queue.enqueue(makeItem('user3'));

    queue.remove(['user1', 'user3']);
    expect(queue.count()).toBe(1);

    const remaining = queue.peek(10);
    expect(remaining[0]!.userId).toBe('user2');
  });

  it('marks items as retried', () => {
    queue.enqueue(makeItem('user1'));
    queue.markRetried(['user1']);

    const items = queue.peek(10);
    expect(items[0]!.retryCount).toBe(1);
  });

  it('purges expired items', () => {
    queue.enqueue(makeItem('user1'));
    queue.enqueue(makeItem('user2'));
    queue.markRetried(['user1']);
    queue.markRetried(['user1']);
    queue.markRetried(['user1']);

    const expired = queue.purgeExpired(3);
    expect(expired).toContain('user1');
    expect(queue.count()).toBe(1);
  });

  it('does not purge items below max retries', () => {
    queue.enqueue(makeItem('user1'));
    queue.markRetried(['user1']);

    const expired = queue.purgeExpired(3);
    expect(expired.length).toBe(0);
    expect(queue.count()).toBe(1);
  });

  it('clears all items', () => {
    queue.enqueue(makeItem('user1'));
    queue.enqueue(makeItem('user2'));
    queue.clear();
    expect(queue.count()).toBe(0);
  });

  it('disposes cleanly', () => {
    queue.enqueue(makeItem('user1'));
    queue.dispose();
    expect(queue.count()).toBe(0);
  });

  it('overwrites duplicate userId on enqueue', () => {
    queue.enqueue(makeItem('user1', 100));
    queue.enqueue(makeItem('user1', 200));
    expect(queue.count()).toBe(1);

    const items = queue.peek(10);
    expect(items[0]!.createdAt).toBe(200);
  });
});
