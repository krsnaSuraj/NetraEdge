import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DefaultSyncManager, SyncStatus } from '../sync/SyncManager';
import type { SyncTransport, NetworkMonitor } from '../sync/SyncManager';
import { InMemorySyncQueue } from '../sync/SyncQueue';

function createMockTransport(): SyncTransport {
  return {
    uploadEnrollment: vi.fn().mockResolvedValue(true),
    uploadBatch: vi.fn().mockResolvedValue(true),
  };
}

function createMockNetwork(isOnline: boolean): NetworkMonitor {
  const listeners: Array<(online: boolean) => void> = [];
  const monitor: NetworkMonitor & { simulateOnline: () => void; simulateOffline: () => void } = {
    isOnline: vi.fn().mockResolvedValue(isOnline),
    onConnectivityChange: vi.fn((cb: (online: boolean) => void) => {
      listeners.push(cb);
      return () => {
        const idx = listeners.indexOf(cb);
        if (idx >= 0) listeners.splice(idx, 1);
      };
    }),
    simulateOnline: () => {
      for (const cb of listeners) cb(true);
    },
    simulateOffline: () => {
      for (const cb of listeners) cb(false);
    },
  };
  return monitor;
}

describe('DefaultSyncManager', () => {
  let transport: SyncTransport;
  let network: NetworkMonitor;
  let queue: InMemorySyncQueue;
  let manager: DefaultSyncManager;

  beforeEach(() => {
    transport = createMockTransport();
    network = createMockNetwork(true);
    queue = new InMemorySyncQueue();
    manager = new DefaultSyncManager(transport, network, queue, undefined, {
      retryDelayMs: 100,
      maxRetries: 2,
    });
  });

  afterEach(() => {
    manager.dispose();
    queue.dispose();
  });

  it('starts in IDLE status', () => {
    expect(manager.status).toBe(SyncStatus.IDLE);
  });

  it('syncs queued items when online', async () => {
    manager.enqueue('user1', new Float32Array([1, 0, 0]));
    manager.enqueue('user2', new Float32Array([0, 1, 0]));

    const result = await manager.syncNow();
    expect(result).toBe(true);
    expect(transport.uploadBatch).toHaveBeenCalled();
  });

  it('returns true when queue is empty', async () => {
    const result = await manager.syncNow();
    expect(result).toBe(true);
  });

  it('does not sync when offline', async () => {
    const offlineNetwork = createMockNetwork(false);
    const offlineQueue = new InMemorySyncQueue();
    const offlineManager = new DefaultSyncManager(transport, offlineNetwork, offlineQueue);
    offlineManager.enqueue('user1', new Float32Array([1, 0, 0]));

    const result = await offlineManager.syncNow();
    expect(result).toBe(false);
    expect(transport.uploadBatch).not.toHaveBeenCalled();
    offlineManager.dispose();
    offlineQueue.dispose();
  });

  it('emits sync_complete event on success', async () => {
    manager.enqueue('user1', new Float32Array([1, 0, 0]));

    const listener = vi.fn();
    manager.onEvent(listener);

    await manager.syncNow();

    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'sync_complete' }),
    );
  });

  it('emits sync_failed after max retries', async () => {
    (transport.uploadBatch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Network error'));
    manager.enqueue('user1', new Float32Array([1, 0, 0]));

    const listener = vi.fn();
    manager.onEvent(listener);

    await manager.syncNow();

    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'sync_failed' }),
    );
  });

  it('unsubscribes listener', async () => {
    const listener = vi.fn();
    const unsubscribe = manager.onEvent(listener);
    unsubscribe();

    manager.enqueue('user1', new Float32Array([1, 0, 0]));
    await manager.syncNow();

    expect(listener).not.toHaveBeenCalled();
  });

  it('tracks pending count', () => {
    expect(manager.pendingCount).toBe(0);
    manager.enqueue('user1', new Float32Array([1, 0, 0]));
    expect(manager.pendingCount).toBe(1);
  });

  it('disposes cleanly', () => {
    manager.dispose();
    expect(manager.status).toBe(SyncStatus.IDLE);
  });
});
