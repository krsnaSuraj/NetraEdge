/**
 * Sync queue — persists pending enrollment operations for offline-to-online sync.
 *
 * When the device is offline, enrollments are queued locally.
 * When connectivity returns, the queue drains to the server.
 * Only after successful upload are items removed from the queue.
 *
 * Design:
 * - FIFO ordering by creation time
 * - Retry tracking per item
 * - Batch peek for efficient upload
 * - Purge support for post-sync cleanup
 */

export interface SyncQueueItem {
  readonly userId: string;
  readonly embedding: Float32Array;
  readonly metadata: Record<string, unknown>;
  readonly createdAt: number;
  readonly retryCount: number;
}

export interface SyncQueue {
  /** Add an enrollment to the queue */
  enqueue(item: SyncQueueItem): void;

  /** Remove items by userId after successful sync */
  remove(userIds: readonly string[]): void;

  /** Peek at the next batch without removing */
  peek(count: number): readonly SyncQueueItem[];

  /** Get total pending count */
  count(): number;

  /** Increment retry count for failed items */
  markRetried(userIds: readonly string[]): void;

  /** Remove items exceeding max retry count */
  purgeExpired(maxRetries: number): readonly string[];

  /** Clear all queued items */
  clear(): void;

  /** Release resources */
  dispose(): void;
}

/**
 * In-memory sync queue implementation.
 *
 * For production, use SQLiteSyncQueueStore (in @netraedge/react-native)
 * which persists across app restarts.
 */
export class InMemorySyncQueue implements SyncQueue {
  private readonly _items = new Map<string, SyncQueueItem>();

  enqueue(item: SyncQueueItem): void {
    this._items.set(item.userId, item);
  }

  remove(userIds: readonly string[]): void {
    for (const id of userIds) {
      this._items.delete(id);
    }
  }

  peek(count: number): readonly SyncQueueItem[] {
    const sorted = Array.from(this._items.values()).sort(
      (a, b) => a.createdAt - b.createdAt,
    );
    return sorted.slice(0, count);
  }

  count(): number {
    return this._items.size;
  }

  markRetried(userIds: readonly string[]): void {
    for (const id of userIds) {
      const existing = this._items.get(id);
      if (existing) {
        this._items.set(id, {
          ...existing,
          retryCount: existing.retryCount + 1,
        });
      }
    }
  }

  purgeExpired(maxRetries: number): readonly string[] {
    const expired: string[] = [];
    for (const [userId, item] of this._items) {
      if (item.retryCount >= maxRetries) {
        this._items.delete(userId);
        expired.push(userId);
      }
    }
    return expired;
  }

  clear(): void {
    this._items.clear();
  }

  dispose(): void {
    this._items.clear();
  }
}
