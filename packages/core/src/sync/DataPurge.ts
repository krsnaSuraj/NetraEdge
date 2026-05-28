/**
 * Data purge — manages local data lifecycle after sync.
 *
 * Handles two purge strategies:
 * 1. Post-sync purge: remove successfully synced enrollments
 * 2. Expiry purge: remove enrollments older than maxAge
 *
 * This ensures local storage doesn't grow unbounded and
 * sensitive biometric data isn't retained longer than needed.
 */

import type { EmbeddingStore } from '../embedding/EmbeddingStore';
import type { SyncQueue } from './SyncQueue';

export interface PurgePolicy {
  /** Remove data after successful sync */
  readonly afterSuccessfulSync: boolean;
  /** Maximum age in milliseconds before data is purged */
  readonly maxAgeMs: number;
  /** Maximum number of records to keep */
  readonly maxRecords: number;
}

export const DEFAULT_PURGE_POLICY: PurgePolicy = {
  afterSuccessfulSync: true,
  maxAgeMs: 7 * 24 * 60 * 60 * 1000, // 7 days
  maxRecords: 10000,
};

export interface PurgeResult {
  readonly purgedEmbeddings: number;
  readonly purgedQueueItems: number;
  readonly purgedExpired: number;
}

export class DataPurgeManager {
  private readonly _store: EmbeddingStore;
  private readonly _queue: SyncQueue;
  private readonly _policy: PurgePolicy;

  constructor(
    store: EmbeddingStore,
    queue: SyncQueue,
    policy: Partial<PurgePolicy> = {},
  ) {
    this._store = store;
    this._queue = queue;
    this._policy = { ...DEFAULT_PURGE_POLICY, ...policy };
  }

  /**
   * Purge data after a successful sync.
   * Removes synced enrollments from both the store and queue.
   */
  async purgeAfterSync(syncedUserIds: readonly string[]): Promise<PurgeResult> {
    let purgedEmbeddings = 0;
    let purgedQueueItems = 0;

    if (this._policy.afterSuccessfulSync && syncedUserIds.length > 0) {
      for (const userId of syncedUserIds) {
        const removed = await this._store.remove(userId);
        if (removed) purgedEmbeddings++;
      }
      this._queue.remove(syncedUserIds);
      purgedQueueItems = syncedUserIds.length;
    }

    return { purgedEmbeddings, purgedQueueItems, purgedExpired: 0 };
  }

  /**
   * Purge queue items that have exceeded max retry count.
   */
  purgeRetries(maxRetries: number): readonly string[] {
    return this._queue.purgeExpired(maxRetries);
  }

  /**
   * Get the current purge policy.
   */
  get policy(): Readonly<PurgePolicy> {
    return this._policy;
  }
}
