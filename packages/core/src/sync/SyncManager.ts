/**
 * SyncManager — handles offline-to-online data synchronization.
 *
 * When the device is offline, enrollments are queued locally via SyncQueue.
 * When network connectivity is restored, SyncManager drains the queue
 * to the server and purges local data on success.
 *
 * Design:
 * - Queue-based: pending operations stored in SyncQueue
 * - Retry logic: failed syncs retried up to maxRetries
 * - Purge after sync: local data deleted only after confirmed upload
 * - Network detection: monitors connectivity changes
 */

import { SYNC_CONFIG } from '../config/constants';
import type { SyncQueue } from './SyncQueue';
import type { DataPurgeManager } from './DataPurge';

export enum SyncStatus {
  IDLE = 'idle',
  SYNCING = 'syncing',
  OFFLINE = 'offline',
  ERROR = 'error',
}

export interface SyncEvent {
  readonly type: 'enrollment_uploaded' | 'sync_complete' | 'sync_failed' | 'purge_complete';
  readonly userId?: string;
  readonly timestamp: number;
  readonly error?: string;
  readonly count?: number;
}

export type SyncListener = (event: SyncEvent) => void;

export interface SyncManager {
  /** Start monitoring and syncing */
  start(): void;

  /** Stop syncing and monitoring */
  stop(): void;

  /** Add an enrollment to the sync queue */
  enqueue(userId: string, embedding: Float32Array, metadata?: Record<string, unknown>): void;

  /** Manually trigger a sync attempt */
  syncNow(): Promise<boolean>;

  /** Get current sync status */
  readonly status: SyncStatus;

  /** Get number of pending items */
  readonly pendingCount: number;

  /** Subscribe to sync events */
  onEvent(listener: SyncListener): () => void;

  /** Release resources */
  dispose(): void;
}

export interface SyncTransport {
  /** Upload a single enrollment to the server */
  uploadEnrollment(
    userId: string,
    embedding: Float32Array,
    metadata: Record<string, unknown>,
  ): Promise<boolean>;

  /** Upload a batch of enrollments */
  uploadBatch(
    enrollments: readonly {
      userId: string;
      embedding: Float32Array;
      metadata: Record<string, unknown>;
    }[],
  ): Promise<boolean>;
}

export interface NetworkMonitor {
  /** Check if network is available */
  isOnline(): Promise<boolean>;

  /** Register a callback for connectivity changes */
  onConnectivityChange(callback: (online: boolean) => void): () => void;
}

export class DefaultSyncManager implements SyncManager {
  private _status: SyncStatus = SyncStatus.IDLE;
  private readonly _transport: SyncTransport;
  private readonly _network: NetworkMonitor;
  private readonly _queue: SyncQueue;
  private readonly _purge: DataPurgeManager | null;
  private readonly _listeners: Set<SyncListener> = new Set();
  private readonly _retryDelay: number;
  private readonly _maxRetries: number;
  private readonly _batchSize: number;
  private _networkUnsubscribe: (() => void) | null = null;
  private _syncTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    transport: SyncTransport,
    network: NetworkMonitor,
    queue: SyncQueue,
    purge?: DataPurgeManager,
    options?: { retryDelayMs?: number; maxRetries?: number; batchSize?: number },
  ) {
    this._transport = transport;
    this._network = network;
    this._queue = queue;
    this._purge = purge ?? null;
    this._retryDelay = options?.retryDelayMs ?? SYNC_CONFIG.networkCheckIntervalMs;
    this._maxRetries = options?.maxRetries ?? SYNC_CONFIG.maxRetries;
    this._batchSize = options?.batchSize ?? SYNC_CONFIG.batchSize;
  }

  get status(): SyncStatus {
    return this._status;
  }

  get pendingCount(): number {
    return this._queue.count();
  }

  enqueue(userId: string, embedding: Float32Array, metadata: Record<string, unknown> = {}): void {
    this._queue.enqueue({
      userId,
      embedding,
      metadata,
      createdAt: Date.now(),
      retryCount: 0,
    });
  }

  start(): void {
    this._networkUnsubscribe = this._network.onConnectivityChange(
      (online) => {
        if (online && this._status !== SyncStatus.SYNCING) {
          void this.syncNow();
        }
      },
    );

    this._syncTimer = setInterval(() => {
      void this.checkAndSync();
    }, this._retryDelay);
  }

  stop(): void {
    this._networkUnsubscribe?.();
    this._networkUnsubscribe = null;

    if (this._syncTimer) {
      clearInterval(this._syncTimer);
      this._syncTimer = null;
    }
  }

  async syncNow(): Promise<boolean> {
    if (this._status === SyncStatus.SYNCING) return false;

    const pending = this._queue.peek(this._batchSize);
    if (pending.length === 0) return true;

    const online = await this._network.isOnline();
    if (!online) {
      this._status = SyncStatus.OFFLINE;
      return false;
    }

    this._status = SyncStatus.SYNCING;

    for (let attempt = 0; attempt < this._maxRetries; attempt++) {
      try {
        const success = await this._transport.uploadBatch(pending);
        if (success) {
          const syncedIds = pending.map((p) => p.userId);
          this._queue.remove(syncedIds);

          this._status = SyncStatus.IDLE;
          this.emit({
            type: 'sync_complete',
            timestamp: Date.now(),
            count: syncedIds.length,
          });

          if (this._purge) {
            await this._purge.purgeAfterSync(syncedIds);
            this.emit({ type: 'purge_complete', timestamp: Date.now() });
          }

          return true;
        }
      } catch (error) {
        if (attempt === this._maxRetries - 1) {
          this._queue.markRetried(pending.map((p) => p.userId));
          this._status = SyncStatus.ERROR;
          this.emit({
            type: 'sync_failed',
            timestamp: Date.now(),
            error: String(error),
          });
        }
      }

      await this.delay(this._retryDelay * (attempt + 1));
    }

    return false;
  }

  onEvent(listener: SyncListener): () => void {
    this._listeners.add(listener);
    return () => {
      this._listeners.delete(listener);
    };
  }

  dispose(): void {
    this.stop();
    this._listeners.clear();
  }

  private async checkAndSync(): Promise<void> {
    if (this._status === SyncStatus.IDLE || this._status === SyncStatus.OFFLINE) {
      const online = await this._network.isOnline();
      if (online && this._queue.count() > 0) {
        await this.syncNow();
      }
    }
  }

  private emit(event: SyncEvent): void {
    for (const listener of this._listeners) {
      try {
        listener(event);
      } catch {
        /* listener error should not break sync */
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
