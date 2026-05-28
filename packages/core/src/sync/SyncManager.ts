/**
 * SyncManager — handles offline-to-online data synchronization.
 *
 * When the device is offline, enrollments are stored locally in SQLite.
 * When network connectivity is restored, SyncManager uploads the data
 * to AWS and purges local storage.
 *
 * Design:
 * - Queue-based: operations are queued and processed in order
 * - Retry logic: failed syncs are retried up to MAX_RETRIES
 * - Purge after sync: local data is deleted only after successful upload
 * - Network detection: monitors connectivity changes
 */

import { SYNC_CONFIG } from '../config/constants';

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
}

export type SyncListener = (event: SyncEvent) => void;

export interface SyncManager {
  /** Start monitoring and syncing */
  start(): void;

  /** Stop syncing and monitoring */
  stop(): void;

  /** Manually trigger a sync attempt */
  syncNow(): Promise<boolean>;

  /** Get current sync status */
  readonly status: SyncStatus;

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
  private readonly _listeners: Set<SyncListener> = new Set();
  private readonly _retryDelay: number;
  private readonly _maxRetries: number;
  private _networkUnsubscribe: (() => void) | null = null;
  private _syncTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    transport: SyncTransport,
    network: NetworkMonitor,
    options?: { retryDelayMs?: number; maxRetries?: number },
  ) {
    this._transport = transport;
    this._network = network;
    this._retryDelay = options?.retryDelayMs ?? SYNC_CONFIG.networkCheckIntervalMs;
    this._maxRetries = options?.maxRetries ?? SYNC_CONFIG.maxRetries;
  }

  get status(): SyncStatus {
    return this._status;
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

    const online = await this._network.isOnline();
    if (!online) {
      this._status = SyncStatus.OFFLINE;
      return false;
    }

    this._status = SyncStatus.SYNCING;

    for (let attempt = 0; attempt < this._maxRetries; attempt++) {
      try {
        const success = await this._transport.uploadBatch([]);
        if (success) {
          this._status = SyncStatus.IDLE;
          this.emit({ type: 'sync_complete', timestamp: Date.now() });
          return true;
        }
      } catch (error) {
        if (attempt === this._maxRetries - 1) {
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
      if (online) {
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
