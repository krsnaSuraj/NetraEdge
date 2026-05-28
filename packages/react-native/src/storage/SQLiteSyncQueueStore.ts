/**
 * SQLite-backed sync queue — persists pending operations across restarts.
 *
 * When the app goes offline, enrollment data is queued here.
 * On reconnect, the queue drains to the server. Only after
 * confirmed upload are items removed.
 */

import type { SyncQueue, SyncQueueItem } from '@netraedge/core';
import type { SQLiteDatabase } from './SQLiteEmbeddingStore';

export class SQLiteSyncQueueStore implements SyncQueue {
  private readonly _db: SQLiteDatabase;
  private _disposed = false;

  constructor(db: SQLiteDatabase) {
    this._db = db;
    this.initialize();
  }

  private initialize(): void {
    this._db.execute(`
      CREATE TABLE IF NOT EXISTS sync_queue (
        user_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        metadata TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0
      )
    `);
    this._db.execute(
      'CREATE INDEX IF NOT EXISTS idx_sync_queue_created ON sync_queue(created_at)',
    );
  }

  enqueue(item: SyncQueueItem): void {
    if (this._disposed) return;

    const blob = item.embedding.buffer.slice(
      item.embedding.byteOffset,
      item.embedding.byteOffset + item.embedding.byteLength,
    );

    this._db.execute(
      'INSERT OR REPLACE INTO sync_queue (user_id, embedding, metadata, created_at, retry_count) VALUES (?, ?, ?, ?, ?)',
      [item.userId, blob, JSON.stringify(item.metadata), item.createdAt, item.retryCount],
    );
  }

  remove(userIds: readonly string[]): void {
    if (this._disposed) return;

    for (const id of userIds) {
      this._db.execute('DELETE FROM sync_queue WHERE user_id = ?', [id]);
    }
  }

  peek(count: number): readonly SyncQueueItem[] {
    if (this._disposed) return [];

    const { rows } = this._db.execute(
      'SELECT user_id, embedding, metadata, created_at, retry_count FROM sync_queue ORDER BY created_at ASC LIMIT ?',
      [count],
    );

    return (rows as Array<{
      user_id: string;
      embedding: ArrayBuffer;
      metadata: string;
      created_at: number;
      retry_count: number;
    }>).map((row) => ({
      userId: row.user_id,
      embedding: new Float32Array(row.embedding),
      metadata: JSON.parse(row.metadata) as Record<string, unknown>,
      createdAt: row.created_at,
      retryCount: row.retry_count,
    }));
  }

  count(): number {
    if (this._disposed) return 0;

    const { rows } = this._db.execute(
      'SELECT COUNT(*) as cnt FROM sync_queue',
    );

    return (rows as Array<{ cnt: number }>)[0]?.cnt ?? 0;
  }

  markRetried(userIds: readonly string[]): void {
    if (this._disposed) return;

    for (const id of userIds) {
      this._db.execute(
        'UPDATE sync_queue SET retry_count = retry_count + 1 WHERE user_id = ?',
        [id],
      );
    }
  }

  purgeExpired(maxRetries: number): readonly string[] {
    if (this._disposed) return [];

    const { rows } = this._db.execute(
      'SELECT user_id FROM sync_queue WHERE retry_count >= ?',
      [maxRetries],
    );

    const expired = (rows as Array<{ user_id: string }>).map((r) => r.user_id);

    for (const id of expired) {
      this._db.execute('DELETE FROM sync_queue WHERE user_id = ?', [id]);
    }

    return expired;
  }

  clear(): void {
    if (this._disposed) return;
    this._db.execute('DELETE FROM sync_queue');
  }

  dispose(): void {
    this._disposed = true;
  }
}
