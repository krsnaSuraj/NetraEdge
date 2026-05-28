/**
 * SQLite-backed embedding store — persists enrolled face embeddings.
 *
 * Unlike InMemoryEmbeddingStore, this survives app restarts.
 * Uses react-native-quick-sqlite for synchronous, fast SQLite access.
 *
 * Schema:
 *   embeddings(
 *     user_id TEXT PRIMARY KEY,
 *     embedding BLOB NOT NULL,
 *     enrolled_at INTEGER NOT NULL,
 *     frame_count INTEGER NOT NULL
 *   )
 */

import type {
  EmbeddingStore,
  IdentificationResult,
} from '@netraedge/core';

export interface SQLiteDatabase {
  execute(sql: string, params?: unknown[]): { rows: unknown[] };
}

export class SQLiteEmbeddingStore implements EmbeddingStore {
  private readonly _db: SQLiteDatabase;
  private _disposed = false;

  constructor(db: SQLiteDatabase) {
    this._db = db;
    this.initialize();
  }

  private initialize(): void {
    this._db.execute(`
      CREATE TABLE IF NOT EXISTS embeddings (
        user_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL,
        enrolled_at INTEGER NOT NULL,
        frame_count INTEGER NOT NULL
      )
    `);
    this._db.execute(
      'CREATE INDEX IF NOT EXISTS idx_embeddings_enrolled ON embeddings(enrolled_at)',
    );
  }

  async enroll(userId: string, embeddings: Float32Array[]): Promise<boolean> {
    if (this._disposed || embeddings.length === 0) return false;

    const averaged = this.averageEmbeddings(embeddings);
    const blob = this.embeddingToBlob(averaged);

    this._db.execute(
      'INSERT OR REPLACE INTO embeddings (user_id, embedding, enrolled_at, frame_count) VALUES (?, ?, ?, ?)',
      [userId, blob, Date.now(), embeddings.length],
    );

    return true;
  }

  async identify(
    probe: Float32Array,
    threshold: number,
  ): Promise<IdentificationResult | null> {
    if (this._disposed) return null;

    const { rows } = this._db.execute(
      'SELECT user_id, embedding FROM embeddings',
    );

    let bestMatch: IdentificationResult | null = null;

    for (const row of rows as Array<{ user_id: string; embedding: ArrayBuffer }>) {
      const stored = this.blobToEmbedding(row.embedding);
      const similarity = this.dotProduct(probe, stored);

      if (similarity >= threshold) {
        if (!bestMatch || similarity > bestMatch.confidence) {
          bestMatch = { userId: row.user_id, confidence: similarity };
        }
      }
    }

    return bestMatch;
  }

  async listUsers(): Promise<readonly string[]> {
    if (this._disposed) return [];

    const { rows } = this._db.execute(
      'SELECT user_id FROM embeddings ORDER BY enrolled_at DESC',
    );

    return (rows as Array<{ user_id: string }>).map((r) => r.user_id);
  }

  async remove(userId: string): Promise<boolean> {
    if (this._disposed) return false;

    const { rows } = this._db.execute(
      'DELETE FROM embeddings WHERE user_id = ?',
      [userId],
    );

    return rows.length > 0;
  }

  async count(): Promise<number> {
    if (this._disposed) return 0;

    const { rows } = this._db.execute(
      'SELECT COUNT(*) as cnt FROM embeddings',
    );

    return (rows as Array<{ cnt: number }>)[0]?.cnt ?? 0;
  }

  async clear(): Promise<void> {
    if (this._disposed) return;
    this._db.execute('DELETE FROM embeddings');
  }

  dispose(): void {
    this._disposed = true;
  }

  private averageEmbeddings(embeddings: readonly Float32Array[]): Float32Array {
    const dim = embeddings[0]?.length ?? 0;
    const avg = new Float32Array(dim);

    for (const emb of embeddings) {
      for (let i = 0; i < dim; i++) {
        avg[i] = (avg[i] ?? 0) + (emb[i] ?? 0);
      }
    }

    for (let i = 0; i < dim; i++) {
      avg[i] = (avg[i] ?? 0) / embeddings.length;
    }

    let magnitude = 0;
    for (let i = 0; i < dim; i++) {
      const val = avg[i] ?? 0;
      magnitude += val * val;
    }
    magnitude = Math.sqrt(magnitude);

    if (magnitude > 0) {
      for (let i = 0; i < dim; i++) {
        avg[i] = (avg[i] ?? 0) / magnitude;
      }
    }

    return avg;
  }

  private embeddingToBlob(embedding: Float32Array): ArrayBuffer {
    return embedding.buffer.slice(
      embedding.byteOffset,
      embedding.byteOffset + embedding.byteLength,
    );
  }

  private blobToEmbedding(blob: ArrayBuffer): Float32Array {
    return new Float32Array(blob);
  }

  private dotProduct(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] ?? 0) * (b[i] ?? 0);
    }
    return sum;
  }
}
