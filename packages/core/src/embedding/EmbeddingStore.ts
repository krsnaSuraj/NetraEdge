/**
 * Embedding storage — manages enrolled face embeddings.
 *
 * Uses SQLite under the hood. Each enrollment stores:
 * - userId (unique identifier)
 * - embedding (128-d float vector)
 * - enrolledAt (timestamp)
 * - frameCount (number of frames used)
 *
 * Storage is local-only. Sync to AWS happens via SyncManager.
 */

export interface Enrollment {
  readonly userId: string;
  readonly embedding: Float32Array;
  readonly enrolledAt: number;
  readonly frameCount: number;
}

export interface EmbeddingStore {
  /** Enroll a new user with an averaged embedding */
  enroll(userId: string, embeddings: Float32Array[]): Promise<boolean>;

  /** Find the best matching user for a probe embedding */
  identify(
    probe: Float32Array,
    threshold: number,
  ): Promise<IdentificationResult | null>;

  /** Get all enrolled user IDs */
  listUsers(): Promise<readonly string[]>;

  /** Remove a user's enrollment */
  remove(userId: string): Promise<boolean>;

  /** Get enrollment count */
  count(): Promise<number>;

  /** Clear all enrollments (used after sync purge) */
  clear(): Promise<void>;

  /** Release resources */
  dispose(): void;
}

export interface IdentificationResult {
  readonly userId: string;
  readonly confidence: number;
}

/**
 * In-memory implementation for testing.
 */
export class InMemoryEmbeddingStore implements EmbeddingStore {
  private readonly _enrollments = new Map<string, Enrollment>();

  async enroll(userId: string, embeddings: Float32Array[]): Promise<boolean> {
    if (embeddings.length === 0) return false;

    const averaged = this.averageEmbeddings(embeddings);

    this._enrollments.set(userId, {
      userId,
      embedding: averaged,
      enrolledAt: Date.now(),
      frameCount: embeddings.length,
    });

    return true;
  }

  async identify(
    probe: Float32Array,
    threshold: number,
  ): Promise<IdentificationResult | null> {
    let bestMatch: IdentificationResult | null = null;

    for (const [userId, enrollment] of this._enrollments) {
      const similarity = this.dotProduct(probe, enrollment.embedding);
      if (similarity >= threshold) {
        if (!bestMatch || similarity > bestMatch.confidence) {
          bestMatch = { userId, confidence: similarity };
        }
      }
    }

    return bestMatch;
  }

  async listUsers(): Promise<readonly string[]> {
    return Array.from(this._enrollments.keys());
  }

  async remove(userId: string): Promise<boolean> {
    return this._enrollments.delete(userId);
  }

  async count(): Promise<number> {
    return this._enrollments.size;
  }

  async clear(): Promise<void> {
    this._enrollments.clear();
  }

  dispose(): void {
    this._enrollments.clear();
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

  private dotProduct(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] ?? 0) * (b[i] ?? 0);
    }
    return sum;
  }
}
