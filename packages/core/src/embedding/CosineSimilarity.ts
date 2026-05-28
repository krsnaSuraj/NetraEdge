/**
 * Cosine similarity between two embedding vectors.
 *
 * Both vectors must be L2-normalized (unit length). If they are,
 * cosine similarity equals dot product — which is O(n) instead of O(2n).
 *
 * @param a - First embedding vector
 * @param b - Second embedding vector (must be same length as a)
 * @returns Similarity score in range [-1.0, 1.0]
 *
 * @example
 * const sim = cosineSimilarity(embeddingA, embeddingB);
 * if (sim > 0.75) { /* match *\/ }
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(
      `Embedding dimension mismatch: ${a.length} vs ${b.length}`,
    );
  }

  let dotProduct = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += (a[i] ?? 0) * (b[i] ?? 0);
  }

  return Math.max(-1, Math.min(1, dotProduct));
}

/**
 * L2-normalize an embedding vector in place.
 * After normalization, the vector has unit length (magnitude = 1.0).
 */
export function l2Normalize(embedding: Float32Array): Float32Array {
  let magnitude = 0;
  for (let i = 0; i < embedding.length; i++) {
    const val = embedding[i] ?? 0;
    magnitude += val * val;
  }
  magnitude = Math.sqrt(magnitude);

  if (magnitude === 0) return embedding;

  const normalized = new Float32Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) {
    normalized[i] = (embedding[i] ?? 0) / magnitude;
  }
  return normalized;
}
