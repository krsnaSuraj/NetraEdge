/**
 * DPNoise — Differential Privacy noise injection for face embeddings.
 *
 * Before uploading embeddings to the cloud, add calibrated Laplace noise
 * so that no single record can be reverse-engineered from the cloud copy.
 *
 * Mathematical basis:
 *   For epsilon-differential privacy, noise scale = sensitivity / epsilon
 *   Laplace(0, sensitivity/epsilon) is added element-wise to the embedding.
 *
 * Reference:
 *   Dwork, C. & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy."
 *   Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407.
 *
 * For face embeddings (MobileFaceNet, 128-dim float32):
 *   - Sensitivity: max L1 distance between two adjacent embeddings = 2.0
 *     (after L2 normalization, max L1 = 2)
 *   - Epsilon: 1.0 (industry standard for face data)
 *   - Noise scale b = sensitivity/epsilon = 2.0 / 1.0 = 2.0
 *   - Result: each element gets Laplace(0, 2.0) noise before re-normalization
 *
 * Trade-off: higher epsilon = less noise = better utility but weaker privacy.
 *   epsilon=1.0 is the conservative NHAI-grade setting.
 *
 * After noise, the embedding is re-normalized to unit L2 so cosine similarity
 * still works downstream.
 */
export interface DPConfig {
  /** Privacy budget — smaller = stronger privacy, more noise. Default 1.0. */
  readonly epsilon: number;
  /** L1 sensitivity. Default 2.0 (max L1 between unit vectors). */
  readonly sensitivity: number;
  /** Random seed for reproducible tests. Production uses Math.random. */
  readonly seed?: number;
}

export const DEFAULT_DP_CONFIG: DPConfig = {
  epsilon: 1.0,
  sensitivity: 2.0,
};

/**
 * Sample one value from Laplace distribution with location=0, scale=b.
 * Uses inverse-CDF method: x = b * sign(u) * ln(1 - 2|u|), where u ~ U(-0.5, 0.5).
 */
function sampleLaplace(scale: number, rand: () => number): number {
  const u = rand() - 0.5; // U(-0.5, 0.5)
  if (u === 0) return 0;
  return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}

/**
 * Mulberry32 — fast, deterministic PRNG. Used when config.seed is set
 * (tests, demo). Production omits seed and uses Math.random.
 */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class DPNoise {
  private readonly scale: number;
  private readonly rand: () => number;

  constructor(config: Partial<DPConfig> = {}) {
    const cfg = { ...DEFAULT_DP_CONFIG, ...config };
    if (cfg.epsilon <= 0) {
      throw new Error('epsilon must be > 0');
    }
    if (cfg.sensitivity <= 0) {
      throw new Error('sensitivity must be > 0');
    }
    this.scale = cfg.sensitivity / cfg.epsilon;
    this.rand = cfg.seed !== undefined ? mulberry32(cfg.seed) : Math.random;
  }

  /**
   * Add Laplace noise to a 128-dim face embedding, then re-normalize to unit L2.
   *
   * @param embedding 128-dim float array (L2-normalized)
   * @returns 128-dim float array with noise, re-normalized
   */
  applyNoise(embedding: Float32Array | number[]): Float32Array {
    const out = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      const e = embedding[i] ?? 0;
      out[i] = e + sampleLaplace(this.scale, this.rand);
    }
    // Re-normalize to unit L2 so cosine similarity still works
    let norm = 0;
    for (let i = 0; i < out.length; i++) {
      const v = out[i] ?? 0;
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1; // avoid divide by zero
    for (let i = 0; i < out.length; i++) {
      out[i] = (out[i] ?? 0) / norm;
    }
    return out;
  }

  /**
   * Apply noise to a batch of embeddings.
   * Used by SyncManager before uploading to AWS.
   */
  applyNoiseBatch(embeddings: Array<Float32Array | number[]>): Float32Array[] {
    return embeddings.map(e => this.applyNoise(e));
  }

  /** For diagnostic logging only. */
  getScale(): number {
    return this.scale;
  }
}
