/**
 * AWS REST transport — uploads enrollment data to a cloud endpoint.
 *
 * Uses standard fetch() for HTTP requests. Compatible with any
 * REST API backend (API Gateway + Lambda, EC2, etc.).
 *
 * SECURITY: applies Differential Privacy (Laplace noise) to embeddings
 * before upload, so the cloud copy cannot be reverse-engineered to
 * reconstruct the original face. See DPNoise.ts.
 */

import { DPNoise, type DPConfig } from '@netraedge/core';
import type { SyncTransport } from '@netraedge/core';

export interface AWSTransportConfig {
  readonly endpoint: string;
  readonly apiKey: string;
  readonly timeoutMs: number;
  /** Differential privacy config. Set dpEnabled=false to disable (NOT recommended). */
  readonly dpConfig?: Partial<DPConfig>;
  readonly dpEnabled?: boolean;
}

const DEFAULT_CONFIG: AWSTransportConfig = {
  endpoint: '',
  apiKey: '',
  timeoutMs: 10000,
  dpEnabled: true,
};

export class AWSSyncTransport implements SyncTransport {
  private readonly _config: AWSTransportConfig;
  private readonly _dp: DPNoise | null;

  constructor(config: Partial<AWSTransportConfig> = {}) {
    this._config = { ...DEFAULT_CONFIG, ...config };
    this._dp = this._config.dpEnabled
      ? new DPNoise(this._config.dpConfig)
      : null;
  }

  private protect(embedding: Float32Array): Float32Array {
    return this._dp ? this._dp.applyNoise(embedding) : embedding;
  }

  async uploadEnrollment(
    userId: string,
    embedding: Float32Array,
    metadata: Record<string, unknown>,
  ): Promise<boolean> {
    if (!this._config.endpoint) return false;

    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      this._config.timeoutMs,
    );

    try {
      const noisy = this.protect(embedding);
      const response = await fetch(`${this._config.endpoint}/enrollments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this._config.apiKey
            ? { Authorization: `Bearer ${this._config.apiKey}` }
            : {}),
        },
        body: JSON.stringify({
          userId,
          embedding: Array.from(noisy),
          metadata: {
            ...metadata,
            dpApplied: !!this._dp,
            dpEpsilon: this._dp ? this._dp.getScale() : 0,
          },
          syncedAt: Date.now(),
        }),
        signal: controller.signal,
      });

      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timeout);
    }
  }

  async uploadBatch(
    enrollments: readonly {
      userId: string;
      embedding: Float32Array;
      metadata: Record<string, unknown>;
    }[],
  ): Promise<boolean> {
    if (!this._config.endpoint || enrollments.length === 0) return false;

    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      this._config.timeoutMs,
    );

    try {
      const payload = enrollments.map((e) => {
        const noisy = this.protect(e.embedding);
        return {
          userId: e.userId,
          embedding: Array.from(noisy),
          metadata: {
            ...e.metadata,
            dpApplied: !!this._dp,
          },
          syncedAt: Date.now(),
        };
      });

      const response = await fetch(`${this._config.endpoint}/enrollments/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this._config.apiKey
            ? { Authorization: `Bearer ${this._config.apiKey}` }
            : {}),
        },
        body: JSON.stringify({ enrollments: payload }),
        signal: controller.signal,
      });

      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timeout);
    }
  }
}
