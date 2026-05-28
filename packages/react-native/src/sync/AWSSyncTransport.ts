/**
 * AWS REST transport — uploads enrollment data to a cloud endpoint.
 *
 * Uses standard fetch() for HTTP requests. Compatible with any
 * REST API backend (API Gateway + Lambda, EC2, etc.).
 *
 * For the hackathon demo, the endpoint can be a mock server
 * or a simple Lambda function that writes to DynamoDB/S3.
 */

import type { SyncTransport } from '@netraedge/core';

export interface AWSTransportConfig {
  readonly endpoint: string;
  readonly apiKey: string;
  readonly timeoutMs: number;
}

const DEFAULT_CONFIG: AWSTransportConfig = {
  endpoint: '',
  apiKey: '',
  timeoutMs: 10000,
};

export class AWSSyncTransport implements SyncTransport {
  private readonly _config: AWSTransportConfig;

  constructor(config: Partial<AWSTransportConfig> = {}) {
    this._config = { ...DEFAULT_CONFIG, ...config };
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
          embedding: Array.from(embedding),
          metadata,
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
      const payload = enrollments.map((e) => ({
        userId: e.userId,
        embedding: Array.from(e.embedding),
        metadata: e.metadata,
        syncedAt: Date.now(),
      }));

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
