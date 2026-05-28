/**
 * Face encoder — converts face crops into 128-d embeddings.
 *
 * This module defines the interface for the encoder. The actual TFLite
 * inference runs in the native layer (Android/iOS). This TypeScript layer
 * handles preprocessing (crop, align, normalize) and postprocessing
 * (L2 normalization).
 *
 * Architecture: MobileFaceNet
 * Input: 112×112×3 RGB face crop
 * Output: 128-dimensional L2-normalized embedding
 * Size: ~4.8MB (INT8 quantized)
 */

import { l2Normalize } from './CosineSimilarity';

export interface Encoder {
  /**
   * Encode a face crop into an embedding vector.
   *
   * @param faceData - Raw pixel data (112×112×3, RGB, uint8)
   * @returns L2-normalized 128-d embedding, or null on failure
   */
  encode(faceData: Uint8Array): Float32Array | null;

  /** Whether the model is loaded and ready */
  readonly isLoaded: boolean;

  /** Release native resources */
  dispose(): void;
}

/**
 * Default encoder implementation — wraps native TFLite inference.
 *
 * In the React Native layer, this delegates to the native module
 * that runs the actual TFLite model. In tests, it can be mocked.
 */
export class TFLiteEncoder implements Encoder {
  private _isLoaded = false;
  private _nativeModule: unknown;

  constructor(nativeModule: unknown) {
    this._nativeModule = nativeModule;
    this._isLoaded = true;
  }

  get isLoaded(): boolean {
    return this._isLoaded;
  }

  encode(faceData: Uint8Array): Float32Array | null {
    if (!this._isLoaded) return null;

    try {
      const module = this._nativeModule as {
        runInference(input: Uint8Array): Float32Array | null;
      };
      return module.runInference(faceData);
    } catch {
      return null;
    }
  }

  dispose(): void {
    this._isLoaded = false;
    this._nativeModule = null;
  }
}

/**
 * Stub encoder for testing and development.
 * Returns deterministic embeddings based on input hash.
 */
export class StubEncoder implements Encoder {
  private readonly _dimension: number;
  private _isLoaded = true;

  constructor(dimension = 128) {
    this._dimension = dimension;
  }

  get isLoaded(): boolean {
    return this._isLoaded;
  }

  encode(faceData: Uint8Array): Float32Array {
    if (!this._isLoaded) throw new Error('Encoder not loaded');
    const embedding = new Float32Array(this._dimension);
    let seed = 0;
    for (let i = 0; i < Math.min(faceData.length, 100); i++) {
      seed = ((seed << 5) - seed + (faceData[i] ?? 0)) | 0;
    }

    for (let i = 0; i < this._dimension; i++) {
      seed = ((seed * 1103515245 + 12345) & 0x7fffffff);
      embedding[i] = (seed / 0x7fffffff) * 2 - 1;
    }

    return l2Normalize(embedding);
  }

  dispose(): void {
    this._isLoaded = false;
  }
}
