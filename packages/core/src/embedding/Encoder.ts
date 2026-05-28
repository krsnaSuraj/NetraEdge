/**
 * Face encoder — converts face crops into 128-d embeddings.
 *
 * This module defines the interface for the encoder. The actual TFLite
 * inference runs in the native layer (Android/iOS). This TypeScript layer
 * handles postprocessing (L2 normalization).
 *
 * Architecture: MobileFaceNet
 * Input: 112×112×3 RGB face crop (float array, normalized 0–1)
 * Output: 128-dimensional L2-normalized embedding
 * Size: ~4.8MB (INT8 quantized)
 */

import { l2Normalize } from './CosineSimilarity';

export interface Encoder {
  /**
   * Encode a face crop into an embedding vector.
   *
   * @param faceData - Normalized pixel data (112×112×3 floats, 0–1)
   * @returns L2-normalized 128-d embedding, or null on failure
   */
  encode(faceData: Float32Array): Promise<Float32Array | null>;

  /** Whether the model is loaded and ready */
  readonly isLoaded: boolean;

  /** Release native resources */
  dispose(): void;
}

/**
 * Native TFLite encoder — wraps the React Native bridge.
 *
 * Delegates inference to the platform-specific TFLite interpreter
 * (Kotlin on Android, Swift on iOS). The native module handles
 * tensor allocation, quantization, and inference.
 */
export class TFLiteEncoder implements Encoder {
  private _isLoaded = false;
  private readonly _nativeModule: {
    runRecognition(pixels: number[]): Promise<number[] | null>;
    isInitialized(): Promise<boolean>;
  };

  constructor(nativeModule: {
    runRecognition(pixels: number[]): Promise<number[] | null>;
    isInitialized(): Promise<boolean>;
  }) {
    this._nativeModule = nativeModule;
    this._isLoaded = true;
  }

  get isLoaded(): boolean {
    return this._isLoaded;
  }

  async encode(faceData: Float32Array): Promise<Float32Array | null> {
    if (!this._isLoaded) return null;

    try {
      const result = await this._nativeModule.runRecognition(
        Array.from(faceData),
      );
      if (!result || result.length === 0) return null;
      return l2Normalize(new Float32Array(result));
    } catch {
      return null;
    }
  }

  dispose(): void {
    this._isLoaded = false;
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

  async encode(faceData: Float32Array): Promise<Float32Array> {
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
