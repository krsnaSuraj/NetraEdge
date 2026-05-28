/**
 * Texture-based liveness detection.
 *
 * Distinguishes real faces from printed photos and screen replays
 * by analyzing micro-texture patterns. Real skin has characteristic
 * patterns (pores, fine wrinkles, specular highlights) that printed
 * media and screens lack.
 *
 * This module defines the interface. The actual CNN inference runs
 * in the native layer via TFLite.
 */

import type { TextureResult } from '../types/Liveness';

export interface TextureAnalyzer {
  /** Analyze a face crop for texture-based liveness */
  analyze(faceData: Float32Array): Promise<TextureResult>;

  readonly isLoaded: boolean;
  dispose(): void;
}

/**
 * TFLite-backed texture analyzer.
 * Input: 112×112×3 RGB face crop (float array, normalized 0–1)
 * Output: 3-class softmax [real, print, screen]
 */
export class CNNTextureAnalyzer implements TextureAnalyzer {
  private _isLoaded = false;
  private readonly _nativeModule: {
    runLiveness(pixels: number[]): Promise<number[] | null>;
  };

  constructor(nativeModule: {
    runLiveness(pixels: number[]): Promise<number[] | null>;
  }) {
    this._nativeModule = nativeModule;
    this._isLoaded = true;
  }

  get isLoaded(): boolean {
    return this._isLoaded;
  }

  async analyze(faceData: Float32Array): Promise<TextureResult> {
    if (!this._isLoaded) {
      return { realScore: 0, printScore: 0, screenScore: 0 };
    }

    try {
      const output = await this._nativeModule.runLiveness(
        Array.from(faceData),
      );

      if (!output || output.length < 3) {
        return { realScore: 0, printScore: 0, screenScore: 0 };
      }

      return {
        realScore: output[0] ?? 0,
        printScore: output[1] ?? 0,
        screenScore: output[2] ?? 0,
      };
    } catch {
      return { realScore: 0, printScore: 0, screenScore: 0 };
    }
  }

  dispose(): void {
    this._isLoaded = false;
  }
}

/**
 * Stub analyzer for testing — always returns "real" with high confidence.
 */
export class StubTextureAnalyzer implements TextureAnalyzer {
  get isLoaded(): boolean {
    return true;
  }

  async analyze(_faceData: Float32Array): Promise<TextureResult> {
    return {
      realScore: 0.95,
      printScore: 0.03,
      screenScore: 0.02,
    };
  }

  dispose(): void {
    /* no-op */
  }
}
