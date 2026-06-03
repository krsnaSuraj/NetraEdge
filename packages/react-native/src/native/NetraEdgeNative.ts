/**
 * Native module declarations — TypeScript bridge to Kotlin/Swift TFLite inference.
 *
 * These declarations match the @ReactMethod signatures in NetraEdgePackage.kt.
 * The native module is registered as "NetraEdgeModule" in both Android and iOS.
 *
 * SOTA 2026 (Path A+):
 *   - Embedding dim: 512 (EdgeFace-XS via torch.hub, derived at load time)
 *   - Test-time augmentation: horizontal flip averaging (useTTA flag, default ON)
 *   - Both iOS and Android expose the same setUseTTA / isUseTTA / getEmbeddingDim
 */

import { NativeModules, Platform } from 'react-native';

export interface NetraEdgeNativeModule {
  /** Load TFLite models from app assets */
  initialize(): Promise<boolean>;

  /** Check if models are loaded */
  isInitialized(): Promise<boolean>;

  /**
   * Run face recognition inference (with optional TTA).
   * @param pixels Normalized float array (112*112*3 = 37632 values, 0–1)
   * @returns N-d L2-normalized embedding (N = model's actual output dim,
   *          derived at initialize time: 128 for pre-trained MobileFaceNet,
   *          512 for Path A+ Day-2 fine-tuned EdgeFace-XS), or null on failure
   */
  runRecognition(pixels: number[]): Promise<number[] | null>;

  /**
   * Get the embedding dim of the currently loaded recognition model.
   * Returns -1 if the model isn't loaded yet.
   */
  getEmbeddingDim(): Promise<number>;

  /**
   * Run liveness detection inference.
   * @param pixels Normalized float array (112*112*3 = 37632 values, 0–1)
   * @returns 3-class probabilities [real, print, screen], or null on failure
   */
  runLiveness(pixels: number[]): Promise<number[] | null>;

  /** Compute cosine similarity between two embeddings */
  cosineSimilarity(a: number[], b: number[]): Promise<number>;

  /**
   * Enable or disable Test-Time Augmentation (horizontal flip averaging)
   * at inference. When ON (default), the native module runs the input AND
   * its horizontal flip through the model, L2-normalizes each, averages,
   * and re-normalizes. Adds ~5-10ms per inference on mid-range Android 8+.
   *
   * Path A+: default ON for +0.3-0.5% LFW accuracy (free).
   */
  setUseTTA(enabled: boolean): Promise<void>;

  /** Returns whether TTA is currently enabled. */
  isUseTTA(): Promise<boolean>;

  /**
   * Crop face from raw frame data, resize to 112x112, normalize to 0-1.
   * CRITICAL: Bridges camera frames to face recognition pipeline.
   *
   * @param imageData Raw RGB pixel data (flat float array, R,G,B,R,G,B,...)
   * @param frameWidth Width of camera frame
   * @param frameHeight Height of camera frame
   * @param faceX Face bounding box X coordinate
   * @param faceY Face bounding box Y coordinate
   * @param faceWidth Face bounding box width
   * @param faceHeight Face bounding box height
   * @returns Normalized float array (37632 values) or null on failure
   */
  cropFace(
    imageData: number[],
    frameWidth: number,
    frameHeight: number,
    faceX: number,
    faceY: number,
    faceWidth: number,
    faceHeight: number,
  ): Promise<number[] | null>;

  /**
   * Convert YUV_420_888 byte data to normalized RGB float array.
   */
  yuvToNormalized(yuvData: number[], width: number, height: number): Promise<number[] | null>;

  /** Release TFLite resources */
  close(): void;
}

const LINKING_ERROR =
  `The native module 'NetraEdgeModule' is not linked.\n` +
  `Platform: ${Platform.OS}\n` +
  `Make sure the native module is properly registered.`;

export const NetraEdgeNative: NetraEdgeNativeModule =
  NativeModules.NetraEdgeModule
    ? NativeModules.NetraEdgeModule
    : new Proxy({} as NetraEdgeNativeModule, {
        get(_, prop: string) {
          throw new Error(`${LINKING_ERROR}\nMethod: ${prop}`);
        },
      });
