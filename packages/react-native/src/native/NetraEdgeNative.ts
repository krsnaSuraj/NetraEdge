/**
 * Native module declarations — TypeScript bridge to Kotlin/Swift TFLite inference.
 *
 * These declarations match the @ReactMethod signatures in NetraEdgePackage.kt.
 * The native module is registered as "NetraEdgeModule" in both Android and iOS.
 */

import { NativeModules, Platform } from 'react-native';

export interface NetraEdgeNativeModule {
  /** Load TFLite models from app assets */
  initialize(): Promise<boolean>;

  /** Check if models are loaded */
  isInitialized(): Promise<boolean>;

  /**
   * Run face recognition inference.
   * @param pixels Normalized float array (112*112*3 = 37632 values, 0–1)
   * @returns 128-d L2-normalized embedding, or null on failure
   */
  runRecognition(pixels: number[]): Promise<number[] | null>;

  /**
   * Run liveness detection inference.
   * @param pixels Normalized float array (112*112*3 = 37632 values, 0–1)
   * @returns 3-class probabilities [real, print, screen], or null on failure
   */
  runLiveness(pixels: number[]): Promise<number[] | null>;

  /** Compute cosine similarity between two embeddings */
  cosineSimilarity(a: number[], b: number[]): Promise<number>;

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
