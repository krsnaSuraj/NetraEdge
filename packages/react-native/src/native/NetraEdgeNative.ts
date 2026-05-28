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
