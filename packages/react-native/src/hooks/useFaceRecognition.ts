/**
 * useFaceRecognition — face recognition and verification hook.
 *
 * Wraps the core FacePipeline for React Native usage.
 * Manages enrollment and verification flows.
 */

import { useCallback, useRef, useState } from 'react';
import { FacePipeline } from '@netraedge/core';
import type { Point3D } from '@netraedge/core';

export interface RecognitionState {
  readonly isProcessing: boolean;
  readonly lastResult: VerificationResult | null;
  readonly enrolledUsers: readonly string[];
  readonly error: string | null;
}

export interface VerificationResult {
  readonly matched: boolean;
  readonly userId: string | null;
  readonly confidence: number;
  readonly livenessPassed: boolean;
}

export function useFaceRecognition(pipeline: FacePipeline): {
  state: RecognitionState;
  verify: (faceData: Float32Array, meshPoints: Point3D[]) => Promise<VerificationResult | null>;
  enroll: (userId: string, frames: Float32Array[], meshPoints: Point3D[][]) => Promise<boolean>;
  refreshUsers: () => Promise<void>;
  reset: () => void;
} {
  const [state, setState] = useState<RecognitionState>({
    isProcessing: false,
    lastResult: null,
    enrolledUsers: [],
    error: null,
  });

  const processingRef = useRef(false);

  const verify = useCallback(
    async (faceData: Float32Array, meshPoints: Point3D[]): Promise<VerificationResult | null> => {
      if (processingRef.current) return null;
      processingRef.current = true;
      setState((prev) => ({ ...prev, isProcessing: true, error: null }));

      try {
        const result = await pipeline.verify(faceData, meshPoints, Date.now());
        if (result.ok) {
          const vr: VerificationResult = {
            matched: result.value.matched,
            userId: result.value.userId,
            confidence: result.value.confidence,
            livenessPassed: result.value.liveness.verdict === 'live',
          };
          setState((prev) => ({ ...prev, isProcessing: false, lastResult: vr }));
          return vr;
        }
        setState((prev) => ({ ...prev, isProcessing: false, error: result.error.message }));
        return null;
      } catch (e) {
        setState((prev) => ({ ...prev, isProcessing: false, error: String(e) }));
        return null;
      } finally {
        processingRef.current = false;
      }
    },
    [pipeline],
  );

  const enroll = useCallback(
    async (userId: string, frames: Float32Array[], meshPoints: Point3D[][]): Promise<boolean> => {
      setState((prev) => ({ ...prev, isProcessing: true, error: null }));
      try {
        const result = await pipeline.enroll(userId, frames, meshPoints, Date.now());
        if (result.ok) {
          await refreshUsers();
          return true;
        }
        setState((prev) => ({ ...prev, error: result.error.message }));
        return false;
      } catch (e) {
        setState((prev) => ({ ...prev, error: String(e) }));
        return false;
      } finally {
        setState((prev) => ({ ...prev, isProcessing: false }));
      }
    },
    [pipeline],
  );

  const refreshUsers = useCallback(async () => {
    const users = await pipeline.listUsers();
    setState((prev) => ({ ...prev, enrolledUsers: users }));
  }, [pipeline]);

  const reset = useCallback(() => {
    setState({ isProcessing: false, lastResult: null, enrolledUsers: [], error: null });
  }, []);

  return { state, verify, enroll, refreshUsers, reset };
}
