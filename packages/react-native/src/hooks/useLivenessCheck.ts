/**
 * useLivenessCheck — real-time liveness detection hook.
 *
 * Delegates to the core LivenessOrchestrator for blink detection,
 * texture analysis, and depth estimation. Surfaces the combined
 * liveness verdict to the React component layer.
 */

import { useCallback, useRef, useState } from 'react';
import type { LivenessOrchestrator } from '@netraedge/core';
import type { Point3D } from '@netraedge/core';

export interface LivenessState {
  readonly blinkDetected: boolean;
  readonly blinkCount: number;
  readonly textureScore: number;
  readonly depthScore: number;
  readonly isLive: boolean;
  readonly prompt: string | null;
}

export function useLivenessCheck(orchestrator: LivenessOrchestrator | null): {
  state: LivenessState;
  processFrame: (meshPoints: Point3D[], faceData?: Float32Array) => Promise<void>;
  reset: () => void;
} {
  const [state, setState] = useState<LivenessState>({
    blinkDetected: false,
    blinkCount: 0,
    textureScore: 0,
    depthScore: 0,
    isLive: false,
    prompt: 'Blink your eyes',
  });

  const processingRef = useRef(false);

  const processFrame = useCallback(
    async (meshPoints: Point3D[], faceData?: Float32Array) => {
      if (!orchestrator || processingRef.current) return;
      processingRef.current = true;

      try {
        const result = await orchestrator.processFrame(
          faceData ?? new Float32Array(0),
          meshPoints,
          Date.now(),
        );

        const blinkDetected = result.blink.detected;
        const blinkCount = orchestrator.blinkCount;

        let prompt = 'Blink your eyes';
        if (result.verdict === 'live') {
          prompt = 'Liveness verified';
        } else if (blinkDetected) {
          prompt = 'Hold still...';
        }

        setState({
          blinkDetected,
          blinkCount,
          textureScore: result.texture.realScore,
          depthScore: result.depth.variance,
          isLive: result.verdict === 'live',
          prompt,
        });
      } finally {
        processingRef.current = false;
      }
    },
    [orchestrator],
  );

  const reset = useCallback(() => {
    orchestrator?.reset();
    setState({
      blinkDetected: false,
      blinkCount: 0,
      textureScore: 0,
      depthScore: 0,
      isLive: false,
      prompt: 'Blink your eyes',
    });
  }, [orchestrator]);

  return { state, processFrame, reset };
}
