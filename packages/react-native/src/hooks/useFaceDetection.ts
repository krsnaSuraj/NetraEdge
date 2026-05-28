/**
 * useFaceDetection — real-time face detection hook.
 *
 * Uses react-native-vision-camera with ML Kit face detection
 * via frame processor plugin.
 *
 * @example
 * const { faces, isDetecting } = useFaceDetection(cameraRef);
 */

import { useCallback, useRef, useState } from 'react';
import type { Face } from 'react-native-vision-camera-face-detector';

export interface FaceDetectionState {
  readonly faces: readonly Face[];
  readonly isDetecting: boolean;
  readonly lastDetectionTime: number;
}

export function useFaceDetection(): {
  state: FaceDetectionState;
  onFacesDetected: (faces: Face[]) => void;
  reset: () => void;
} {
  const [state, setState] = useState<FaceDetectionState>({
    faces: [],
    isDetecting: false,
    lastDetectionTime: 0,
  });

  const frameCount = useRef(0);

  const onFacesDetected = useCallback((faces: Face[]) => {
    frameCount.current++;

    if (frameCount.current % 3 !== 0) return;

    setState({
      faces,
      isDetecting: faces.length > 0,
      lastDetectionTime: Date.now(),
    });
  }, []);

  const reset = useCallback(() => {
    frameCount.current = 0;
    setState({ faces: [], isDetecting: false, lastDetectionTime: 0 });
  }, []);

  return { state, onFacesDetected, reset };
}
