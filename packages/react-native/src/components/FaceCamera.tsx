/**
 * FaceCamera — camera component with real-time face detection.
 *
 * Integrates react-native-vision-camera with ML Kit face detection
 * via frame processor plugin.
 *
 * @example
 * <FaceCamera
 *   onFaceDetected={handleFace}
 *   isActive={true}
 * />
 */

import React, { useCallback, useMemo, useRef } from 'react';
import { StyleSheet, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { detectFaces } from 'react-native-vision-camera-face-detector';

export interface FaceCameraProps {
  readonly onFaceDetected: (faces: unknown[]) => void;
  readonly isActive: boolean;
  readonly style?: object;
}

export function FaceCamera({
  onFaceDetected,
  isActive,
  style,
}: FaceCameraProps): React.JSX.Element {
  const device = useCameraDevice('front');
  const frameCountRef = useRef(0);

  const processFrame = useCallback(
    (frame: unknown) => {
      'worklet';
      frameCountRef.current++;
      if (frameCountRef.current % 3 !== 0) return;

      try {
        const faces = detectFaces(frame as never);
        onFaceDetected(faces as unknown[]);
      } catch {
        /* frame processor errors are non-fatal */
      }
    },
    [onFaceDetected],
  );

  const frameProcessor = useFrameProcessor(processFrame, [onFaceDetected]);

  if (!device) {
    return (
      <View style={[styles.container, style]}>
        <View style={styles.placeholder}>
          {/* Camera not available */}
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    overflow: 'hidden',
  },
  placeholder: {
    flex: 1,
    backgroundColor: '#000',
  },
});
