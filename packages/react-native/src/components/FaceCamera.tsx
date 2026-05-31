/**
 * FaceCamera — full-screen camera with real-time face detection.
 *
 * Renders a full-bleed camera preview with ML Kit face detection.
 * Children are rendered as overlays on top of the camera.
 */

import React, { useCallback, useRef } from 'react';
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
  readonly children?: React.ReactNode;
}

export function FaceCamera({
  onFaceDetected,
  isActive,
  children,
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
      <View style={styles.container}>
        <View style={styles.noCamera}>
          <View style={styles.guideCircle} />
        </View>
        {children}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />
      <View style={styles.guideContainer}>
        <View style={styles.guideCircle} />
      </View>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  noCamera: {
    flex: 1,
    backgroundColor: '#111',
    justifyContent: 'center',
    alignItems: 'center',
  },
  guideContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    pointerEvents: 'none',
  },
  guideCircle: {
    width: 240,
    height: 300,
    borderRadius: 120,
    borderWidth: 3,
    borderColor: 'rgba(59, 130, 246, 0.5)',
    backgroundColor: 'transparent',
  },
});
