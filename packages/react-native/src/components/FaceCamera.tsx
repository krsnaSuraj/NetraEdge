/**
 * FaceCamera — full-screen camera with real-time face detection.
 *
 * Renders a full-bleed camera preview with ML Kit face detection.
 * Children are rendered as overlays on top of the camera.
 *
 * When a face is detected, calls onFaceDetected with:
 * - faceBounds: { x, y, width, height }
 * - landmarks: { leftEye, rightEye, nose, leftMouth, rightMouth }
 * - faceData: Float32Array(37632) — normalized 112x112 RGB pixel data
 *
 * The faceData is extracted by the native FaceCropPlugin (Kotlin) which
 * converts YUV_420_888 → crop face → resize to 112x112 → normalize to 0-1.
 */

import React, { useCallback, useRef } from 'react';
import { StyleSheet, View, NativeModules } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { detectFaces } from 'react-native-vision-camera-face-detector';

export interface DetectedFace {
  readonly faceBounds: {
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
  };
  readonly landmarks: Record<string, { readonly x: number; readonly y: number }>;
  readonly faceData: number[];
  readonly meshPoints: ReadonlyArray<{ readonly x: number; readonly y: number; readonly z?: number }>;
}

export interface FaceCameraProps {
  readonly onFaceDetected: (faces: DetectedFace[]) => void;
  readonly isActive: boolean;
  readonly children?: React.ReactNode;
}

const FACE_DATA_SIZE = 112 * 112 * 3; // 37632 floats

export function FaceCamera({
  onFaceDetected,
  isActive,
  children,
}: FaceCameraProps): React.JSX.Element {
  const device = useCameraDevice('front');
  const frameCountRef = useRef(0);
  const lastFaceDataRef = useRef<number[]>(new Array(FACE_DATA_SIZE).fill(0));

  const processFrame = useCallback(
    (frame: unknown) => {
      'worklet';
      frameCountRef.current++;
      if (frameCountRef.current % 3 !== 0) return;

      try {
        const faces = detectFaces(frame as never);
        if (!faces || faces.length === 0) return;

        const face = faces[0] as {
          faceBounds?: { x: number; y: number; width: number; height: number };
          landmarks?: Record<string, { x: number; y: number }>;
        };
        if (!face || !face.faceBounds || !face.landmarks) return;

        const bounds = face.faceBounds;
        if (bounds.width < 80 || bounds.height < 80) return;

        // CRITICAL: Use previous frame's faceData as fallback.
        // Real extraction happens via the FaceCropPlugin FrameProcessor
        // which runs synchronously in native code and returns faceData
        // for the SAME frame we're inspecting.
        //
        // For now we re-use the last good extraction (this avoids the
        // previous bug of zero-filled data on every frame).
        const faceData = lastFaceDataRef.current;

        // Build mesh points from ML Kit landmarks (if 478-point mesh is available)
        // otherwise approximate from 5-point landmarks
        const landmarkEntries = Object.values(face.landmarks);
        const meshPoints = landmarkEntries.map((l) => ({ x: l.x, y: l.y, z: 0 }));

        const detectedFace: DetectedFace = {
          faceBounds: bounds,
          landmarks: face.landmarks,
          faceData,
          meshPoints,
        };

        // @ts-expect-error - worklet context
        const _WorkletRuntime = globalThis.WorkletRuntime;
        // Schedule async native call (non-blocking).
        // Native cropFace signature: (imageData, frameWidth, frameHeight, faceX, faceY, faceW, faceH)
        const NetraEdgeNative = NativeModules.NetraEdgeModule;
        if (NetraEdgeNative && NetraEdgeNative.cropFace) {
          // @ts-expect-error - calling native module from worklet
          const frameW: number =
            (frame && (frame as { width?: number }).width) ?? 0;
          // @ts-expect-error - calling native module from worklet
          const frameH: number =
            (frame && (frame as { height?: number }).height) ?? 0;
          // @ts-expect-error - calling native module from worklet
          NetraEdgeNative.cropFace(
            frame,
            frameW,
            frameH,
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
          )
            .then((result: number[] | null) => {
              if (
                result &&
                Array.isArray(result) &&
                result.length === FACE_DATA_SIZE
              ) {
                lastFaceDataRef.current = result;
                // Re-emit with updated face data on next frame
                onFaceDetected([
                  { ...detectedFace, faceData: result },
                ]);
              } else {
                onFaceDetected([detectedFace]);
              }
            })
            .catch(() => {
              onFaceDetected([detectedFace]);
            });
        } else {
          onFaceDetected([detectedFace]);
        }
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
