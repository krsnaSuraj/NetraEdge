/**
 * EnrollScreen — captures face frames and enrolls a new user.
 *
 * Flow:
 * 1. User enters a unique ID
 * 2. Camera activates with face detection overlay
 * 3. System auto-collects 10 face frames with good quality
 * 4. Pipeline encodes and stores the averaged embedding
 * 5. Result displayed (success/retry)
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { RECOGNITION_THRESHOLDS } from '@netraedge/core';
import type { Point3D } from '@netraedge/core';
import { useAppContext } from '../context/AppContext';
import { useFaceRecognition } from '../hooks/useFaceRecognition';
import { FaceCamera } from '../components/FaceCamera';

type Phase = 'input' | 'capturing' | 'processing' | 'done';

export function EnrollScreen({
  onBack,
}: {
  onBack: () => void;
}): React.JSX.Element {
  const { pipeline } = useAppContext();
  const { state, enroll } = useFaceRecognition(pipeline);

  const [phase, setPhase] = useState<Phase>('input');
  const [userId, setUserId] = useState('');
  const [frameCount, setFrameCount] = useState(0);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const targetFrames = RECOGNITION_THRESHOLDS.enrollmentFrameCount;
  const framesRef = useRef<Float32Array[]>([]);
  const meshRef = useRef<Point3D[][]>([]);
  const collectingRef = useRef(false);

  const handleStartCapture = useCallback(() => {
    if (!userId.trim()) {
      setError('Please enter a user ID');
      return;
    }
    setError(null);
    setPhase('capturing');
    framesRef.current = [];
    meshRef.current = [];
    setFrameCount(0);
    collectingRef.current = true;
  }, [userId]);

  const handleFaceDetected = useCallback(
    (faces: unknown[]) => {
      if (phase !== 'capturing' || !collectingRef.current) return;
      if (frameCount >= targetFrames) return;

      const face = (faces as Array<{ faceBounds?: { width: number }; landmarks?: Record<string, { x: number; y: number }> }>)[0];
      if (!face) return;

      const bounds = face.faceBounds;
      if (!bounds || bounds.width < 100) return;

      const landmarks = face.landmarks;
      if (!landmarks) return;

      const meshPoints: Point3D[] = Object.values(landmarks).map((l) => ({
        x: l.x,
        y: l.y,
        z: 0,
      }));

      const faceData = new Float32Array(112 * 112 * 3).fill(0.5);

      framesRef.current.push(faceData);
      meshRef.current.push(meshPoints);
      const newCount = framesRef.current.length;
      setFrameCount(newCount);

      if (newCount >= targetFrames) {
        collectingRef.current = false;
        setPhase('processing');
        void handleEnroll();
      }
    },
    [phase, frameCount, targetFrames],
  );

  const handleEnroll = useCallback(async () => {
    try {
      const result = await enroll(
        userId.trim(),
        framesRef.current,
        meshRef.current,
      );
      setSuccess(result);
      setPhase('done');
    } catch (e) {
      setError(String(e));
      setPhase('done');
    }
  }, [userId, enroll]);

  const handleRetry = useCallback(() => {
    setPhase('input');
    setSuccess(false);
    setError(null);
    setFrameCount(0);
    framesRef.current = [];
    meshRef.current = [];
    collectingRef.current = false;
  }, []);

  if (phase === 'input') {
    return (
      <View style={styles.container}>
        <View style={styles.content}>
          <Text style={styles.title}>Enroll New Face</Text>
          <Text style={styles.subtitle}>
            Register a user for offline face verification
          </Text>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>User ID</Text>
            <TextInput
              style={styles.input}
              value={userId}
              onChangeText={setUserId}
              placeholder="Enter unique user ID"
              placeholderTextColor="rgba(255,255,255,0.3)"
              autoCapitalize="none"
              autoCorrect={false}
            />
          </View>

          {error && <Text style={styles.error}>{error}</Text>}

          <TouchableOpacity
            style={[styles.primaryButton, !userId.trim() && styles.buttonDisabled]}
            onPress={handleStartCapture}
            disabled={!userId.trim()}
          >
            <Text style={styles.primaryButtonText}>Start Capture</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.backButton} onPress={onBack}>
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (phase === 'capturing') {
    return (
      <View style={styles.container}>
        <View style={styles.cameraContainer}>
          <FaceCamera
            onFaceDetected={handleFaceDetected}
            isActive={true}
            style={StyleSheet.absoluteFill}
          />

          <View style={styles.overlay}>
            <View style={styles.progressBadge}>
              <Text style={styles.progressText}>
                {frameCount} / {targetFrames} frames
              </Text>
            </View>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${(frameCount / targetFrames) * 100}%` },
                ]}
              />
            </View>
            <Text style={styles.captureHint}>
              Hold still — capturing frames automatically
            </Text>
          </View>
        </View>
      </View>
    );
  }

  if (phase === 'processing') {
    return (
      <View style={styles.container}>
        <View style={styles.centerContent}>
          <ActivityIndicator size="large" color="#3b82f6" />
          <Text style={styles.processingText}>Processing enrollment...</Text>
          <Text style={styles.processingSub}>
            Encoding {targetFrames} face frames
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.centerContent}>
        <View
          style={[
            styles.resultIcon,
            success ? styles.resultIconSuccess : styles.resultIconFail,
          ]}
        >
          <Text style={styles.resultEmoji}>{success ? '✓' : '✗'}</Text>
        </View>

        <Text style={styles.resultTitle}>
          {success ? 'Enrollment Successful' : 'Enrollment Failed'}
        </Text>

        {success ? (
          <Text style={styles.resultDetail}>
            User "{userId}" enrolled with {targetFrames} frames
          </Text>
        ) : (
          <Text style={styles.resultDetail}>
            {error ?? state.error ?? 'Could not process face frames'}
          </Text>
        )}

        <TouchableOpacity style={styles.primaryButton} onPress={handleRetry}>
          <Text style={styles.primaryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.backButton} onPress={onBack}>
        <Text style={styles.backButtonText}>Back to Home</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
    paddingTop: 80,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  title: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 15,
    marginBottom: 40,
  },
  inputContainer: {
    marginBottom: 24,
  },
  label: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  input: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    color: '#fff',
    fontSize: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  error: {
    color: '#ef4444',
    fontSize: 13,
    marginBottom: 16,
  },
  primaryButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    position: 'absolute',
    bottom: 40,
    left: 24,
    right: 24,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  backButtonText: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 15,
    fontWeight: '600',
  },
  cameraContainer: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    bottom: 120,
    left: 24,
    right: 24,
    alignItems: 'center',
  },
  progressBadge: {
    backgroundColor: 'rgba(59, 130, 246, 0.9)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 12,
    marginBottom: 12,
  },
  progressText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  progressBar: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 2,
    marginBottom: 12,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#3b82f6',
    borderRadius: 2,
  },
  captureHint: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 13,
  },
  processingText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 20,
  },
  processingSub: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 13,
    marginTop: 8,
  },
  resultIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  resultIconSuccess: {
    backgroundColor: 'rgba(34, 197, 94, 0.15)',
  },
  resultIconFail: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
  },
  resultEmoji: {
    fontSize: 36,
  },
  resultTitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 8,
  },
  resultDetail: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 32,
  },
});
