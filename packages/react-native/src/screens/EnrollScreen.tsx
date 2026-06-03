/**
 * EnrollScreen — premium face enrollment with camera.
 */

import React, { useCallback, useRef, useState } from 'react';
import {
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
import { FaceCamera, AnimatedFaceRing, Haptics, type DetectedFace } from '../components';

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
      Haptics.warn();
      setError('Please enter a user ID');
      return;
    }
    Haptics.tap();
    setError(null);
    setPhase('capturing');
    framesRef.current = [];
    meshRef.current = [];
    setFrameCount(0);
    collectingRef.current = true;
  }, [userId]);

  const handleFaceDetected = useCallback(
    (faces: DetectedFace[]) => {
      if (phase !== 'capturing' || !collectingRef.current) return;
      if (frameCount >= targetFrames) return;

      const face = faces[0];
      if (!face) return;

      const bounds = face.faceBounds;
      if (!bounds || bounds.width < 100) return;

      const landmarks = face.landmarks;
      if (!landmarks) return;

      const meshPoints: Point3D[] = (Object.values(landmarks) as Array<{ x: number; y: number }>).map((l) => ({
        x: l.x,
        y: l.y,
        z: 0,
      }));

      // Use REAL face data from camera (112x112 normalized RGB)
      // faceData is extracted by the native face crop pipeline
      const faceData = new Float32Array(face.faceData);
      framesRef.current.push(faceData);
      meshRef.current.push(meshPoints);
      const newCount = framesRef.current.length;
      setFrameCount(newCount);
      // Light haptic every 5 frames as gentle feedback
      if (newCount % 5 === 0) Haptics.selection();

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
      const result = await enroll(userId.trim(), framesRef.current, meshRef.current);
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

  // ─── Input Phase ───
  if (phase === 'input') {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onBack} style={styles.backBtn}>
            <Text style={styles.backBtnText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Enroll Face</Text>
          <View style={styles.headerRight} />
        </View>

        <View style={styles.content}>
          <View style={styles.iconCircle}>
            <Text style={styles.iconText}>＋</Text>
          </View>
          <Text style={styles.title}>New Enrollment</Text>
          <Text style={styles.subtitle}>
            Enter a unique ID to register this face
          </Text>

          <View style={styles.inputGroup}>
            <Text style={styles.inputLabel}>USER ID</Text>
            <TextInput
              style={styles.input}
              value={userId}
              onChangeText={setUserId}
              placeholder="e.g. EMP-001"
              placeholderTextColor="rgba(255,255,255,0.2)"
              autoCapitalize="characters"
              autoCorrect={false}
            />
          </View>

          {error && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          <TouchableOpacity
            style={[styles.mainBtn, !userId.trim() && styles.mainBtnDisabled]}
            onPress={handleStartCapture}
            disabled={!userId.trim()}
            activeOpacity={0.8}
          >
            <Text style={styles.mainBtnText}>Start Capture</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ─── Capturing Phase ───
  if (phase === 'capturing') {
    const progress = frameCount / targetFrames;
    return (
      <View style={styles.container}>
        <FaceCamera onFaceDetected={handleFaceDetected} isActive={true}>
          {/* Top overlay */}
          <View style={styles.captureTopOverlay}>
            <Text style={styles.captureTitle}>Hold Still</Text>
            <Text style={styles.captureSubtitle}>
              Position face inside the oval
            </Text>
          </View>

          {/* Center ring — animated breathing indicator */}
          <View style={styles.captureCenter}>
            <AnimatedFaceRing size={280} active={true} progress={progress} />
          </View>

          {/* Bottom overlay */}
          <View style={styles.captureBottomOverlay}>
            <View style={styles.progressContainer}>
              <View style={styles.progressBarBg}>
                <View style={[styles.progressBarFill, { width: `${progress * 100}%` }]} />
              </View>
              <Text style={styles.progressText}>
                {frameCount} / {targetFrames}
              </Text>
            </View>

            {/* Frame dots */}
            <View style={styles.dotsRow}>
              {Array.from({ length: targetFrames }, (_, i) => (
                <View
                  key={i}
                  style={[styles.dot, i < frameCount && styles.dotFilled]}
                />
              ))}
            </View>
          </View>
        </FaceCamera>
      </View>
    );
  }

  // ─── Processing Phase ───
  if (phase === 'processing') {
    return (
      <View style={styles.container}>
        <View style={styles.centerContent}>
          <AnimatedFaceRing size={200} active={true} progress={0.5} color="#22c55e" />
          <Text style={styles.processingTitle}>Processing</Text>
          <Text style={styles.processingSub}>
            Encoding {targetFrames} face frames...
          </Text>
        </View>
      </View>
    );
  }

  // ─── Done Phase ───
  return (
    <View style={styles.container}>
      <View style={styles.centerContent}>
        <View style={[styles.resultCircle, success ? styles.resultCircleOk : styles.resultCircleFail]}>
          <Text style={styles.resultEmoji}>{success ? '✓' : '✗'}</Text>
        </View>

        <Text style={styles.resultTitle}>
          {success ? 'Enrolled Successfully' : 'Enrollment Failed'}
        </Text>

        {success ? (
          <View style={styles.resultInfo}>
            <Text style={styles.resultUserId}>{userId}</Text>
            <Text style={styles.resultDetail}>{targetFrames} frames captured</Text>
          </View>
        ) : (
          <Text style={styles.resultDetail}>
            {error ?? state.error ?? 'Could not process face frames'}
          </Text>
        )}

        <TouchableOpacity style={styles.mainBtn} onPress={handleRetry} activeOpacity={0.8}>
          <Text style={styles.mainBtnText}>Try Again</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryBtn} onPress={onBack} activeOpacity={0.8}>
          <Text style={styles.secondaryBtnText}>Back to Home</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#050510',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 56,
    paddingBottom: 16,
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.06)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  backBtnText: {
    color: '#fff',
    fontSize: 20,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
  },
  headerRight: {
    width: 40,
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconCircle: {
    width: 72,
    height: 72,
    borderRadius: 24,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.2)',
  },
  iconText: {
    fontSize: 32,
    color: '#3b82f6',
  },
  title: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: -0.5,
    marginBottom: 8,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 14,
    marginBottom: 32,
    textAlign: 'center',
  },
  inputGroup: {
    width: '100%',
    marginBottom: 20,
  },
  inputLabel: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.5,
    marginBottom: 8,
  },
  input: {
    backgroundColor: 'rgba(255,255,255,0.04)',
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 14,
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    width: '100%',
  },
  errorBox: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.2)',
    width: '100%',
  },
  errorText: {
    color: '#ef4444',
    fontSize: 13,
    textAlign: 'center',
  },
  mainBtn: {
    backgroundColor: '#3b82f6',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    width: '100%',
  },
  mainBtnDisabled: {
    opacity: 0.3,
  },
  mainBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: -0.3,
  },
  secondaryBtn: {
    paddingVertical: 14,
    alignItems: 'center',
    width: '100%',
  },
  secondaryBtnText: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 14,
    fontWeight: '600',
  },
  // Capture overlay
  captureTopOverlay: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  captureTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  captureSubtitle: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 13,
    marginTop: 4,
  },
  captureCenter: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureBottomOverlay: {
    position: 'absolute',
    bottom: 80,
    left: 24,
    right: 24,
    alignItems: 'center',
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    marginBottom: 16,
  },
  progressBarBg: {
    flex: 1,
    height: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 2,
    marginRight: 12,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#3b82f6',
    borderRadius: 2,
  },
  progressText: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 13,
    fontWeight: '600',
    minWidth: 40,
    textAlign: 'right',
  },
  dotsRow: {
    flexDirection: 'row',
    gap: 6,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  dotFilled: {
    backgroundColor: '#3b82f6',
  },
  // Processing
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  processingCircle: {
    width: 80,
    height: 80,
    borderRadius: 24,
    backgroundColor: 'rgba(59, 130, 246, 0.08)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  processingTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 8,
  },
  processingSub: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 13,
  },
  // Result
  resultCircle: {
    width: 80,
    height: 80,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  resultCircleOk: {
    backgroundColor: 'rgba(34, 197, 94, 0.12)',
    borderWidth: 2,
    borderColor: 'rgba(34, 197, 94, 0.3)',
  },
  resultCircleFail: {
    backgroundColor: 'rgba(239, 68, 68, 0.12)',
    borderWidth: 2,
    borderColor: 'rgba(239, 68, 68, 0.3)',
  },
  resultEmoji: {
    fontSize: 32,
    color: '#fff',
  },
  resultTitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '800',
    marginBottom: 8,
    letterSpacing: -0.3,
  },
  resultInfo: {
    alignItems: 'center',
    marginBottom: 32,
  },
  resultUserId: {
    color: '#3b82f6',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  resultDetail: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 13,
    textAlign: 'center',
    marginBottom: 32,
  },
});
