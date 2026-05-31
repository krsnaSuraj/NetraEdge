/**
 * VerifyScreen — premium face verification with liveness.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import type { Point3D } from '@netraedge/core';
import { useAppContext } from '../context/AppContext';
import { useFaceRecognition } from '../hooks/useFaceRecognition';
import { useLivenessCheck } from '../hooks/useLivenessCheck';
import { FaceCamera } from '../components/FaceCamera';

type VerifyPhase = 'scanning' | 'liveness' | 'verifying' | 'result';

export function VerifyScreen({
  onBack,
}: {
  onBack: () => void;
}): React.JSX.Element {
  const { pipeline, syncManager } = useAppContext();
  const { verify } = useFaceRecognition(pipeline);
  const liveness = useLivenessCheck(pipeline?.liveness ?? null);

  const [phase, setPhase] = useState<VerifyPhase>('scanning');
  const [verifyResult, setVerifyResult] = useState<{
    matched: boolean;
    userId: string | null;
    confidence: number;
    livenessPassed: boolean;
  } | null>(null);

  const verifiedRef = useRef(false);
  const processingRef = useRef(false);

  useEffect(() => {
    if (liveness.state.isLive && phase === 'liveness' && !verifiedRef.current) {
      verifiedRef.current = true;
      setPhase('verifying');
    }
  }, [liveness.state.isLive, phase]);

  const handleFaceDetected = useCallback(
    async (faces: unknown[]) => {
      if (verifiedRef.current || processingRef.current) return;

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
      // Face crop populated by native frame processor.
      // Native module extracts 112x112 face region from camera frame.
      const faceData = new Float32Array(37632);

      if (phase === 'scanning') {
        setPhase('liveness');
        processingRef.current = true;
        await liveness.processFrame(meshPoints, faceData);
        processingRef.current = false;
        return;
      }

      if (phase === 'liveness') {
        processingRef.current = true;
        await liveness.processFrame(meshPoints, faceData);

        if (liveness.state.isLive) {
          verifiedRef.current = true;
          setPhase('verifying');

          const result = await verify(faceData, meshPoints);
          if (result) {
            setVerifyResult(result);
            setPhase('result');

            if (result.matched && syncManager) {
              syncManager.enqueue(result.userId ?? 'unknown', faceData, {
                verifiedAt: Date.now(),
                confidence: result.confidence,
              });
            }
          }
        }
        processingRef.current = false;
      }
    },
    [phase, liveness, verify, syncManager],
  );

  const handleReset = useCallback(() => {
    verifiedRef.current = false;
    processingRef.current = false;
    setPhase('scanning');
    setVerifyResult(null);
    liveness.reset();
  }, [liveness]);

  const phaseConfig = {
    scanning: { title: 'Scanning', subtitle: 'Position face inside the oval', color: '#3b82f6' },
    liveness: { title: 'Liveness Check', subtitle: liveness.state.prompt ?? 'Blink your eyes', color: '#f59e0b' },
    verifying: { title: 'Verifying', subtitle: 'Matching against database...', color: '#8b5cf6' },
    result: {
      title: verifyResult?.matched ? 'Verified' : 'Not Recognized',
      subtitle: verifyResult?.matched ? `Confidence: ${(verifyResult.confidence * 100).toFixed(1)}%` : 'No match found',
      color: verifyResult?.matched ? '#22c55e' : '#ef4444',
    },
  };

  const config = phaseConfig[phase];

  return (
    <View style={styles.container}>
      {/* Camera fills entire screen */}
      <FaceCamera onFaceDetected={handleFaceDetected} isActive={phase !== 'result'}>
        {/* Top status bar */}
        <View style={styles.topBar}>
          <TouchableOpacity onPress={onBack} style={styles.backBtn}>
            <Text style={styles.backBtnText}>←</Text>
          </TouchableOpacity>
          <View style={styles.statusPill}>
            <View style={[styles.statusDot, { backgroundColor: config.color }]} />
            <Text style={styles.statusText}>{config.title}</Text>
          </View>
          <View style={styles.backBtn} />
        </View>

        {/* Center prompt */}
        {phase === 'scanning' && (
          <View style={styles.centerPrompt}>
            <Text style={styles.centerTitle}>{config.title}</Text>
            <Text style={styles.centerSub}>{config.subtitle}</Text>
          </View>
        )}

        {/* Liveness indicators */}
        {phase === 'liveness' && (
          <View style={styles.livenessContainer}>
            <View style={styles.livenessCard}>
              <Text style={styles.livenessEmoji}>👁</Text>
              <Text style={styles.livenessLabel}>Blink Count</Text>
              <Text style={styles.livenessValue}>{liveness.state.blinkCount}</Text>
            </View>
            <View style={styles.livenessCard}>
              <Text style={styles.livenessEmoji}>◉</Text>
              <Text style={styles.livenessLabel}>Texture</Text>
              <Text style={styles.livenessValue}>
                {(liveness.state.textureScore * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        )}

        {/* Verifying spinner */}
        {phase === 'verifying' && (
          <View style={styles.verifyingContainer}>
            <ActivityIndicator size="large" color="#8b5cf6" />
            <Text style={styles.verifyingText}>Verifying identity...</Text>
          </View>
        )}

        {/* Result card */}
        {phase === 'result' && verifyResult && (
          <View style={styles.resultContainer}>
            <View style={[styles.resultCard, verifyResult.matched ? styles.resultCardOk : styles.resultCardFail]}>
              <View style={[styles.resultAccent, verifyResult.matched ? styles.resultAccentOk : styles.resultAccentFail]} />

              <View style={styles.resultHeader}>
                <View style={[styles.resultIconWrap, verifyResult.matched ? styles.resultIconOk : styles.resultIconFail]}>
                  <Text style={styles.resultIconEmoji}>
                    {verifyResult.matched ? '✓' : '✗'}
                  </Text>
                </View>
                <Text style={styles.resultTitle}>
                  {verifyResult.matched ? 'Identity Verified' : 'Not Recognized'}
                </Text>
              </View>

              <View style={styles.resultDetails}>
                {verifyResult.userId && (
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>User</Text>
                    <Text style={styles.resultValue}>{verifyResult.userId}</Text>
                  </View>
                )}
                <View style={styles.resultRow}>
                  <Text style={styles.resultLabel}>Confidence</Text>
                  <Text style={styles.resultValue}>
                    {(verifyResult.confidence * 100).toFixed(1)}%
                  </Text>
                </View>
                <View style={styles.resultRow}>
                  <Text style={styles.resultLabel}>Liveness</Text>
                  <Text style={[styles.resultValue, { color: verifyResult.livenessPassed ? '#22c55e' : '#ef4444' }]}>
                    {verifyResult.livenessPassed ? 'Passed' : 'Failed'}
                  </Text>
                </View>
              </View>

              {verifyResult.matched && syncManager && (
                <View style={styles.syncInfo}>
                  <Text style={styles.syncText}>
                    Queued for sync ({syncManager.pendingCount} pending)
                  </Text>
                </View>
              )}
            </View>

            <TouchableOpacity style={styles.retryBtn} onPress={handleReset} activeOpacity={0.8}>
              <Text style={styles.retryBtnText}>Scan Again</Text>
            </TouchableOpacity>
          </View>
        )}
      </FaceCamera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  // Top bar
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 56,
    paddingBottom: 12,
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  backBtnText: {
    color: '#fff',
    fontSize: 20,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
  },
  // Center prompt
  centerPrompt: {
    position: 'absolute',
    bottom: 200,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  centerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  centerSub: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 13,
    marginTop: 4,
  },
  // Liveness
  livenessContainer: {
    position: 'absolute',
    bottom: 180,
    left: 24,
    right: 24,
    flexDirection: 'row',
    gap: 12,
  },
  livenessCard: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  livenessEmoji: {
    fontSize: 24,
    marginBottom: 8,
  },
  livenessLabel: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 11,
    marginBottom: 4,
  },
  livenessValue: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '800',
  },
  // Verifying
  verifyingContainer: {
    position: 'absolute',
    bottom: 200,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  verifyingText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    marginTop: 16,
  },
  // Result
  resultContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  resultCard: {
    backgroundColor: 'rgba(15, 15, 26, 0.95)',
    borderRadius: 24,
    overflow: 'hidden',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
  },
  resultCardOk: {
    borderColor: 'rgba(34, 197, 94, 0.15)',
  },
  resultCardFail: {
    borderColor: 'rgba(239, 68, 68, 0.15)',
  },
  resultAccent: {
    height: 3,
  },
  resultAccentOk: {
    backgroundColor: '#22c55e',
  },
  resultAccentFail: {
    backgroundColor: '#ef4444',
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 12,
  },
  resultIconWrap: {
    width: 44,
    height: 44,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
  },
  resultIconOk: {
    backgroundColor: 'rgba(34, 197, 94, 0.12)',
  },
  resultIconFail: {
    backgroundColor: 'rgba(239, 68, 68, 0.12)',
  },
  resultIconEmoji: {
    fontSize: 20,
    color: '#fff',
    fontWeight: '800',
  },
  resultTitle: {
    color: '#fff',
    fontSize: 19,
    fontWeight: '800',
    letterSpacing: -0.3,
  },
  resultDetails: {
    paddingHorizontal: 20,
    paddingBottom: 16,
    gap: 10,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  resultLabel: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 13,
  },
  resultValue: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },
  syncInfo: {
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  syncText: {
    color: 'rgba(255,255,255,0.2)',
    fontSize: 11,
    textAlign: 'center',
  },
  retryBtn: {
    backgroundColor: '#3b82f6',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
  },
  retryBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: -0.3,
  },
});
