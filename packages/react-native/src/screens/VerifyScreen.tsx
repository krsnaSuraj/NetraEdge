/**
 * VerifyScreen — real-time face verification with liveness detection.
 *
 * Flow:
 * 1. Camera activates with face detection
 * 2. Liveness check runs (blink detection via LivenessOrchestrator)
 * 3. When liveness passes, verification runs against enrolled users
 * 4. Result displayed with confidence score
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
  const liveness = useLivenessCheck(
    pipeline ? (pipeline as unknown as { _liveness: import('@netraedge/core').LivenessOrchestrator })._liveness : null,
  );

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

      const faceData = new Float32Array(112 * 112 * 3).fill(0.5);

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

  const statusText = (): string => {
    switch (phase) {
      case 'scanning':
        return 'Position face in frame';
      case 'liveness':
        return liveness.state.prompt ?? 'Blink your eyes';
      case 'verifying':
        return 'Verifying identity...';
      case 'result':
        return verifyResult?.matched ? 'Verified' : 'Not Recognized';
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.cameraContainer}>
        <FaceCamera
          onFaceDetected={handleFaceDetected}
          isActive={phase !== 'result'}
          style={StyleSheet.absoluteFill}
        />

        <View style={styles.statusOverlay}>
          <View
            style={[
              styles.statusBadge,
              phase === 'result' && verifyResult?.matched
                ? styles.badgeSuccess
                : phase === 'result'
                  ? styles.badgeFailed
                  : styles.badgePending,
            ]}
          >
            <Text style={styles.statusText}>{statusText()}</Text>
          </View>
        </View>

        {phase === 'liveness' && (
          <View style={styles.livenessOverlay}>
            <View style={styles.livenessBadge}>
              <Text style={styles.livenessText}>
                Blinks: {liveness.state.blinkCount}
              </Text>
            </View>
            <View style={styles.livenessBadge}>
              <Text style={styles.livenessText}>
                Texture: {(liveness.state.textureScore * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        )}

        {phase === 'verifying' && (
          <View style={styles.verifyingOverlay}>
            <ActivityIndicator size="large" color="#3b82f6" />
            <Text style={styles.verifyingText}>Verifying identity...</Text>
          </View>
        )}

        {phase === 'result' && verifyResult && (
          <View style={styles.resultCard}>
            <View
              style={[
                styles.resultBar,
                verifyResult.matched ? styles.barSuccess : styles.barFailed,
              ]}
            />
            <View style={styles.resultContent}>
              <Text style={styles.resultTitle}>
                {verifyResult.matched ? 'Verified' : 'Not Recognized'}
              </Text>
              {verifyResult.userId && (
                <Text style={styles.resultDetail}>
                  User: {verifyResult.userId}
                </Text>
              )}
              <Text style={styles.resultDetail}>
                Confidence: {(verifyResult.confidence * 100).toFixed(1)}%
              </Text>
              <Text style={styles.resultDetail}>
                Liveness: {verifyResult.livenessPassed ? 'Passed' : 'Failed'}
              </Text>
            </View>

            {verifyResult.matched && (
              <View style={styles.syncStatus}>
                <Text style={styles.syncText}>
                  {syncManager
                    ? `Pending sync: ${syncManager.pendingCount}`
                    : 'Offline — queued for sync'}
                </Text>
              </View>
            )}
          </View>
        )}
      </View>

      <View style={styles.bottomBar}>
        {phase === 'result' && (
          <TouchableOpacity
            style={styles.retryButton}
            onPress={handleReset}
          >
            <Text style={styles.retryButtonText}>Scan Again</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.backButton} onPress={onBack}>
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  cameraContainer: {
    flex: 1,
  },
  statusOverlay: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  statusBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 12,
  },
  badgePending: {
    backgroundColor: 'rgba(234, 179, 8, 0.9)',
  },
  badgeSuccess: {
    backgroundColor: 'rgba(34, 197, 94, 0.9)',
  },
  badgeFailed: {
    backgroundColor: 'rgba(239, 68, 68, 0.9)',
  },
  statusText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  livenessOverlay: {
    position: 'absolute',
    top: 100,
    left: 24,
    flexDirection: 'row',
    gap: 8,
  },
  livenessBadge: {
    backgroundColor: 'rgba(59, 130, 246, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  livenessText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  verifyingOverlay: {
    position: 'absolute',
    bottom: 200,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  verifyingText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginTop: 12,
  },
  resultCard: {
    position: 'absolute',
    bottom: 120,
    left: 24,
    right: 24,
    backgroundColor: '#1c1c1e',
    borderRadius: 16,
    overflow: 'hidden',
  },
  resultBar: {
    height: 3,
  },
  barSuccess: {
    backgroundColor: '#22c55e',
  },
  barFailed: {
    backgroundColor: '#ef4444',
  },
  resultContent: {
    padding: 20,
  },
  resultTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 12,
  },
  resultDetail: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    marginTop: 4,
  },
  syncStatus: {
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  syncText: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 12,
  },
  bottomBar: {
    paddingHorizontal: 24,
    paddingBottom: 40,
    gap: 12,
  },
  retryButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
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
});
