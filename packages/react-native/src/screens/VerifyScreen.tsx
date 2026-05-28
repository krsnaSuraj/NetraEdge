/**
 * VerifyScreen — real-time face verification with liveness.
 */

import React, { useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export interface VerifyResult {
  readonly matched: boolean;
  readonly userId: string | null;
  readonly confidence: number;
  readonly livenessPassed: boolean;
}

export function VerifyScreen({
  onBack,
}: {
  onBack: () => void;
}): React.JSX.Element {
  const [result] = useState<VerifyResult | null>(null);

  return (
    <View style={styles.container}>
      <View style={styles.cameraContainer}>
        {/* Camera will be rendered here */}
        <View style={styles.cameraPlaceholder}>
          <Text style={styles.placeholderText}>Camera Preview</Text>
        </View>

        {/* Status overlay */}
        <View style={styles.statusOverlay}>
          <View
            style={[
              styles.statusBadge,
              result?.matched ? styles.badgeSuccess : styles.badgePending,
            ]}
          >
            <Text style={styles.statusText}>{status}</Text>
          </View>
        </View>

        {/* Result card */}
        {result && (
          <View style={styles.resultCard}>
            <View
              style={[
                styles.resultBar,
                result.matched ? styles.barSuccess : styles.barFailed,
              ]}
            />
            <View style={styles.resultContent}>
              <Text style={styles.resultTitle}>
                {result.matched ? 'Verified' : 'Not Recognized'}
              </Text>
              {result.userId && (
                <Text style={styles.resultDetail}>User: {result.userId}</Text>
              )}
              <Text style={styles.resultDetail}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </Text>
              <Text style={styles.resultDetail}>
                Liveness: {result.livenessPassed ? 'Passed' : 'Failed'}
              </Text>
            </View>
          </View>
        )}
      </View>

      {/* Back button */}
      <TouchableOpacity style={styles.backButton} onPress={onBack}>
        <Text style={styles.backButtonText}>Back</Text>
      </TouchableOpacity>
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
  cameraPlaceholder: {
    flex: 1,
    backgroundColor: '#111',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 14,
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
  statusText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  resultCard: {
    position: 'absolute',
    bottom: 100,
    left: 24,
    right: 24,
    backgroundColor: '#1c1c1e',
    borderRadius: 12,
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
    padding: 16,
  },
  resultTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  resultDetail: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 13,
    marginTop: 2,
  },
  backButton: {
    position: 'absolute',
    bottom: 40,
    left: 24,
    right: 24,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
