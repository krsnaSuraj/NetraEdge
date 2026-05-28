/**
 * ResultCard — displays face verification result.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

export interface ResultCardProps {
  readonly matched: boolean;
  readonly userId: string | null;
  readonly confidence: number;
  readonly livenessPassed: boolean;
}

export function ResultCard({
  matched,
  userId,
  confidence,
  livenessPassed,
}: ResultCardProps): React.JSX.Element {
  return (
    <View style={styles.container}>
      <View
        style={[
          styles.statusBar,
          matched ? styles.statusMatch : styles.statusNoMatch,
        ]}
      />

      <View style={styles.content}>
        <Text style={styles.title}>
          {matched ? 'Verified' : 'Not Recognized'}
        </Text>

        {userId && (
          <Text style={styles.userId}>User: {userId}</Text>
        )}

        <Text style={styles.detail}>
          Confidence: {(confidence * 100).toFixed(1)}%
        </Text>

        <Text style={styles.detail}>
          Liveness: {livenessPassed ? 'Passed' : 'Failed'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1c1c1e',
    borderRadius: 12,
    overflow: 'hidden',
    marginHorizontal: 16,
  },
  statusBar: {
    height: 4,
  },
  statusMatch: {
    backgroundColor: '#22c55e',
  },
  statusNoMatch: {
    backgroundColor: '#ef4444',
  },
  content: {
    padding: 16,
  },
  title: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  userId: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 14,
    marginBottom: 4,
  },
  detail: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 12,
    marginTop: 2,
  },
});
