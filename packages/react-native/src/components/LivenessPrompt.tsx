/**
 * LivenessPrompt — UI overlay for liveness challenge.
 *
 * Shows the user what action to perform (blink, smile, turn head)
 * and displays the current liveness status.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

export interface LivenessPromptProps {
  readonly prompt: string | null;
  readonly blinkCount: number;
  readonly isLive: boolean;
}

export function LivenessPrompt({
  prompt,
  blinkCount,
  isLive,
}: LivenessPromptProps): React.JSX.Element {
  return (
    <View style={styles.container}>
      <View
        style={[
          styles.badge,
          isLive ? styles.badgeSuccess : styles.badgePending,
        ]}
      >
        <Text style={styles.badgeText}>
          {isLive ? 'LIVE' : 'CHECKING'}
        </Text>
      </View>

      {prompt && !isLive && (
        <View style={styles.promptContainer}>
          <Text style={styles.promptText}>{prompt}</Text>
          {blinkCount > 0 && (
            <Text style={styles.counterText}>
              Blinks: {blinkCount}
            </Text>
          )}
        </View>
      )}

      {isLive && (
        <View style={styles.successContainer}>
          <Text style={styles.successText}>Liveness Verified</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  badge: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 12,
  },
  badgeSuccess: {
    backgroundColor: 'rgba(34, 197, 94, 0.9)',
  },
  badgePending: {
    backgroundColor: 'rgba(234, 179, 8, 0.9)',
  },
  badgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
  },
  promptContainer: {
    marginTop: 12,
    alignItems: 'center',
  },
  promptText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  counterText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    marginTop: 4,
  },
  successContainer: {
    marginTop: 12,
  },
  successText: {
    color: '#22c55e',
    fontSize: 14,
    fontWeight: '600',
  },
});
