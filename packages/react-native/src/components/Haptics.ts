/**
 * Haptics — cross-platform tactile feedback.
 *
 * Uses the built-in Vibration API so we don't need react-native-haptic-feedback
 * (which requires native linking). Provides richer patterns: tap, success, warn,
 * error, and selection.
 *
 * - tap: 10ms — button presses
 * - success: 30ms (pause 50ms) 50ms — verification success
 * - warn: 60ms (pause 100ms) 60ms — soft warning
 * - error: triple 80ms — spoof / hard failure
 * - selection: 15ms — segment / radio change
 */

import { Platform, Vibration } from 'react-native';

export type HapticPattern = 'tap' | 'success' | 'warn' | 'error' | 'selection';

const PATTERNS: Record<HapticPattern, number | number[]> = {
  tap: 10,
  selection: 15,
  success: [0, 30, 50, 50],
  warn: [0, 60, 100, 60],
  error: [0, 80, 80, 80, 80, 80],
};

export function haptic(pattern: HapticPattern = 'tap'): void {
  try {
    if (Platform.OS === 'ios') {
      // iOS ignores Vibration API; in production, swap for
      // react-native-haptic-feedback. This is a safe no-op fallback.
      return;
    }
    Vibration.vibrate(PATTERNS[pattern] as number | number[]);
  } catch {
    // noop
  }
}

export const Haptics = {
  tap: () => haptic('tap'),
  success: () => haptic('success'),
  warn: () => haptic('warn'),
  error: () => haptic('error'),
  selection: () => haptic('selection'),
};
