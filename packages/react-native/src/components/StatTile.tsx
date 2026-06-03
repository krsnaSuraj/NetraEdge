/**
 * StatTile — small card showing a single label/value pair.
 * Used on the HomeScreen info row.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { palette, radius, spacing, type as typeScale } from '../theme';

export interface StatTileProps {
  readonly value: string;
  readonly label: string;
  readonly accent?: 'blue' | 'green' | 'amber';
}

export function StatTile({ value, label, accent = 'blue' }: StatTileProps): React.JSX.Element {
  const accentColor =
    accent === 'green' ? palette.success : accent === 'amber' ? palette.warning : palette.primary;
  return (
    <View style={styles.card}>
      <View style={[styles.dot, { backgroundColor: accentColor }]} />
      <Text style={[styles.value, { color: accentColor }]}>{value}</Text>
      <Text style={styles.label}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: palette.bgElevated,
    borderRadius: radius.lg,
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: palette.border,
    gap: spacing.xs,
  },
  dot: { width: 6, height: 6, borderRadius: 3, marginBottom: spacing.xs },
  value: { ...typeScale.title, fontSize: 22, marginTop: 2 },
  label: { ...typeScale.micro, color: palette.textTertiary, textTransform: 'uppercase', textAlign: 'center' },
});
