/**
 * Section — labelled card section used in SettingsScreen.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { palette, radius, spacing, type as typeScale } from '../theme';

export interface SectionProps {
  readonly label: string;
  readonly children: React.ReactNode;
}

export function Section({ label, children }: SectionProps): React.JSX.Element {
  return (
    <View style={styles.wrap}>
      <Text style={styles.label}>{label}</Text>
      <View style={styles.card}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: spacing.xl },
  label: {
    ...typeScale.micro,
    color: palette.textTertiary,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
    paddingHorizontal: spacing.sm,
  },
  card: {
    backgroundColor: palette.bgElevated,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: palette.border,
    overflow: 'hidden',
  },
});
