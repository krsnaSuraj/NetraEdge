/**
 * ActionRow + ListRow — primitives used in SettingsScreen rows.
 */

import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Icon, type IconName } from './Icon';
import { Haptics } from './Haptics';
import { palette, spacing, type as typeScale } from '../theme';

export interface ActionRowProps {
  readonly label: string;
  readonly onPress: () => void;
  readonly icon?: IconName;
  readonly trailingText?: string;
  readonly danger?: boolean;
  readonly showChevron?: boolean;
}

export function ActionRow({
  label, onPress, icon, trailingText, danger = false, showChevron = true,
}: ActionRowProps): React.JSX.Element {
  const fg = danger ? palette.danger : palette.textPrimary;
  return (
    <TouchableOpacity
      style={styles.row}
      onPress={() => { Haptics.selection(); onPress(); }}
      activeOpacity={0.7}
    >
      {icon && <Icon name={icon} size={20} color={fg} />}
      <Text style={[styles.label, { color: fg, flex: 1 }]}>{label}</Text>
      {trailingText && <Text style={styles.meta}>{trailingText}</Text>}
      {showChevron && <Text style={styles.chev}>›</Text>}
    </TouchableOpacity>
  );
}

export interface ListRowProps {
  readonly label: string;
  readonly meta?: string;
  readonly divider?: boolean;
  readonly children?: React.ReactNode;
}

export function ListRow({ label, meta, divider = false, children }: ListRowProps): React.JSX.Element {
  return (
    <View style={[styles.row, divider && styles.divider]}>
      <Text style={styles.label}>{label}</Text>
      {meta && <Text style={styles.meta}>{meta}</Text>}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.lg,
    gap: spacing.md,
  },
  divider: {
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },
  label: { ...typeScale.body, color: palette.textPrimary, flex: 1 },
  meta: { ...typeScale.caption, color: palette.textTertiary },
  chev: { color: palette.textDisabled, fontSize: 22, fontWeight: '300' },
});
