/**
 * Button — primary, secondary, and danger variants.
 * Includes optional loading state and haptics on press.
 */

import React from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  type ViewStyle,
} from 'react-native';
import { Icon, type IconName } from './Icon';
import { Haptics } from './Haptics';
import { palette, radius, spacing, type as typeScale } from '../theme';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

export interface ButtonProps {
  readonly label: string;
  readonly onPress: () => void;
  readonly variant?: ButtonVariant;
  readonly icon?: IconName;
  readonly trailingIcon?: IconName;
  readonly loading?: boolean;
  readonly disabled?: boolean;
  readonly fullWidth?: boolean;
  readonly style?: ViewStyle;
}

export function Button({
  label,
  onPress,
  variant = 'primary',
  icon,
  trailingIcon,
  loading = false,
  disabled = false,
  fullWidth = true,
  style,
}: ButtonProps): React.JSX.Element {
  const v = variantStyle(variant, disabled);
  return (
    <TouchableOpacity
      style={[
        styles.base,
        v.bg,
        v.border,
        fullWidth && styles.fullWidth,
        disabled && styles.disabled,
        style,
      ]}
      activeOpacity={0.85}
      onPress={() => {
        if (disabled || loading) return;
        Haptics.tap();
        onPress();
      }}
    >
      {loading ? (
        <ActivityIndicator size="small" color={v.fg} />
      ) : (
        <View style={styles.inner}>
          {icon && <Icon name={icon} size={18} color={v.fg} />}
          <Text style={[styles.label, { color: v.fg }]}>{label}</Text>
          {trailingIcon && <Icon name={trailingIcon} size={18} color={v.fg} />}
        </View>
      )}
    </TouchableOpacity>
  );
}

function variantStyle(v: ButtonVariant, disabled: boolean) {
  if (disabled) {
    return { bg: { backgroundColor: palette.glass }, border: { borderColor: palette.border }, fg: palette.textDisabled };
  }
  switch (v) {
    case 'primary':
      return { bg: { backgroundColor: palette.primary }, border: { borderColor: palette.primary }, fg: '#ffffff' };
    case 'secondary':
      return { bg: { backgroundColor: palette.bgElevated }, border: { borderColor: palette.border }, fg: palette.textPrimary };
    case 'danger':
      return { bg: { backgroundColor: palette.dangerSoft }, border: { borderColor: palette.dangerBorder }, fg: palette.danger };
    case 'ghost':
      return { bg: { backgroundColor: 'transparent' }, border: { borderColor: 'transparent' }, fg: palette.textSecondary };
  }
}

const styles = StyleSheet.create({
  base: {
    minHeight: 52,
    borderRadius: radius.lg,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
  },
  fullWidth: { alignSelf: 'stretch' },
  disabled: { opacity: 0.5 },
  inner: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  label: { ...typeScale.heading, fontSize: 15 },
});
