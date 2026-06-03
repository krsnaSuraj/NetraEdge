/**
 * Design tokens — single source of truth for colors, spacing, radii, type.
 * Inspired by Material 3 + Apple Human Interface Guidelines.
 */

export const palette = {
  bg: '#050510',
  bgElevated: '#0f0f1a',
  bgSunken: '#08081a',
  border: 'rgba(255,255,255,0.06)',
  borderStrong: 'rgba(255,255,255,0.12)',

  // Brand
  primary: '#3b82f6',          // blue 500
  primarySoft: 'rgba(59,130,246,0.12)',
  primaryBorder: 'rgba(59,130,246,0.30)',
  primaryText: '#60a5fa',

  // Semantic
  success: '#22c55e',
  successSoft: 'rgba(34,197,94,0.10)',
  successBorder: 'rgba(34,197,94,0.20)',
  warning: '#f59e0b',
  warningSoft: 'rgba(245,158,11,0.12)',
  warningBorder: 'rgba(245,158,11,0.25)',
  danger: '#ef4444',
  dangerSoft: 'rgba(239,68,68,0.12)',
  dangerBorder: 'rgba(239,68,68,0.25)',

  // Text
  textPrimary: '#ffffff',
  textSecondary: 'rgba(255,255,255,0.65)',
  textTertiary: 'rgba(255,255,255,0.40)',
  textDisabled: 'rgba(255,255,255,0.20)',

  // Surfaces
  scrim: 'rgba(0,0,0,0.55)',
  glass: 'rgba(255,255,255,0.04)',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 48,
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
  pill: 999,
} as const;

export const type = {
  display: { fontSize: 32, fontWeight: '800' as const, letterSpacing: -0.8 },
  title:   { fontSize: 22, fontWeight: '700' as const, letterSpacing: -0.4 },
  heading: { fontSize: 17, fontWeight: '700' as const, letterSpacing: -0.2 },
  body:    { fontSize: 15, fontWeight: '500' as const },
  caption: { fontSize: 13, fontWeight: '500' as const },
  micro:   { fontSize: 11, fontWeight: '600' as const, letterSpacing: 0.4 },
} as const;

export const motion = {
  fast: 150,
  base: 250,
  slow: 400,
  pulse: 1200,
} as const;
