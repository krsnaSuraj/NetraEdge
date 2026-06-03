/**
 * HomeScreen — premium main menu with enroll/verify/settings.
 * Uses i18n (en/hi) via the t() helper and the design tokens.
 */

import React, { useEffect, useState } from 'react';
import { StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { t, subscribeLocale, type LocaleCode } from '../i18n';
import { Icon, type IconName, StatTile } from '../components';
import { palette, radius, spacing, type as typeScale } from '../theme';
import { Haptics } from '../components/Haptics';

export interface HomeScreenProps {
  readonly onNavigate: (screen: 'enroll' | 'verify' | 'settings') => void;
}

function useTicker(): LocaleCode {
  const [, setN] = useState(0);
  useEffect(() => subscribeLocale(() => setN((n) => n + 1)), []);
  return require('../i18n').getLocale();
}

export function HomeScreen({ onNavigate }: HomeScreenProps): React.JSX.Element {
  useTicker(); // re-render on locale change

  // Display the actual embedding dim of the loaded model. -1 while loading,
  // 128 for the pre-trained MobileFaceNet on disk, 192 for the Day-2
  // fine-tuned GhostFaceNet-W1.
  const [embeddingDim, setEmbeddingDim] = useState<number>(-1);
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      const netraEdge = require('../native/NetraEdgeNative').NetraEdgeNative;
      netraEdge.getEmbeddingDim()
        .then((d: number) => { if (!cancelled) setEmbeddingDim(d); })
        .catch(() => { if (!cancelled) setEmbeddingDim(-1); });
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const cards: ReadonlyArray<{
    key: 'enroll' | 'verify' | 'settings';
    titleKey: string;
    descKey: string;
    icon: IconName;
    primary?: boolean;
  }> = [
    { key: 'enroll',  titleKey: 'home.enrollTitle',  descKey: 'home.enrollDesc',  icon: 'plus' },
    { key: 'verify',  titleKey: 'home.verifyTitle',  descKey: 'home.verifyDesc',  icon: 'scan', primary: true },
    { key: 'settings',titleKey: 'settings.title',    descKey: 'home.footer',      icon: 'settings' },
  ];

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={palette.bg} />

      <View style={styles.header}>
        <View style={styles.logoRow}>
          <View style={styles.logoIcon}>
            <Icon name="shield" size={22} color={palette.primaryText} />
          </View>
          <Text style={styles.logo}>{t('app.name')}</Text>
        </View>
        <Text style={styles.tagline}>{t('app.tagline')}</Text>
        <View style={styles.badge}>
          <View style={styles.badgeDot} />
          <Text style={styles.badgeText}>{t('home.offlineBadge')}</Text>
        </View>
      </View>

      <View style={styles.cards}>
        {cards.map((c) => (
          <TouchableOpacity
            key={c.key}
            style={[styles.card, c.primary && styles.cardPrimary]}
            onPress={() => { Haptics.tap(); onNavigate(c.key); }}
            activeOpacity={0.85}
          >
            <View style={[styles.cardIcon, c.primary && styles.cardIconPrimary]}>
              <Icon name={c.icon} size={22} color={c.primary ? palette.primaryText : palette.textTertiary} />
            </View>
            <View style={styles.cardContent}>
              <Text style={[styles.cardTitle, c.primary && styles.cardTitlePrimary]}>{t(c.titleKey)}</Text>
              <Text style={[styles.cardDesc, c.primary && styles.cardDescPrimary]}>{t(c.descKey)}</Text>
            </View>
            <Text style={[styles.cardArrow, c.primary && styles.cardArrowPrimary]}>›</Text>
          </TouchableOpacity>
        ))}

        <View style={styles.statsRow}>
          <StatTile value={embeddingDim > 0 ? String(embeddingDim) : '—'} label={t('home.infoDim')} accent="blue" />
          <StatTile value="7"   label={t('home.infoLayers')} accent="green" />
          <StatTile value="44"  label={t('home.infoStates')} accent="amber" />
        </View>
      </View>

      <View style={styles.footer}>
        <View style={styles.footerDivider} />
        <Text style={styles.footerText}>{t('home.footer')}</Text>
        <Text style={styles.footerVersion}>{t('app.version')}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: palette.bg },
  header: { alignItems: 'center', paddingTop: 60, paddingBottom: spacing.xxl, paddingHorizontal: spacing.xl },
  logoRow: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.md, gap: spacing.md },
  logoIcon: {
    width: 44, height: 44, borderRadius: 14,
    backgroundColor: palette.primarySoft,
    justifyContent: 'center', alignItems: 'center',
    borderWidth: 1, borderColor: palette.primaryBorder,
  },
  logo: { color: palette.textPrimary, ...typeScale.display, fontSize: 30 },
  tagline: { color: palette.textTertiary, ...typeScale.caption, letterSpacing: 0.5 },
  badge: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: palette.successSoft,
    paddingHorizontal: spacing.md, paddingVertical: 6,
    borderRadius: radius.pill, marginTop: spacing.lg,
    borderWidth: 1, borderColor: palette.successBorder,
  },
  badgeDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: palette.success, marginRight: spacing.sm },
  badgeText: { color: palette.success, ...typeScale.micro, fontSize: 12, letterSpacing: 0.5 },
  cards: { flex: 1, paddingHorizontal: spacing.xl, gap: spacing.md, justifyContent: 'center' },
  card: {
    backgroundColor: palette.bgElevated, borderRadius: radius.xl, padding: spacing.xl,
    flexDirection: 'row', alignItems: 'center',
    borderWidth: 1, borderColor: palette.border,
  },
  cardPrimary: { backgroundColor: palette.primarySoft, borderColor: palette.primaryBorder },
  cardIcon: {
    width: 50, height: 50, borderRadius: 16,
    backgroundColor: palette.glass,
    justifyContent: 'center', alignItems: 'center',
    marginRight: spacing.lg,
  },
  cardIconPrimary: { backgroundColor: palette.primarySoft },
  cardContent: { flex: 1 },
  cardTitle: { color: palette.textPrimary, ...typeScale.heading },
  cardTitlePrimary: { color: palette.primaryText },
  cardDesc: { color: palette.textTertiary, fontSize: 13, marginTop: 4 },
  cardDescPrimary: { color: 'rgba(96,165,250,0.6)' },
  cardArrow: { fontSize: 24, color: palette.textDisabled, fontWeight: '300' },
  cardArrowPrimary: { color: 'rgba(59,130,246,0.3)' },
  statsRow: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.md },
  footer: { alignItems: 'center', paddingBottom: 40, paddingHorizontal: spacing.xl },
  footerDivider: { width: 40, height: 3, backgroundColor: palette.borderStrong, borderRadius: 2, marginBottom: spacing.lg },
  footerText: { color: palette.textTertiary, fontSize: 12 },
  footerVersion: { color: palette.textDisabled, fontSize: 11, marginTop: 4 },
});
