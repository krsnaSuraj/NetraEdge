/**
 * SettingsScreen — language switcher + sync status + about.
 * Uses design tokens + Icons + Haptics + i18n.
 */

import React, { useEffect, useState } from 'react';
import {
  Alert,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { t, getLocale, setLocale, subscribeLocale, SUPPORTED_LOCALES, type LocaleCode } from '../i18n';
import { Icon, Haptics, Section, ActionRow, ListRow } from '../components';
import { palette, spacing, type as typeScale } from '../theme';

export interface SettingsScreenProps {
  readonly onBack: () => void;
  readonly onSyncNow?: () => Promise<void> | void;
  readonly onPurge?: () => Promise<void> | void;
}

export function SettingsScreen({
  onBack,
  onSyncNow,
  onPurge,
}: SettingsScreenProps): React.JSX.Element {
  const [locale, setLocaleState] = useState<LocaleCode>(getLocale());
  const [embeddingDim, setEmbeddingDim] = useState<number>(-1);

  useEffect(() => {
    return subscribeLocale((l) => setLocaleState(l));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const netraEdge = require('../native/NetraEdgeNative').NetraEdgeNative;
    const refresh = () => {
      if (cancelled) return;
      netraEdge.getEmbeddingDim()
        .then((d: number) => { if (!cancelled) setEmbeddingDim(d); })
        .catch(() => { if (!cancelled) setEmbeddingDim(-1); });
    };
    refresh();
    const id = setInterval(refresh, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const handleLanguageChange = (code: LocaleCode) => {
    Haptics.selection();
    setLocale(code);
    setLocaleState(code);
  };

  const handleSync = async () => {
    if (!onSyncNow) {
      Alert.alert(t('settings.sync'), 'Sync manager not initialized');
      return;
    }
    Haptics.tap();
    await onSyncNow();
  };

  const handlePurge = () => {
    if (!onPurge) return;
    Haptics.warn();
    Alert.alert(
      t('settings.purge'),
      t('settings.purgeConfirm'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.confirm'),
          style: 'destructive',
          onPress: () => { Haptics.tap(); void onPurge(); },
        },
      ],
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={palette.bg} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => { Haptics.tap(); onBack(); }} style={styles.backButton} activeOpacity={0.7}>
          <Icon name="arrow-left" size={22} color={palette.textPrimary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('settings.title')}</Text>
        <View style={styles.backButton} />
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>

        {/* Language */}
        <Section label={t('settings.language')}>
          {SUPPORTED_LOCALES.map((l, idx) => (
            <TouchableOpacity
              key={l.code}
              style={[styles.langRow, idx < SUPPORTED_LOCALES.length - 1 && styles.divider]}
              onPress={() => handleLanguageChange(l.code)}
              activeOpacity={0.7}
            >
              <Text style={styles.langText}>{l.label}</Text>
              <View style={[styles.radio, locale === l.code && styles.radioOn]}>
                {locale === l.code && <View style={styles.radioDot} />}
              </View>
            </TouchableOpacity>
          ))}
        </Section>

        {/* Sync */}
        <Section label={t('settings.sync')}>
          <ActionRow
            label={t('settings.syncNow')}
            icon="cloud"
            trailingText={t('settings.syncIdle')}
            showChevron={false}
            onPress={handleSync}
          />
        </Section>

        {/* Purge */}
        <Section label={t('settings.purge')}>
          <ActionRow
            label={t('settings.purgeNow')}
            icon="trash"
            danger
            showChevron={false}
            onPress={handlePurge}
          />
        </Section>

        {/* About */}
        <Section label={t('settings.about')}>
          <ListRow label={t('settings.version')}  meta="0.1.0" divider />
          <ListRow label={t('settings.model')}    meta={t('settings.modelVersion')} divider />
          <ListRow label={t('settings.liveness')} meta={t('settings.livenessVersion')} divider />
          <ListRow label="Embedding dim" meta={embeddingDim > 0 ? `${embeddingDim}-d` : '—'} divider />
          <ListRow label={t('settings.buildInfo')} meta="dev" />
        </Section>

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: palette.bg },
  header: {
    flexDirection: 'row', alignItems: 'center',
    paddingTop: 60, paddingBottom: spacing.xl, paddingHorizontal: spacing.xl,
    justifyContent: 'space-between',
  },
  backButton: { width: 36, height: 36, justifyContent: 'center', alignItems: 'center' },
  headerTitle: { color: palette.textPrimary, ...typeScale.heading, fontSize: 18 },
  scroll: { paddingHorizontal: spacing.xl, paddingBottom: 40 },
  langRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    padding: spacing.lg,
  },
  divider: { borderBottomWidth: 1, borderBottomColor: palette.border },
  langText: { color: palette.textPrimary, ...typeScale.body },
  radio: {
    width: 22, height: 22, borderRadius: 11,
    borderWidth: 2, borderColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center', alignItems: 'center',
  },
  radioOn: { borderColor: palette.primary },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: palette.primary },
});
