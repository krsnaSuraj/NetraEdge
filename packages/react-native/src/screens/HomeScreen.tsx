/**
 * HomeScreen — premium main menu with enroll/verify options.
 */

import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View, StatusBar } from 'react-native';

export interface HomeScreenProps {
  readonly onNavigate: (screen: 'enroll' | 'verify') => void;
}

export function HomeScreen({ onNavigate }: HomeScreenProps): React.JSX.Element {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#050510" />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.logoContainer}>
          <View style={styles.logoIcon}>
            <Text style={styles.logoEmoji}>◉</Text>
          </View>
          <Text style={styles.logo}>NetraEdge</Text>
        </View>
        <Text style={styles.tagline}>Secure Offline Face Recognition</Text>
        <View style={styles.badge}>
          <View style={styles.badgeDot} />
          <Text style={styles.badgeText}>100% Offline</Text>
        </View>
      </View>

      {/* Cards */}
      <View style={styles.cards}>
        {/* Enroll Card */}
        <TouchableOpacity
          style={styles.card}
          onPress={() => onNavigate('enroll')}
          activeOpacity={0.8}
        >
          <View style={styles.cardIconContainer}>
            <Text style={styles.cardIcon}>＋</Text>
          </View>
          <View style={styles.cardContent}>
            <Text style={styles.cardTitle}>Enroll New Face</Text>
            <Text style={styles.cardDesc}>Register a user with face biometrics</Text>
          </View>
          <Text style={styles.cardArrow}>›</Text>
        </TouchableOpacity>

        {/* Verify Card */}
        <TouchableOpacity
          style={[styles.card, styles.cardPrimary]}
          onPress={() => onNavigate('verify')}
          activeOpacity={0.8}
        >
          <View style={[styles.cardIconContainer, styles.cardIconPrimary]}>
            <Text style={[styles.cardIcon, styles.cardIconTextPrimary]}>⟳</Text>
          </View>
          <View style={styles.cardContent}>
            <Text style={[styles.cardTitle, styles.cardTitlePrimary]}>Verify Identity</Text>
            <Text style={[styles.cardDesc, styles.cardDescPrimary]}>
              Scan face with liveness check
            </Text>
          </View>
          <Text style={[styles.cardArrow, styles.cardArrowPrimary]}>›</Text>
        </TouchableOpacity>

        {/* Info Cards */}
        <View style={styles.infoRow}>
          <View style={styles.infoCard}>
            <Text style={styles.infoNumber}>128</Text>
            <Text style={styles.infoLabel}>Dimensions</Text>
          </View>
          <View style={styles.infoCard}>
            <Text style={styles.infoNumber}>3</Text>
            <Text style={styles.infoLabel}>Liveness Checks</Text>
          </View>
          <View style={styles.infoCard}>
            <Text style={styles.infoNumber}>&lt;1s</Text>
            <Text style={styles.infoLabel}>Inference</Text>
          </View>
        </View>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <View style={styles.footerDivider} />
        <Text style={styles.footerText}>All biometric data stored on-device only</Text>
        <Text style={styles.footerVersion}>v1.0.0 • NHAI Hackathon 7.0</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#050510',
  },
  header: {
    alignItems: 'center',
    paddingTop: 60,
    paddingBottom: 32,
    paddingHorizontal: 24,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  logoIcon: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  logoEmoji: {
    fontSize: 22,
    color: '#3b82f6',
  },
  logo: {
    color: '#fff',
    fontSize: 30,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  tagline: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 14,
    letterSpacing: 0.5,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginTop: 16,
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.2)',
  },
  badgeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#22c55e',
    marginRight: 8,
  },
  badgeText: {
    color: '#22c55e',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  cards: {
    flex: 1,
    paddingHorizontal: 20,
    gap: 14,
    justifyContent: 'center',
  },
  card: {
    backgroundColor: '#0f0f1a',
    borderRadius: 20,
    padding: 22,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
  },
  cardPrimary: {
    backgroundColor: 'rgba(59, 130, 246, 0.08)',
    borderColor: 'rgba(59, 130, 246, 0.2)',
  },
  cardIconContainer: {
    width: 50,
    height: 50,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.05)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  cardIconPrimary: {
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
  },
  cardIcon: {
    fontSize: 22,
    color: 'rgba(255,255,255,0.5)',
  },
  cardIconTextPrimary: {
    color: '#3b82f6',
  },
  cardContent: {
    flex: 1,
  },
  cardTitle: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
    letterSpacing: -0.3,
  },
  cardTitlePrimary: {
    color: '#60a5fa',
  },
  cardDesc: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 13,
    marginTop: 4,
  },
  cardDescPrimary: {
    color: 'rgba(96, 165, 250, 0.5)',
  },
  cardArrow: {
    fontSize: 24,
    color: 'rgba(255,255,255,0.15)',
    fontWeight: '300',
  },
  cardArrowPrimary: {
    color: 'rgba(59, 130, 246, 0.3)',
  },
  infoRow: {
    flexDirection: 'row',
    gap: 10,
  },
  infoCard: {
    flex: 1,
    backgroundColor: '#0f0f1a',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.04)',
  },
  infoNumber: {
    color: '#3b82f6',
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  infoLabel: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 11,
    marginTop: 4,
    textAlign: 'center',
  },
  footer: {
    alignItems: 'center',
    paddingBottom: 40,
    paddingHorizontal: 24,
  },
  footerDivider: {
    width: 40,
    height: 3,
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 2,
    marginBottom: 16,
  },
  footerText: {
    color: 'rgba(255,255,255,0.2)',
    fontSize: 12,
  },
  footerVersion: {
    color: 'rgba(255,255,255,0.12)',
    fontSize: 11,
    marginTop: 4,
  },
});
