/**
 * HomeScreen — main menu with enroll/verify options.
 */

import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export interface HomeScreenProps {
  readonly onNavigate: (screen: 'enroll' | 'verify') => void;
}

export function HomeScreen({ onNavigate }: HomeScreenProps): React.JSX.Element {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.logo}>NetraEdge</Text>
        <Text style={styles.tagline}>Offline Face Recognition</Text>
        <Text style={styles.version}>v0.1.0</Text>
      </View>

      <View style={styles.buttons}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => onNavigate('enroll')}
          activeOpacity={0.7}
        >
          <Text style={styles.buttonIcon}>+</Text>
          <View style={styles.buttonTextContainer}>
            <Text style={styles.buttonTitle}>Enroll Face</Text>
            <Text style={styles.buttonSubtitle}>Register a new user identity</Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.buttonPrimary]}
          onPress={() => onNavigate('verify')}
          activeOpacity={0.7}
        >
          <Text style={[styles.buttonIcon, styles.buttonIconPrimary]}>?</Text>
          <View style={styles.buttonTextContainer}>
            <Text style={[styles.buttonTitle, styles.buttonTitlePrimary]}>
              Verify Identity
            </Text>
            <Text style={[styles.buttonSubtitle, styles.buttonSubtitlePrimary]}>
              Scan and recognize a face
            </Text>
          </View>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>All data stored on-device</Text>
        <Text style={styles.footerText}>No internet required</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  header: {
    alignItems: 'center',
    paddingTop: 80,
    paddingBottom: 40,
  },
  logo: {
    color: '#fff',
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: -1,
  },
  tagline: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 14,
    marginTop: 4,
  },
  version: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 11,
    marginTop: 8,
  },
  buttons: {
    flex: 1,
    paddingHorizontal: 24,
    gap: 16,
    justifyContent: 'center',
  },
  button: {
    backgroundColor: '#1c1c1e',
    borderRadius: 16,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  buttonPrimary: {
    backgroundColor: '#2563eb',
    borderColor: '#3b82f6',
  },
  buttonIcon: {
    fontSize: 24,
    color: 'rgba(255,255,255,0.5)',
    marginRight: 16,
    width: 32,
    textAlign: 'center',
  },
  buttonIconPrimary: {
    color: 'rgba(255,255,255,0.8)',
  },
  buttonTextContainer: {
    flex: 1,
  },
  buttonTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  buttonTitlePrimary: {
    color: '#fff',
  },
  buttonSubtitle: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 13,
    marginTop: 2,
  },
  buttonSubtitlePrimary: {
    color: 'rgba(255,255,255,0.7)',
  },
  footer: {
    alignItems: 'center',
    paddingBottom: 40,
    gap: 4,
  },
  footerText: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 11,
  },
});
