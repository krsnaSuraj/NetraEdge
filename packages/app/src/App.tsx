import React, { useCallback, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

type Screen = 'home' | 'enroll' | 'verify';

export default function App(): React.JSX.Element {
  const [screen, setScreen] = useState<Screen>('home');

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>NetraEdge</Text>
        <Text style={styles.subtitle}>Offline Face Recognition</Text>
      </View>

      {screen === 'home' && (
        <View style={styles.menu}>
          <TouchableOpacity
            style={styles.menuButton}
            onPress={() => setScreen('enroll')}
          >
            <Text style={styles.menuButtonText}>Enroll New Face</Text>
            <Text style={styles.menuButtonSubtext}>Register a new user</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.menuButton, styles.menuButtonPrimary]}
            onPress={() => setScreen('verify')}
          >
            <Text style={[styles.menuButtonText, styles.menuButtonTextPrimary]}>
              Verify Identity
            </Text>
            <Text style={[styles.menuButtonSubtext, styles.menuButtonSubtextPrimary]}>
              Scan and recognize
            </Text>
          </TouchableOpacity>
        </View>
      )}

      {screen === 'enroll' && (
        <EnrollScreen onBack={() => setScreen('home')} />
      )}

      {screen === 'verify' && (
        <VerifyScreen onBack={() => setScreen('home')} />
      )}

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          v0.1.0 | Offline Mode
        </Text>
      </View>
    </SafeAreaView>
  );
}

function EnrollScreen({ onBack }: { onBack: () => void }): React.JSX.Element {
  const [userId, setUserId] = useState('');

  return (
    <View style={styles.screen}>
      <Text style={styles.screenTitle}>Enroll New Face</Text>
      <Text style={styles.screenDescription}>
        Position your face in the camera and hold still for 2 seconds.
      </Text>
      <TouchableOpacity style={styles.backButton} onPress={onBack}>
        <Text style={styles.backButtonText}>Back</Text>
      </TouchableOpacity>
    </View>
  );
}

function VerifyScreen({ onBack }: { onBack: () => void }): React.JSX.Element {
  return (
    <View style={styles.screen}>
      <Text style={styles.screenTitle}>Verify Identity</Text>
      <Text style={styles.screenDescription}>
        Look at the camera and blink when prompted.
      </Text>
      <TouchableOpacity style={styles.backButton} onPress={onBack}>
        <Text style={styles.backButtonText}>Back</Text>
      </TouchableOpacity>
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
    paddingVertical: 32,
  },
  title: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  subtitle: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 14,
    marginTop: 4,
  },
  menu: {
    flex: 1,
    paddingHorizontal: 24,
    gap: 16,
  },
  menuButton: {
    backgroundColor: '#1c1c1e',
    borderRadius: 16,
    padding: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  menuButtonPrimary: {
    backgroundColor: '#2563eb',
    borderColor: '#3b82f6',
  },
  menuButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  menuButtonTextPrimary: {
    color: '#fff',
  },
  menuButtonSubtext: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 13,
    marginTop: 4,
  },
  menuButtonSubtextPrimary: {
    color: 'rgba(255,255,255,0.7)',
  },
  screen: {
    flex: 1,
    paddingHorizontal: 24,
  },
  screenTitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 8,
  },
  screenDescription: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    lineHeight: 20,
  },
  backButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  backButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  footerText: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 11,
  },
});
