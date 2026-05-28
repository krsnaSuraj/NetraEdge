/**
 * NetraEdge App — Offline Face Recognition + Liveness Detection
 *
 * Main entry point for the React Native application.
 * Integrates camera, face detection, TFLite inference, and sync.
 *
 * Architecture:
 *   AppProvider (init native + pipeline)
 *     → AppNavigator (screen routing)
 *       → HomeScreen | EnrollScreen | VerifyScreen
 */

import React, { useState } from 'react';
import { StatusBar, StyleSheet, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AppProvider } from '@netraedge/react-native/src/providers/AppProvider';
import { HomeScreen } from '@netraedge/react-native/src/screens/HomeScreen';
import { EnrollScreen } from '@netraedge/react-native/src/screens/EnrollScreen';
import { VerifyScreen } from '@netraedge/react-native/src/screens/VerifyScreen';

type Screen = 'home' | 'enroll' | 'verify';

function AppNavigator(): React.JSX.Element {
  const [screen, setScreen] = useState<Screen>('home');

  switch (screen) {
    case 'enroll':
      return <EnrollScreen onBack={() => setScreen('home')} />;
    case 'verify':
      return <VerifyScreen onBack={() => setScreen('home')} />;
    default:
      return <HomeScreen onNavigate={setScreen} />;
  }
}

export default function App(): React.JSX.Element {
  return (
    <AppProvider>
      <SafeAreaProvider>
        <StatusBar barStyle="light-content" backgroundColor="#0a0a0a" />
        <View style={styles.container}>
          <AppNavigator />
        </View>
      </SafeAreaProvider>
    </AppProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
