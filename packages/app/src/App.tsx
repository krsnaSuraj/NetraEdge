/**
 * NetraEdge App — Offline Face Recognition + Liveness Detection
 *
 * Main entry point for the React Native application.
 * Integrates camera, face detection, TFLite inference, and sync.
 *
 * Architecture:
 *   AppProvider (init native + pipeline)
 *     → AppNavigator (screen routing)
 *       → HomeScreen | EnrollScreen | VerifyScreen | SettingsScreen
 */

import React, { useState } from 'react';
import { StatusBar, StyleSheet, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AppProvider } from '@netraedge/react-native/src/providers/AppProvider';
import { useAppContext } from '@netraedge/react-native/src/context/AppContext';
import { HomeScreen } from '@netraedge/react-native/src/screens/HomeScreen';
import { EnrollScreen } from '@netraedge/react-native/src/screens/EnrollScreen';
import { VerifyScreen } from '@netraedge/react-native/src/screens/VerifyScreen';
import { SettingsScreen } from '@netraedge/react-native/src/screens/SettingsScreen';

type Screen = 'home' | 'enroll' | 'verify' | 'settings';

function AppNavigator(): React.JSX.Element {
  const [screen, setScreen] = useState<Screen>('home');
  const ctx = useAppContext();

  // Manual sync. Local cache is auto-purged by SyncManager after success.
  // The "purge now" button just forces an extra sync cycle.
  const handleSyncNow = async () => {
    if (!ctx.syncManager) return;
    try {
      await ctx.syncManager.syncNow();
    } catch {
      // Sync errors are emitted via SyncManager events; no-op here.
    }
  };

  switch (screen) {
    case 'enroll':
      return <EnrollScreen onBack={() => setScreen('home')} />;
    case 'verify':
      return <VerifyScreen onBack={() => setScreen('home')} />;
    case 'settings':
      return (
        <SettingsScreen
          onBack={() => setScreen('home')}
          onSyncNow={handleSyncNow}
          onPurge={handleSyncNow}
        />
      );
    default:
      return <HomeScreen onNavigate={setScreen} />;
  }
}

export default function App(): React.JSX.Element {
  return (
    <AppProvider>
      <SafeAreaProvider>
        <StatusBar barStyle="light-content" backgroundColor="#050510" />
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
    backgroundColor: '#050510',
  },
});
