/**
 * AppProvider — initializes native modules and provides pipeline context.
 *
 * On mount:
 * 1. Loads TFLite models via native module (falls back to StubEncoder if missing)
 * 2. Creates encoder, store, and liveness orchestrator
 * 3. Initializes the face pipeline
 * 4. Sets up sync manager with queue
 * 5. Provides everything via AppContext
 *
 * Shows a loading screen while initializing.
 * If models are missing, runs in demo mode with stubs.
 */

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  TFLiteEncoder,
  StubEncoder,
  CNNTextureAnalyzer,
  StubTextureAnalyzer,
  LivenessOrchestrator,
  FacePipeline,
  InMemoryEmbeddingStore,
  DefaultSyncManager,
  InMemorySyncQueue,
  DataPurgeManager,
} from '@netraedge/core';
import type { Encoder, TextureAnalyzer, SyncManager } from '@netraedge/core';
import { AppContext, type AppContextValue } from '../context/AppContext';
import { NetraEdgeNative } from '../native/NetraEdgeNative';

interface AppProviderProps {
  readonly children: React.ReactNode;
}

type InitState =
  | { phase: 'loading' }
  | { phase: 'ready'; context: AppContextValue; demoMode: boolean }
  | { phase: 'error'; message: string };

export function AppProvider({ children }: AppProviderProps): React.JSX.Element {
  const [state, setState] = useState<InitState>({ phase: 'loading' });

  useEffect(() => {
    let disposed = false;

    async function initialize() {
      try {
        let encoder: Encoder;
        let textureAnalyzer: TextureAnalyzer;
        let demoMode = false;

        try {
          const initialized = await NetraEdgeNative.initialize();
          if (initialized) {
            encoder = new TFLiteEncoder(NetraEdgeNative);
            textureAnalyzer = new CNNTextureAnalyzer(NetraEdgeNative);
          } else {
            throw new Error('Models not loaded');
          }
        } catch {
          encoder = new StubEncoder();
          textureAnalyzer = new StubTextureAnalyzer();
          demoMode = true;
        }

        const store = new InMemoryEmbeddingStore();
        const liveness = new LivenessOrchestrator(textureAnalyzer);
        const pipeline = new FacePipeline(encoder, store, liveness);

        const queue = new InMemorySyncQueue();
        const purge = new DataPurgeManager(store, queue);
        const transport = {
          async uploadBatch() { return true; },
          async uploadEnrollment() { return true; },
        };
        const network = {
          async isOnline() { return false; },
          onConnectivityChange() { return () => {}; },
        };
        const syncManager: SyncManager = new DefaultSyncManager(
          transport, network, queue, purge,
        );

        if (!disposed) {
          setState({
            phase: 'ready',
            context: { pipeline, syncManager },
            demoMode,
          });
        }
      } catch (e) {
        if (!disposed) {
          setState({
            phase: 'error',
            message: `Initialization failed: ${String(e)}`,
          });
        }
      }
    }

    void initialize();

    return () => { disposed = true; };
  }, []);

  if (state.phase === 'loading') {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#3b82f6" />
        <Text style={styles.loadingText}>Initializing NetraEdge...</Text>
        <Text style={styles.loadingSub}>Loading TFLite models</Text>
      </View>
    );
  }

  if (state.phase === 'error') {
    return (
      <View style={styles.center}>
        <Text style={styles.errorTitle}>Initialization Error</Text>
        <Text style={styles.errorText}>{state.message}</Text>
      </View>
    );
  }

  return (
    <AppContext.Provider value={state.context}>
      {state.demoMode && (
        <View style={styles.demoBanner}>
          <Text style={styles.demoText}>
            Demo Mode — Train models for production accuracy
          </Text>
        </View>
      )}
      {children}
    </AppContext.Provider>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0a0a0a',
    paddingHorizontal: 32,
  },
  loadingText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 20,
  },
  loadingSub: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 13,
    marginTop: 8,
  },
  errorTitle: {
    color: '#ef4444',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 12,
  },
  errorText: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  demoBanner: {
    backgroundColor: 'rgba(234, 179, 8, 0.9)',
    paddingVertical: 6,
    paddingHorizontal: 12,
    alignItems: 'center',
  },
  demoText: {
    color: '#000',
    fontSize: 11,
    fontWeight: '600',
  },
});
