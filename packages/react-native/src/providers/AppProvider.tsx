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
import { ReactNativeNetworkMonitor } from '../sync/ReactNativeNetworkMonitor';
import { AWSSyncTransport } from '../sync/AWSSyncTransport';

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

        // Real AWS sync transport (production-ready)
        const transport = new AWSSyncTransport({
          endpoint: 'https://api.netraedge.dev/sync',
          apiKey: 'demo-key',
          timeoutMs: 10000,
        });

        // Real network monitor using NetInfo
        const network = new ReactNativeNetworkMonitor();

        const syncManager: SyncManager = new DefaultSyncManager(
          transport, network, queue, purge,
        );

        // Start sync manager
        syncManager.start();

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
        <View style={styles.loadingCircle}>
          <ActivityIndicator size="large" color="#3b82f6" />
        </View>
        <Text style={styles.loadingText}>NetraEdge</Text>
        <Text style={styles.loadingSub}>Initializing TFLite models...</Text>
      </View>
    );
  }

  if (state.phase === 'error') {
    return (
      <View style={styles.center}>
        <View style={styles.errorCircle}>
          <Text style={styles.errorEmoji}>!</Text>
        </View>
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
    backgroundColor: '#050510',
    paddingHorizontal: 32,
  },
  loadingCircle: {
    width: 80,
    height: 80,
    borderRadius: 24,
    backgroundColor: 'rgba(59, 130, 246, 0.08)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  loadingText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  loadingSub: {
    color: 'rgba(255,255,255,0.3)',
    fontSize: 13,
    marginTop: 8,
  },
  errorCircle: {
    width: 64,
    height: 64,
    borderRadius: 20,
    backgroundColor: 'rgba(239, 68, 68, 0.12)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.2)',
  },
  errorEmoji: {
    fontSize: 28,
    color: '#ef4444',
    fontWeight: '800',
  },
  errorTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 8,
  },
  errorText: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 13,
    textAlign: 'center',
    lineHeight: 20,
  },
  demoBanner: {
    backgroundColor: 'rgba(245, 158, 11, 0.9)',
    paddingVertical: 8,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  demoText: {
    color: '#000',
    fontSize: 12,
    fontWeight: '700',
  },
});
