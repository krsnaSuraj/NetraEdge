/**
 * @netraedge/react-native
 *
 * React Native bridge for NetraEdge face recognition.
 * Provides hooks, components, storage, and sync for camera-based
 * face detection, recognition, and liveness verification.
 *
 * @example
 * import { FaceCamera, useFaceDetection, useLivenessCheck } from '@netraedge/react-native';
 */

// Hooks
export { useFaceDetection } from './hooks/useFaceDetection';
export { useFaceRecognition, type RecognitionState, type VerificationResult } from './hooks/useFaceRecognition';
export { useLivenessCheck, type LivenessState } from './hooks/useLivenessCheck';

// Components
export { FaceCamera, type FaceCameraProps } from './components/FaceCamera';
export { LivenessPrompt, type LivenessPromptProps } from './components/LivenessPrompt';
export { ResultCard, type ResultCardProps } from './components/ResultCard';

// Screens
export { HomeScreen } from './screens/HomeScreen';
export { VerifyScreen } from './screens/VerifyScreen';

// Storage
export { SQLiteEmbeddingStore, type SQLiteDatabase } from './storage/SQLiteEmbeddingStore';
export { SQLiteSyncQueueStore } from './storage/SQLiteSyncQueueStore';

// Sync
export { ReactNativeNetworkMonitor } from './sync/ReactNativeNetworkMonitor';
export { AWSSyncTransport, type AWSTransportConfig } from './sync/AWSSyncTransport';

// Context
export { AppContext, useAppContext, type AppContextValue } from './context/AppContext';

// Providers
export { AppProvider } from './providers/AppProvider';

// Native
export { NetraEdgeNative, type NetraEdgeNativeModule } from './native/NetraEdgeNative';

// Re-export core types
export {
  type FaceDetection,
  type LivenessResult,
  LivenessVerdict,
  type Result,
  type Point3D,
  type EnrollmentResult,
  type VerificationResult as CoreVerificationResult,
  FacePipeline,
  LivenessOrchestrator,
  TFLiteEncoder,
  StubEncoder,
  CNNTextureAnalyzer,
  StubTextureAnalyzer,
  InMemoryEmbeddingStore,
  InMemorySyncQueue,
  DefaultSyncManager,
  DataPurgeManager,
  type SyncManager,
  type EmbeddingStore,
  type Encoder,
  type TextureAnalyzer,
  type SyncQueue,
  type SyncTransport,
  type NetworkMonitor,
} from '@netraedge/core';
