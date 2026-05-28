/**
 * @netraedge/react-native
 *
 * React Native bridge for NetraEdge face recognition.
 * Provides hooks and components for camera-based face detection,
 * recognition, and liveness verification.
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

// Re-export core types
export {
  type FaceDetection,
  type LivenessResult,
  LivenessVerdict,
  type Result,
  type EnrollmentResult,
  type VerificationResult as CoreVerificationResult,
} from '@netraedge/core';
