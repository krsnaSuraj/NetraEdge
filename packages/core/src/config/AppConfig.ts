import type { NetraEdgeError } from '../types/Result';
import { ErrorCode } from '../types/Result';

export interface AppConfig {
  readonly modelPath: string;
  readonly enableLiveness: boolean;
  readonly livenessMode: LivenessMode;
  readonly recognitionThreshold: number;
  readonly syncEnabled: boolean;
  readonly awsEndpoint?: string;
}

export enum LivenessMode {
  BLINK_ONLY = 'blink_only',
  BLINK_AND_TEXTURE = 'blink_and_texture',
  MULTI_MODAL = 'multi_modal',
}

const VALID_MODES = new Set(Object.values(LivenessMode));

export function validateConfig(config: Partial<AppConfig>): NetraEdgeError | null {
  if (config.recognitionThreshold !== undefined) {
    if (config.recognitionThreshold < 0 || config.recognitionThreshold > 1) {
      return Object.assign(
        Object.create(NullError.prototype) as NullError,
        {
          code: ErrorCode.INVALID_CONFIG,
          message: `recognitionThreshold must be 0.0–1.0, got ${config.recognitionThreshold}`,
          timestamp: Date.now(),
        },
      );
    }
  }

  if (config.livenessMode !== undefined && !VALID_MODES.has(config.livenessMode)) {
    return Object.assign(
      Object.create(NullError.prototype) as NullError,
      {
        code: ErrorCode.INVALID_CONFIG,
        message: `Invalid livenessMode: ${config.livenessMode}`,
        timestamp: Date.now(),
      },
    );
  }

  return null;
}

class NullError extends NetraEdgeError {
  code = ErrorCode.INVALID_CONFIG;
  message = '';
  timestamp = Date.now();
}
