import { NetraEdgeError, ErrorCode } from '../types/Result';

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
      return new NullError(
        `recognitionThreshold must be 0.0–1.0, got ${config.recognitionThreshold}`,
      );
    }
  }

  if (config.livenessMode !== undefined && !VALID_MODES.has(config.livenessMode)) {
    return new NullError(`Invalid livenessMode: ${config.livenessMode}`);
  }

  return null;
}

class NullError extends NetraEdgeError {
  override code = ErrorCode.INVALID_CONFIG;
  override message: string;

  constructor(message: string) {
    super();
    this.message = message;
  }
}
