/**
 * Result type — functional error handling without exceptions.
 *
 * Use `Result<T, E>` instead of throwing errors. This forces
 * callers to handle both success and failure cases explicitly.
 *
 * @example
 * function divide(a: number, b: number): Result<number, string> {
 *   if (b === 0) return { ok: false, error: 'Division by zero' };
 *   return { ok: true, value: a / b };
 * }
 */

export type Result<T, E = NetraEdgeError> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

export function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

/**
 * Unwrap a Result or throw if it's an error.
 * Use only at the boundary — internal code should pattern-match.
 */
export function unwrap<T>(result: Result<T, NetraEdgeError>): T {
  if (result.ok) return result.value;
  throw new Error(result.error.message);
}

/**
 * Base error class for all NetraEdge errors.
 * Provides structured error codes for programmatic handling.
 */
export abstract class NetraEdgeError {
  abstract readonly code: ErrorCode;
  abstract readonly message: string;
  readonly timestamp: number;
  readonly context?: Record<string, unknown>;

  constructor(context?: Record<string, unknown>) {
    this.timestamp = Date.now();
    this.context = context;
  }
}

export enum ErrorCode {
  FACE_NOT_FOUND = 'FACE_NOT_FOUND',
  MULTIPLE_FACES = 'MULTIPLE_FACES',
  FACE_TOO_SMALL = 'FACE_TOO_LOW_CONFIDENCE',
  QUALITY_TOO_LOW = 'QUALITY_TOO_LOW',
  LIVENESS_FAILED = 'LIVENESS_FAILED',
  MODEL_NOT_LOADED = 'MODEL_NOT_LOADED',
  MODEL_INFERENCE_FAILED = 'MODEL_INFERENCE_FAILED',
  ENROLLMENT_FAILED = 'ENROLLMENT_FAILED',
  VERIFICATION_FAILED = 'VERIFICATION_FAILED',
  STORAGE_ERROR = 'STORAGE_ERROR',
  SYNC_FAILED = 'SYNC_FAILED',
  NETWORK_ERROR = 'NETWORK_ERROR',
  INVALID_CONFIG = 'INVALID_CONFIG',
}
