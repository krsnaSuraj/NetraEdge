import { describe, it, expect } from 'vitest';
import { ErrorCode } from '../types/Result';
import { ok, err } from '../types/Result';
import type { Result } from '../types/Result';

describe('Result pattern', () => {
  it('creates ok result', () => {
    const result = ok(42);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe(42);
    }
  });

  it('creates err result', () => {
    const result = err('something went wrong');
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toBe('something went wrong');
    }
  });

  it('works with division function', () => {
    function divide(a: number, b: number): Result<number, string> {
      if (b === 0) return err('Division by zero');
      return ok(a / b);
    }

    expect(divide(10, 2).ok).toBe(true);
    expect(divide(10, 0).ok).toBe(false);
  });

  it('ErrorCode has expected values', () => {
    expect(ErrorCode.FACE_NOT_FOUND).toBe('FACE_NOT_FOUND');
    expect(ErrorCode.LIVENESS_FAILED).toBe('LIVENESS_FAILED');
    expect(ErrorCode.MODEL_NOT_LOADED).toBe('MODEL_NOT_LOADED');
  });
});
