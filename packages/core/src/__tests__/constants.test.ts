import { describe, it, expect } from 'vitest';
import { RECOGNITION_THRESHOLDS, LIVENESS_THRESHOLDS, MODEL_CONFIG } from '../config/constants';

describe('Configuration Constants', () => {
  describe('RECOGNITION_THRESHOLDS', () => {
    it('has valid match confidence range', () => {
      expect(RECOGNITION_THRESHOLDS.matchConfidence).toBeGreaterThan(0);
      expect(RECOGNITION_THRESHOLDS.matchConfidence).toBeLessThan(1);
    });

    it('has positive min face size', () => {
      expect(RECOGNITION_THRESHOLDS.minFaceSize).toBeGreaterThan(0);
    });

    it('has valid roll angle limit', () => {
      expect(RECOGNITION_THRESHOLDS.maxRollAngle).toBeGreaterThan(0);
      expect(RECOGNITION_THRESHOLDS.maxRollAngle).toBeLessThan(90);
    });
  });

  describe('LIVENESS_THRESHOLDS', () => {
    it('has valid EAR threshold', () => {
      expect(LIVENESS_THRESHOLDS.blinkEARThreshold).toBeGreaterThan(0);
      expect(LIVENESS_THRESHOLDS.blinkEARThreshold).toBeLessThan(0.5);
    });

    it('has positive blink min duration', () => {
      expect(LIVENESS_THRESHOLDS.blinkMinDuration).toBeGreaterThan(0);
    });

    it('has valid texture confidence', () => {
      expect(LIVENESS_THRESHOLDS.textureConfidence).toBeGreaterThan(0);
      expect(LIVENESS_THRESHOLDS.textureConfidence).toBeLessThan(1);
    });

    it('has valid depth variance threshold', () => {
      expect(LIVENESS_THRESHOLDS.depthVariance).toBeGreaterThan(0);
    });
  });

  describe('MODEL_CONFIG', () => {
    it('has correct input dimensions for recognition', () => {
      expect(MODEL_CONFIG.recognition.inputWidth).toBe(112);
      expect(MODEL_CONFIG.recognition.inputHeight).toBe(112);
    });

    it('has correct embedding dimension', () => {
      expect(MODEL_CONFIG.recognition.embeddingDimension).toBe(128);
    });

    it('has correct liveness classes', () => {
      expect(MODEL_CONFIG.liveness.numClasses).toBe(3);
    });
  });
});
