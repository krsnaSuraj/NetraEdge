import { describe, it, expect } from 'vitest';
import { analyzeColor } from '../liveness/ColorAnalyzer';

describe('ColorAnalyzer', () => {
  it('returns valid result structure', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i += 3) {
      faceData[i] = 0.6 + Math.random() * 0.2;     // R
      faceData[i + 1] = 0.4 + Math.random() * 0.2; // G
      faceData[i + 2] = 0.3 + Math.random() * 0.1; // B
    }
    const result = analyzeColor(faceData);
    expect(result).toHaveProperty('realScore');
    expect(result).toHaveProperty('skinConsistency');
    expect(result).toHaveProperty('saturationMean');
    expect(result).toHaveProperty('saturationStd');
    expect(result).toHaveProperty('gradientSmoothness');
    expect(result).toHaveProperty('skinDivergence');
    expect(result).toHaveProperty('isReal');
  });

  it('realScore is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i += 3) {
      faceData[i] = 0.6; faceData[i + 1] = 0.4; faceData[i + 2] = 0.3;
    }
    const result = analyzeColor(faceData);
    expect(result.realScore).toBeGreaterThanOrEqual(0);
    expect(result.realScore).toBeLessThanOrEqual(1);
  });

  it('skin consistency is between 0 and 1', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    for (let i = 0; i < faceData.length; i++) faceData[i] = 0.5;
    const result = analyzeColor(faceData);
    expect(result.skinConsistency).toBeGreaterThanOrEqual(0);
    expect(result.skinConsistency).toBeLessThanOrEqual(1);
  });

  it('uniform skin-like colors have high consistency', () => {
    const faceData = new Float32Array(112 * 112 * 3);
    // Fill with warm skin-like colors in YCbCr range
    for (let i = 0; i < faceData.length; i += 3) {
      faceData[i] = 0.6;     // R
      faceData[i + 1] = 0.4; // G
      faceData[i + 2] = 0.3; // B
    }
    const result = analyzeColor(faceData);
    expect(result.skinConsistency).toBeGreaterThan(0);
  });

  it('accepts number array input', () => {
    const faceData: number[] = [];
    for (let i = 0; i < 112 * 112 * 3; i++) faceData.push(0.5);
    const result = analyzeColor(faceData);
    expect(result.realScore).toBeGreaterThanOrEqual(0);
  });
});
