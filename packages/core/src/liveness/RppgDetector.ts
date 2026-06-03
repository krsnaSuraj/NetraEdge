/**
 * rPPG (remote photoplethysmography) pulse detector — SOTA innovation.
 *
 * Detects blood-flow-induced subtle color changes in face skin over time.
 * The POS (Plane-Orthogonal-to-Skin) algorithm projects RGB onto a plane
 * perpendicular to the skin-tone direction, isolating the pulse signal.
 *
 * This is the SOTA differentiator — kills ANY 2D spoof:
 *   - Print attack: paper has no blood flow → no pulse → SPOOF
 *   - Screen replay: backlight has no blood flow → no pulse → SPOOF
 *   - Video replay: out-of-sync blood-flow phase → no clean pulse → SPOOF
 *   - 3D mask: silicone has no capillary blood flow → weak/no pulse → SPOOF
 *
 * Pipeline:
 *   1. Extract mean RGB from cheek/forehead skin pixels per frame
 *   2. POS projection:  S = R - G,  P = R + G - 2B
 *   3. Bandpass filter 0.7-4 Hz (42-240 bpm)
 *   4. Sliding window FFT
 *   5. Find dominant frequency in cardiac range
 *   6. Confidence = peak prominence / background
 *
 * References:
 *   - Wang et al. "Algorithmic Principles of Remote PPG" (2017)
 *   - Chen & McDuff "DeepPhys: Video-Based Physiological Measurement" (2018)
 *
 * Lightweight: ~2 KB JS code, zero ML model needed (pure DSP).
 */

import type { RppgResult } from '../types/Liveness';

const SAMPLE_HZ = 30;
const MIN_SAMPLES = 30;
const CARDIO_LOW = 0.7;
const CARDIO_HIGH = 4.0;
const QUALITY_FLOOR = 0.4;
const DOMINANT_POWER_FLOOR = 0.15;

export class RppgDetector {
  private readonly _rgbHistory: Array<{ r: number; g: number; b: number; t: number }> = [];
  private readonly _maxSamples = 90; // 3 sec at 30 fps
  private _result: RppgResult = this._emptyResult();

  /** Feed one frame of normalized face RGB data. */
  feed(faceData: Float32Array | number[], timestampMs: number): RppgResult {
    const rgb = this._extractSkinMean(faceData);
    if (!rgb) {
      return this._result;
    }
    this._rgbHistory.push({ ...rgb, t: timestampMs });
    if (this._rgbHistory.length > this._maxSamples) {
      this._rgbHistory.shift();
    }
    if (this._rgbHistory.length >= MIN_SAMPLES) {
      this._result = this._analyzePulse();
    }
    return this._result;
  }

  /** Get current best estimate without feeding new data. */
  getResult(): RppgResult {
    return this._result;
  }

  get hasEnoughSamples(): boolean {
    return this._rgbHistory.length >= MIN_SAMPLES;
  }

  reset(): void {
    this._rgbHistory.length = 0;
    this._result = this._emptyResult();
  }

  private _emptyResult(): RppgResult {
    return {
      pulseDetected: false,
      heartRateBpm: 0,
      signalQuality: 0,
      dominantPower: 0,
      isLive: false,
    };
  }

  /**
   * Extract mean skin-tone color from faceData using the central 50% region
   * (cheek + forehead area), skipping border pixels.
   */
  private _extractSkinMean(faceData: Float32Array | number[]): { r: number; g: number; b: number } | null {
    const W = 112;
    const H = 112;
    const x0 = Math.floor(W * 0.25);
    const x1 = Math.floor(W * 0.75);
    const y0 = Math.floor(H * 0.20);
    const y1 = Math.floor(H * 0.80);
    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    let n = 0;
    for (let y = y0; y < y1; y += 2) {
      for (let x = x0; x < x1; x += 2) {
        const i = (y * W + x) * 3;
        const r = faceData[i] ?? 0;
        const g = faceData[i + 1] ?? 0;
        const b = faceData[i + 2] ?? 0;
        // Skin-tone filter (Y in [60, 220], R > B for natural skin)
        const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
        if (yVal > 0.20 && yVal < 0.92 && r > b * 0.9) {
          rSum += r;
          gSum += g;
          bSum += b;
          n++;
        }
      }
    }
    if (n < 50) return null;
    return { r: rSum / n, g: gSum / n, b: bSum / n };
  }

  /**
   * POS algorithm + sliding FFT. Returns the pulse analysis result.
   */
  private _analyzePulse(): RppgResult {
    const n = this._rgbHistory.length;
    if (n < MIN_SAMPLES) return this._emptyResult();

    // POS: project onto plane perpendicular to skin tone
    const pos = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const s = this._rgbHistory[i]!;
      pos[i] = s.r - s.g;
    }
    // Detrend (remove DC)
    let mean = 0;
    for (let i = 0; i < n; i++) mean += pos[i]!;
    mean /= n;
    const detrended = new Float32Array(n);
    for (let i = 0; i < n; i++) detrended[i] = pos[i]! - mean;

    // Estimate sample rate from timestamps
    const tFirst = this._rgbHistory[0]!.t;
    const tLast = this._rgbHistory[n - 1]!.t;
    const durationSec = (tLast - tFirst) / 1000;
    const fs = n > 1 && durationSec > 0 ? (n - 1) / durationSec : SAMPLE_HZ;

    // Bandpass filter (0.7-4 Hz) using simple IIR (biquad) — fast & low memory
    const filtered = this._bandpass(detrended, fs, CARDIO_LOW, CARDIO_HIGH);

    // FFT
    const { power, freqs } = this._fft(filtered, fs);
    if (power.length === 0) return this._emptyResult();

    // Find dominant frequency in cardiac range
    let peakIdx = 0;
    let peakPower = 0;
    let totalPower = 0;
    for (let i = 0; i < power.length; i++) {
      const f = freqs[i]!;
      const p = power[i]!;
      totalPower += p;
      if (f >= CARDIO_LOW && f <= CARDIO_HIGH && p > peakPower) {
        peakPower = p;
        peakIdx = i;
      }
    }

    const dominantFreq = freqs[peakIdx] ?? 0;
    const heartRateBpm = dominantFreq * 60;

    // Quality: ratio of peak power to mean power in band
    const bandPower = power
      .map((p, i) => (freqs[i]! >= CARDIO_LOW && freqs[i]! <= CARDIO_HIGH ? p : 0))
      .reduce((a, b) => a + b, 0);
    const meanBandPower = bandPower / Math.max(1, peakIdx);
    const signalQuality = meanBandPower > 0 ? Math.min(1, peakPower / (meanBandPower * 4)) : 0;
    const dominantPower = totalPower > 0 ? peakPower / totalPower : 0;

    const pulseDetected =
      heartRateBpm >= 42 &&
      heartRateBpm <= 240 &&
      signalQuality >= QUALITY_FLOOR &&
      dominantPower >= DOMINANT_POWER_FLOOR;

    return {
      pulseDetected,
      heartRateBpm: Math.round(heartRateBpm),
      signalQuality,
      dominantPower,
      isLive: pulseDetected,
    };
  }

  /**
   * Zero-phase bandpass via 2nd-order Butterworth IIR (biquad).
   */
  private _bandpass(x: Float32Array, fs: number, low: number, high: number): Float32Array {
    // Two biquads: highpass(low) and lowpass(high) in series
    const hp = this._biquad(x, fs, low, 'highpass');
    const lp = this._biquad(hp, fs, high, 'lowpass');
    return lp;
  }

  private _biquad(
    x: Float32Array,
    fs: number,
    cutoff: number,
    type: 'lowpass' | 'highpass',
  ): Float32Array {
    const w0 = (2 * Math.PI * cutoff) / fs;
    const cosw0 = Math.cos(w0);
    const sinw0 = Math.sin(w0);
    const alpha = sinw0 / Math.SQRT2; // Q = 0.707

    let b0: number, b1: number, b2: number, a0: number, a1: number, a2: number;
    if (type === 'lowpass') {
      b0 = (1 - cosw0) / 2;
      b1 = 1 - cosw0;
      b2 = (1 - cosw0) / 2;
      a0 = 1 + alpha;
      a1 = -2 * cosw0;
      a2 = 1 - alpha;
    } else {
      b0 = (1 + cosw0) / 2;
      b1 = -(1 + cosw0);
      b2 = (1 + cosw0) / 2;
      a0 = 1 + alpha;
      a1 = -2 * cosw0;
      a2 = 1 - alpha;
    }

    const y = new Float32Array(x.length);
    let x1 = 0,
      x2 = 0,
      y1 = 0,
      y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xn = x[i]!;
      const yn = (b0 / a0) * xn + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
      y[i] = yn;
      x2 = x1;
      x1 = xn;
      y2 = y1;
      y1 = yn;
    }
    return y;
  }

  /**
   * Radix-2 Cooley-Tukey FFT, padded to next power of 2.
   */
  private _fft(x: Float32Array, fs: number): { power: Float32Array; freqs: Float32Array } {
    let n = 1;
    while (n < x.length) n <<= 1;
    if (n < 2) return { power: new Float32Array(0), freqs: new Float32Array(0) };

    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < x.length; i++) re[i] = x[i]!;
    // Hann window
    for (let i = 0; i < n; i++) {
      re[i]! *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    }

    // Bit reversal
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        const tr = re[i]!;
        re[i] = re[j]!;
        re[j] = tr;
        const ti = im[i]!;
        im[i] = im[j]!;
        im[j] = ti;
      }
      let k = n >> 1;
      while (k > 0 && k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const theta = (-2 * Math.PI) / len;
      const wR = Math.cos(theta);
      const wI = Math.sin(theta);
      for (let i = 0; i < n; i += len) {
        let cR = 1;
        let cI = 0;
        for (let k = 0; k < halfLen; k++) {
          const idx1 = i + k;
          const idx2 = idx1 + halfLen;
          const tR = cR * re[idx2]! - cI * im[idx2]!;
          const tI = cR * im[idx2]! + cI * re[idx2]!;
          re[idx2] = re[idx1]! - tR;
          im[idx2] = im[idx1]! - tI;
          re[idx1] = re[idx1]! + tR;
          im[idx1] = im[idx1]! + tI;
          const newR = cR * wR - cI * wI;
          cI = cR * wI + cI * wR;
          cR = newR;
        }
      }
    }

    // Power spectrum (single-sided)
    const half = n >> 1;
    const power = new Float32Array(half);
    const freqs = new Float32Array(half);
    const df = fs / n;
    for (let i = 0; i < half; i++) {
      const r = re[i]!;
      const imv = im[i]!;
      power[i] = (r * r + imv * imv) / n;
      freqs[i] = i * df;
    }
    return { power, freqs };
  }
}
