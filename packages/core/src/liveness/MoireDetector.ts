/**
 * MoireDetector — FFT-based screen spoof detection.
 *
 * Analyzes frequency domain patterns in face images to detect
 * moiré interference patterns characteristic of screen captures.
 *
 * Screen spoofs: peaks at 8-40 cycles/image in power spectrum.
 * Real faces: smooth ~1/f spectrum with no periodic peaks.
 *
 * Performance: ~8-15ms on mobile CPU
 */

import type { MoirResult } from '../types/Liveness';

export type { MoirResult };

const MOIRE_THRESHOLD = 0.4;
const MIN_PEAK_MAGNITUDE = 2.5;
const FREQ_LOW = 8;
const FREQ_HIGH = 40;

export function detectMoire(
  faceData: Float32Array | number[],
  width = 112,
  height = 112,
): MoirResult {
  const n = width * height;
  const grayscale = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = faceData[i * 3] ?? 0;
    const g = faceData[i * 3 + 1] ?? 0;
    const b = faceData[i * 3 + 2] ?? 0;
    grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  const windowed = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    const wy = 0.5 * (1 - Math.cos((2 * Math.PI * y) / Math.max(1, height - 1)));
    for (let x = 0; x < width; x++) {
      const wx = 0.5 * (1 - Math.cos((2 * Math.PI * x) / Math.max(1, width - 1)));
      const idx = y * width + x;
      windowed[idx] = (grayscale[idx] ?? 0) * wx * wy;
    }
  }

  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  real.set(windowed);

  for (let y = 0; y < height; y++) {
    const rowR = new Float32Array(width);
    const rowI = new Float32Array(width);
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      rowR[x] = real[idx] ?? 0;
      rowI[x] = imag[idx] ?? 0;
    }
    fft1D(rowR, rowI, width);
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      real[idx] = rowR[x] ?? 0;
      imag[idx] = rowI[x] ?? 0;
    }
  }

  for (let x = 0; x < width; x++) {
    const colR = new Float32Array(height);
    const colI = new Float32Array(height);
    for (let y = 0; y < height; y++) {
      const idx = y * width + x;
      colR[y] = real[idx] ?? 0;
      colI[y] = imag[idx] ?? 0;
    }
    fft1D(colR, colI, height);
    for (let y = 0; y < height; y++) {
      const idx = y * width + x;
      real[idx] = colR[y] ?? 0;
      imag[idx] = colI[y] ?? 0;
    }
  }

  const power = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const re = real[i] ?? 0;
    const im = imag[i] ?? 0;
    power[i] = re * re + im * im;
  }

  const centerX = width / 2;
  const centerY = height / 2;
  const maxRadius = Math.floor(Math.min(width, height) / 2);
  const profile = new Float32Array(maxRadius);
  const counts = new Float32Array(maxRadius);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      const radius = Math.floor(Math.sqrt(dx * dx + dy * dy));
      if (radius < maxRadius) {
        const pVal = power[y * width + x] ?? 0;
        const cVal = counts[radius] ?? 0;
        profile[radius] = (profile[radius] ?? 0) + pVal;
        counts[radius] = cVal + 1;
      }
    }
  }

  for (let i = 0; i < maxRadius; i++) {
    const c = counts[i] ?? 0;
    if (c > 0) {
      profile[i] = (profile[i] ?? 0) / c;
    }
  }

  const start = Math.min(FREQ_LOW, profile.length);
  const end = Math.min(FREQ_HIGH, profile.length);
  const peaks: number[] = [];

  if (end - start >= 3) {
    let sum = 0;
    let sumSq = 0;
    let count = 0;
    for (let i = start; i < end; i++) {
      const v = profile[i] ?? 0;
      sum += v;
      sumSq += v * v;
      count++;
    }
    const mean = count > 0 ? sum / count : 0;
    const variance = count > 0 ? sumSq / count - mean * mean : 0;
    const std = Math.sqrt(Math.max(0, variance));

    for (let i = start + 1; i < end - 1; i++) {
      const curr = profile[i] ?? 0;
      const prev = profile[i - 1] ?? 0;
      const next = profile[i + 1] ?? 0;
      if (curr > prev && curr > next && curr > mean + MIN_PEAK_MAGNITUDE * std) {
        peaks.push(i);
      }
    }
  }

  let periodicity = 0;
  if (end - start >= 5) {
    const segment: number[] = [];
    for (let i = start; i < end; i++) {
      segment.push(profile[i] ?? 0);
    }
    const segN = segment.length;
    let segMean = 0;
    for (let i = 0; i < segN; i++) segMean += segment[i] ?? 0;
    segMean /= segN;
    let segVar = 0;
    for (let i = 0; i < segN; i++) {
      const diff = (segment[i] ?? 0) - segMean;
      segVar += diff * diff;
    }
    segVar /= segN;
    if (segVar > 1e-10) {
      let maxCorr = 0;
      for (let lag = 1; lag < Math.floor(segN / 2); lag++) {
        let corr = 0;
        for (let i = 0; i < segN - lag; i++) {
          corr += ((segment[i] ?? 0) - segMean) * ((segment[i + lag] ?? 0) - segMean);
        }
        corr /= (segN - lag) * segVar;
        maxCorr = Math.max(maxCorr, Math.abs(corr));
      }
      periodicity = maxCorr;
    }
  }

  const moireScore = Math.min(1, peaks.length * 0.15 + periodicity * 0.5);

  return {
    moireScore,
    peaks,
    periodicity,
    detected: moireScore > MOIRE_THRESHOLD,
  };
}

function fft1D(real: Float32Array, imag: Float32Array, n: number): void {
  let paddedN = 1;
  while (paddedN < n) paddedN <<= 1;
  if (paddedN === n) {
    fft1DRadix2(real, imag, n);
    return;
  }
  const padR = new Float32Array(paddedN);
  const padI = new Float32Array(paddedN);
  for (let i = 0; i < n; i++) {
    padR[i] = real[i] ?? 0;
    padI[i] = imag[i] ?? 0;
  }
  fft1DRadix2(padR, padI, paddedN);
  for (let i = 0; i < n; i++) {
    real[i] = padR[i] ?? 0;
    imag[i] = padI[i] ?? 0;
  }
}

function fft1DRadix2(real: Float32Array, imag: Float32Array, n: number): void {
  if (n <= 1) return;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tmpR = real[i] ?? 0;
      const tmpI = imag[i] ?? 0;
      real[i] = real[j] ?? 0;
      imag[i] = imag[j] ?? 0;
      real[j] = tmpR;
      imag[j] = tmpI;
    }
    let k = n >> 1;
    while (k > 0 && k <= j) { j -= k; k >>= 1; }
    j += k;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wR = Math.cos(angle);
    const wI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let cR = 1;
      let cI = 0;
      for (let k = 0; k < halfLen; k++) {
        const idx1 = i + k;
        const idx2 = i + k + halfLen;
        const r1 = real[idx1] ?? 0;
        const i1 = imag[idx1] ?? 0;
        const r2 = real[idx2] ?? 0;
        const i2 = imag[idx2] ?? 0;
        const tR = cR * r2 - cI * i2;
        const tI = cR * i2 + cI * r2;
        real[idx2] = r1 - tR;
        imag[idx2] = i1 - tI;
        real[idx1] = r1 + tR;
        imag[idx1] = i1 + tI;
        const newR = cR * wR - cI * wI;
        cI = cR * wI + cI * wR;
        cR = newR;
      }
    }
  }
}
