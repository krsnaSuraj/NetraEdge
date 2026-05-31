/**
 * ColorAnalyzer — multi-color-space analysis for liveness detection.
 *
 * Analyzes chrominance distribution to distinguish real skin from
 * printed photos and screen captures.
 *
 * Real skin: consistent chrominance, warm undertones, smooth gradients
 * Print: flat saturation, halftone patterns, high gradient variance
 * Screen: high saturation spikes, moiré in chrominance, cool bias
 */

export interface ColorResult {
  readonly realScore: number;
  readonly skinConsistency: number;
  readonly saturationMean: number;
  readonly saturationStd: number;
  readonly gradientSmoothness: number;
  readonly skinDivergence: number;
  readonly isReal: boolean;
}

const SKIN_CB_MIN = 77;
const SKIN_CB_MAX = 127;
const SKIN_CR_MIN = 133;
const SKIN_CR_MAX = 173;

export function analyzeColor(
  faceData: Float32Array | number[],
  width = 112,
  height = 112,
): ColorResult {
  const n = width * height;

  // Convert to YCbCr
  const cb = new Float32Array(n);
  const cr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = (faceData[i * 3] ?? 0) * 255;
    const g = (faceData[i * 3 + 1] ?? 0) * 255;
    const b = (faceData[i * 3 + 2] ?? 0) * 255;
    cb[i] = 128 - 0.169 * r - 0.331 * g + 0.500 * b;
    cr[i] = 128 + 0.500 * r - 0.419 * g - 0.081 * b;
  }

  // Convert to HSV
  const saturation = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = faceData[i * 3] ?? 0;
    const g = faceData[i * 3 + 1] ?? 0;
    const b = faceData[i * 3 + 2] ?? 0;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    saturation[i] = max === 0 ? 0 : (max - min) / max;
  }

  // Skin mask
  let skinCount = 0;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const cbVal = cb[i] ?? 0;
    const crVal = cr[i] ?? 0;
    if (cbVal >= SKIN_CB_MIN && cbVal <= SKIN_CB_MAX &&
        crVal >= SKIN_CR_MIN && crVal <= SKIN_CR_MAX) {
      mask[i] = 1;
      skinCount++;
    }
  }

  // Skin consistency
  const skinConsistency = n > 0 ? skinCount / n : 0;

  // Saturation stats on skin pixels
  let satSum = 0;
  let satSumSq = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i]) {
      const s = (saturation[i] ?? 0) * 255;
      satSum += s;
      satSumSq += s * s;
    }
  }
  const satMean = skinCount > 0 ? satSum / skinCount : 0;
  const satVar = skinCount > 0 ? satSumSq / skinCount - satMean * satMean : 0;
  const satStd = Math.sqrt(Math.max(0, satVar));

  // Gradient smoothness
  let gradSum = 0;
  let gradCount = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      const i = y * width + x;
      if (mask[i] && mask[i + 1]) {
        gradSum += Math.abs((cb[i + 1] ?? 0) - (cb[i] ?? 0));
        gradSum += Math.abs((cr[i + 1] ?? 0) - (cr[i] ?? 0));
        gradCount += 2;
      }
    }
  }
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (mask[i] && mask[i + width]) {
        gradSum += Math.abs((cb[i + width] ?? 0) - (cb[i] ?? 0));
        gradSum += Math.abs((cr[i + width] ?? 0) - (cr[i] ?? 0));
        gradCount += 2;
      }
    }
  }
  const avgGrad = gradCount > 0 ? gradSum / gradCount : 30;
  const gradientSmoothness = Math.max(0, 1 - avgGrad / 30);

  // Skin divergence
  let cbMean = 0;
  let crMean = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i]) {
      cbMean += cb[i] ?? 0;
      crMean += cr[i] ?? 0;
    }
  }
  cbMean = skinCount > 0 ? cbMean / skinCount : 102;
  crMean = skinCount > 0 ? crMean / skinCount : 153;
  const cbDiff = (cbMean - 102) / 15;
  const crDiff = (crMean - 153) / 10;
  const skinDivergence = Math.min(1, Math.sqrt(cbDiff * cbDiff + crDiff * crDiff) / 10);

  // Scores
  const consistencyScore = skinConsistency > 0.4 && skinConsistency < 0.8 ? 1.0 :
    skinConsistency <= 0.4 ? skinConsistency / 0.4 :
    Math.max(0, 1 - (skinConsistency - 0.8) / 0.2);

  const satScore = satMean > 80 && satMean < 160 ? 1.0 :
    satMean <= 80 ? satMean / 80 :
    Math.max(0, 1 - (satMean - 160) / 100);

  const satStdScore = Math.max(0, 1 - satStd / 50);
  const divergenceScore = Math.max(0, 1 - skinDivergence);

  const realScore = Math.max(0, Math.min(1,
    0.15 * consistencyScore +
    0.25 * satScore +
    0.15 * satStdScore +
    0.25 * gradientSmoothness +
    0.20 * divergenceScore
  ));

  return {
    realScore,
    skinConsistency,
    saturationMean: satMean,
    saturationStd: satStd,
    gradientSmoothness,
    skinDivergence,
    isReal: realScore > 0.5,
  };
}
