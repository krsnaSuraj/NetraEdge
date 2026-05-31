/**
 * QualityGate — rejects poor quality face frames before encoding.
 *
 * Validates brightness, sharpness, face size, and pose angle.
 * Prevents garbage data from entering the face pipeline.
 */

export interface QualityResult {
  readonly score: number;
  readonly passed: boolean;
  readonly checks: {
    readonly brightness: { readonly score: number; readonly passed: boolean };
    readonly sharpness: { readonly score: number; readonly passed: boolean };
    readonly faceSize: { readonly score: number; readonly passed: boolean };
    readonly poseAngle: { readonly score: number; readonly passed: boolean };
  };
  readonly failureReason: string | null;
}

interface QualityConfig {
  readonly minBrightness: number;
  readonly maxBrightness: number;
  readonly minSharpness: number;
  readonly minFaceSize: number;
  readonly maxYawAngle: number;
  readonly maxPitchAngle: number;
  readonly threshold: number;
}

const DEFAULT_CONFIG: QualityConfig = {
  minBrightness: 40,
  maxBrightness: 220,
  minSharpness: 50,
  minFaceSize: 80,
  maxYawAngle: 20,
  maxPitchAngle: 20,
  threshold: 0.6,
};

export function evaluateQuality(
  faceData: Float32Array | number[],
  faceBounds?: { readonly width: number; readonly height: number },
  landmarks?: Record<string, { readonly x: number; readonly y: number }>,
  config: QualityConfig = DEFAULT_CONFIG,
): QualityResult {
  const width = 112;
  const height = 112;

  // Brightness
  let brightnessSum = 0;
  const npix = width * height;
  for (let i = 0; i < npix; i++) {
    brightnessSum +=
      0.299 * (faceData[i * 3] ?? 0) * 255 +
      0.587 * (faceData[i * 3 + 1] ?? 0) * 255 +
      0.114 * (faceData[i * 3 + 2] ?? 0) * 255;
  }
  const brightness = npix > 0 ? brightnessSum / npix : 128;
  const brightnessMid = (config.minBrightness + config.maxBrightness) / 2;
  const brightnessRange = (config.maxBrightness - config.minBrightness) / 2;
  const brightnessScore = Math.max(0, 1 - Math.abs(brightness - brightnessMid) / brightnessRange);
  const brightnessPassed = brightness >= config.minBrightness && brightness <= config.maxBrightness;

  // Sharpness (Laplacian variance)
  const gray = new Float32Array(npix);
  for (let i = 0; i < npix; i++) {
    gray[i] = 0.299 * (faceData[i * 3] ?? 0) + 0.587 * (faceData[i * 3 + 1] ?? 0) + 0.114 * (faceData[i * 3 + 2] ?? 0);
  }
  let lapSum = 0;
  let lapSumSq = 0;
  let lapCount = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const center = gray[idx] ?? 0;
      const lap = -4 * center +
        (gray[(y - 1) * width + x] ?? 0) +
        (gray[(y + 1) * width + x] ?? 0) +
        (gray[y * width + (x - 1)] ?? 0) +
        (gray[y * width + (x + 1)] ?? 0);
      lapSum += lap;
      lapSumSq += lap * lap;
      lapCount++;
    }
  }
  const lapMean = lapCount > 0 ? lapSum / lapCount : 0;
  const lapVar = lapCount > 0 ? lapSumSq / lapCount - lapMean * lapMean : 0;
  const sharpness = Math.min(1, Math.max(0, lapVar / 200));
  const sharpnessPassed = sharpness * 200 >= config.minSharpness;

  // Face size
  const faceW = faceBounds?.width ?? 150;
  const faceH = faceBounds?.height ?? 150;
  const faceSize = Math.min(faceW, faceH);
  const faceSizeScore = faceSize >= 150 ? 1 : faceSize <= 80 ? 0 : (faceSize - 80) / 70;
  const faceSizePassed = faceW >= config.minFaceSize && faceH >= config.minFaceSize;

  // Pose
  const leftEye = landmarks?.['leftEye'];
  const rightEye = landmarks?.['rightEye'];
  const nose = landmarks?.['nose'];
  let yaw = 0;
  let pitch = 0;
  if (leftEye && rightEye && nose) {
    const eyeDist = Math.abs(rightEye.x - leftEye.x);
    if (eyeDist > 0) {
      const noseOffset = nose.x - (leftEye.x + rightEye.x) / 2;
      yaw = Math.atan2(noseOffset, eyeDist / 2) * (180 / Math.PI);
      const eyeY = (leftEye.y + rightEye.y) / 2;
      pitch = Math.atan2(nose.y - eyeY, eyeDist / 2) * (180 / Math.PI);
    }
  }
  const yawScore = Math.max(0, 1 - Math.abs(yaw) / config.maxYawAngle);
  const pitchScore = Math.max(0, 1 - Math.abs(pitch) / config.maxPitchAngle);
  const poseScore = (yawScore + pitchScore) / 2;
  const posePassed = Math.abs(yaw) <= config.maxYawAngle && Math.abs(pitch) <= config.maxPitchAngle;

  const score =
    0.25 * brightnessScore +
    0.30 * sharpness +
    0.20 * faceSizeScore +
    0.25 * poseScore;

  const passed = score >= config.threshold;

  let failureReason: string | null = null;
  if (!passed) {
    if (!brightnessPassed) failureReason = brightness < config.minBrightness ? 'Too dark' : 'Too bright';
    else if (!sharpnessPassed) failureReason = 'Image too blurry';
    else if (!faceSizePassed) failureReason = 'Face too small';
    else if (!posePassed) failureReason = 'Face angle too extreme';
  }

  return {
    score,
    passed,
    checks: {
      brightness: { score: brightnessScore, passed: brightnessPassed },
      sharpness: { score: sharpness, passed: sharpnessPassed },
      faceSize: { score: faceSizeScore, passed: faceSizePassed },
      poseAngle: { score: poseScore, passed: posePassed },
    },
    failureReason,
  };
}
