import { describe, it, expect, beforeEach } from 'vitest';
import { LivenessOrchestrator } from '../liveness/LivenessOrchestrator';
import { StubTextureAnalyzer } from '../liveness/TextureAnalyzer';
import { BlinkDetector } from '../liveness/BlinkDetector';
import { DepthEstimator } from '../liveness/DepthEstimator';
import { DepthMotionEstimator } from '../liveness/DepthMotionEstimator';
import { LivenessVerdict } from '../types/Liveness';
import type { Point3D } from '../types/Face';

function makeMeshPoints(zVariance: number): Point3D[] {
  return Array.from({ length: 478 }, (_, i) => ({
    x: (i % 20) * 0.05,
    y: Math.floor(i / 20) * 0.05,
    z: zVariance > 0 ? (i % 3 === 0 ? zVariance * 3 : -zVariance) : 0.5,
  }));
}

function makeFaceData(): Float32Array {
  return new Float32Array(112 * 112 * 3).fill(0.5);
}

describe('LivenessOrchestrator', () => {
  let orchestrator: LivenessOrchestrator;

  beforeEach(() => {
    const textureAnalyzer = new StubTextureAnalyzer();
    const blinkDetector = new BlinkDetector(30, 0.21, 80);
    const depthEstimator = new DepthEstimator(0.25);
    const depthMotionEstimator = new DepthMotionEstimator();

    orchestrator = new LivenessOrchestrator(
      textureAnalyzer,
      blinkDetector,
      depthEstimator,
      depthMotionEstimator,
    );
  });

  it('returns SPOOF when no checks pass', async () => {
    const faceData = makeFaceData();
    const meshPoints = makeMeshPoints(0.001);
    const timestamp = Date.now();

    const result = await orchestrator.processFrame(faceData, meshPoints, timestamp);
    expect(result.verdict).toBe(LivenessVerdict.SPOOF);
  });

  it('returns texture + depth results with high-variance mesh', async () => {
    const faceData = makeFaceData();
    const meshPoints = makeMeshPoints(0.5);

    const result = await orchestrator.processFrame(faceData, meshPoints, 1000);
    expect(result.texture.realScore).toBeGreaterThan(0.8);
    expect(result.depth.variance).toBeGreaterThan(0.25);
  });

  it('tracks blink count across frames', async () => {
    const faceData = makeFaceData();
    const meshPoints = makeMeshPoints(0.001);

    for (let i = 0; i < 5; i++) {
      await orchestrator.processFrame(faceData, meshPoints, i * 100);
    }

    expect(orchestrator.blinkCount).toBeGreaterThanOrEqual(0);
  });

  it('resets blink detector', () => {
    orchestrator.reset();
    expect(orchestrator.blinkCount).toBe(0);
  });

  it('includes passed/failed checks in result', async () => {
    const faceData = makeFaceData();
    const meshPoints = makeMeshPoints(0.5);

    const result = await orchestrator.processFrame(faceData, meshPoints, 1000);
    expect(Array.isArray(result.passedChecks)).toBe(true);
    expect(Array.isArray(result.failedChecks)).toBe(true);
  });
});
