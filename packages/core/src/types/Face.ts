/**
 * Face detection result from the native camera layer.
 *
 * Coordinates are normalized (0.0–1.0) relative to the frame dimensions.
 */
export interface FaceDetection {
  readonly boundingBox: BoundingBox;
  readonly landmarks: FaceLandmarks;
  readonly rollAngle: number;
  readonly yawAngle: number;
  readonly confidence: number;
}

export interface BoundingBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface FaceLandmarks {
  readonly leftEye: Point2D;
  readonly rightEye: Point2D;
  readonly noseBase: Point2D;
  readonly leftMouth: Point2D;
  readonly rightMouth: Point2D;
}

export interface Point2D {
  readonly x: number;
  readonly y: number;
}

/**
 * 478-point face mesh from MediaPipe.
 * Each point has x, y, z in normalized coordinates.
 */
export interface FaceMesh {
  readonly points: ReadonlyArray<Point3D>;
  readonly boundingBox: BoundingBox;
}

export interface Point3D {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}
