package com.netraedge

import android.graphics.PointF
import android.graphics.RectF

/**
 * Extracts the 5 canonical alignment keypoints (left eye, right eye, nose,
 * left mouth, right mouth) from MediaPipe's 468-landmark face mesh.
 *
 * MediaPipe canonical indices (per google/mediapipe face mesh topology):
 *   left eye:  33
 *   right eye: 263
 *   nose tip:  1
 *   left mouth: 61
 *   right mouth: 291
 */
object KeypointExtractor {

    private const val LEFT_EYE = 33
    private const val RIGHT_EYE = 263
    private const val NOSE_TIP = 1
    private const val LEFT_MOUTH = 61
    private const val RIGHT_MOUTH = 291

    fun extract(landmarks: List<PointF>): Array<PointF> {
        require(landmarks.size > RIGHT_MOUTH) { "expected >291 landmarks, got ${landmarks.size}" }
        return arrayOf(
            landmarks[LEFT_EYE],
            landmarks[RIGHT_EYE],
            landmarks[NOSE_TIP],
            landmarks[LEFT_MOUTH],
            landmarks[RIGHT_MOUTH]
        )
    }

    fun boundingBox(landmarks: List<PointF>, imageWidth: Int, imageHeight: Int): RectF {
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        for (kp in landmarks) {
            if (kp.x < minX) minX = kp.x
            if (kp.y < minY) minY = kp.y
            if (kp.x > maxX) maxX = kp.x
            if (kp.y > maxY) maxY = kp.y
        }
        return RectF(
            minX * imageWidth,
            minY * imageHeight,
            maxX * imageWidth,
            maxY * imageHeight
        )
    }
}
