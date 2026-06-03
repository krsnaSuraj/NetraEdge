package com.netraedge

import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * SOTA 5-point face alignment.
 *
 * Given 5 source keypoints (left eye, right eye, nose tip, left mouth,
 * right mouth), compute a similarity transform (rotation + uniform scale
 * + translation) to MTCNN's canonical 112x112 positions, and sample the
 * source image at the warped grid.
 *
 * Two source formats are supported:
 *   1) YUV_420_888 (CameraX) — fast, no decoding.
 *   2) ARGB_8888  Bitmap      — for React Native (base64 JPEG/PNG from JS).
 *
 * Output is a CHW float buffer of 3*112*112 = 37632 values, normalized
 * to mean=127.5, std=128.0 (MobileFaceNet's expected input range).
 */
object FaceAligner {

    private val DST = arrayOf(
        floatArrayOf(38.2946f, 51.6963f),  // left eye
        floatArrayOf(73.5318f, 51.5014f),  // right eye
        floatArrayOf(56.0252f, 71.7366f),  // nose
        floatArrayOf(41.5493f, 92.3655f),  // left mouth
        floatArrayOf(70.7299f, 92.2041f)   // right mouth
    )

    /**
     * YUV_420_888 source. Used by the standalone CameraX demo.
     */
    fun align(
        yuv: ByteArray?,
        yRowStride: Int,
        width: Int,
        height: Int,
        srcKey: Array<PointF>,
        out: FloatArray,
        bitmap: Bitmap? = null
    ) {
        require(srcKey.size == 5) { "expected 5 source keypoints" }
        if (bitmap != null) {
            alignFromBitmap(bitmap, srcKey, out)
            return
        }
        requireNotNull(yuv) { "Either yuv or bitmap must be provided" }
        val params = SimilarityTransform.solve(srcKey, DST)
        val a = params[0]; val b = params[1]
        val tx = params[2]; val ty = params[3]
        val w = 112; val c = 112
        for (y in 0 until c) {
            for (x in 0 until w) {
                val sx = (a * x + b * y + tx)
                val sy = (-b * x + a * y + ty)
                val ix = sx.toInt().coerceIn(0, width - 1)
                val iy = sy.toInt().coerceIn(0, height - 1)
                val yVal = yuv[iy * yRowStride + ix].toInt() and 0xFF
                out[0 * w * c + y * w + x] = (yVal - 127.5f) / 128.0f
                out[1 * w * c + y * w + x] = (yVal - 127.5f) / 128.0f
                out[2 * w * c + y * w + x] = (yVal - 127.5f) / 128.0f
            }
        }
    }

    /**
     * Bitmap (ARGB_8888) source. Used by the React Native bridge.
     * Falls back to nearest-neighbor sampling for speed — face alignment
     * is robust enough at 112x112.
     */
    private fun alignFromBitmap(bitmap: Bitmap, srcKey: Array<PointF>, out: FloatArray) {
        val params = SimilarityTransform.solve(srcKey, DST)
        val a = params[0]; val b = params[1]
        val tx = params[2]; val ty = params[3]
        val w = 112; val c = 112
        val bw = bitmap.width
        val bh = bitmap.height
        val pixels = IntArray(bw * bh)
        bitmap.getPixels(pixels, 0, bw, 0, 0, bw, bh)
        for (y in 0 until c) {
            for (x in 0 until w) {
                val sx = (a * x + b * y + tx)
                val sy = (-b * x + a * y + ty)
                val ix = sx.toInt().coerceIn(0, bw - 1)
                val iy = sy.toInt().coerceIn(0, bh - 1)
                val argb = pixels[iy * bw + ix]
                val r = ((argb shr 16) and 0xFF).toFloat()
                val g = ((argb shr 8) and 0xFF).toFloat()
                val bl = (argb and 0xFF).toFloat()
                out[0 * w * c + y * w + x] = (r - 127.5f) / 128.0f
                out[1 * w * c + y * w + x] = (g - 127.5f) / 128.0f
                out[2 * w * c + y * w + x] = (bl - 127.5f) / 128.0f
            }
        }
    }

    /**
     * 5-point similarity transform solver (rotation + uniform scale + translation).
     * Closed-form solution matching the canonical MTCNN alignment.
     */
    private object SimilarityTransform {
        fun solve(src: Array<PointF>, dst: Array<FloatArray>): FloatArray {
            var srcCx = 0f; var srcCy = 0f
            var dstCx = 0f; var dstCy = 0f
            for (i in 0 until 5) {
                srcCx += src[i].x; srcCy += src[i].y
                dstCx += dst[i][0]; dstCy += dst[i][1]
            }
            srcCx /= 5f; srcCy /= 5f
            dstCx /= 5f; dstCy /= 5f

            var srcSxx = 0f; var srcSyy = 0f
            var dstSxx = 0f; var dstSyy = 0f
            for (i in 0 until 5) {
                val dxS = src[i].x - srcCx
                val dyS = src[i].y - srcCy
                val dxD = dst[i][0] - dstCx
                val dyD = dst[i][1] - dstCy
                srcSxx += dxS * dxS; srcSyy += dyS * dyS
                dstSxx += dxD * dxD; dstSyy += dyD * dyD
            }

            val srcScale = sqrt(srcSxx + srcSyy)
            val dstScale = sqrt(dstSxx + dstSyy)
            val scale = dstScale / srcScale

            var num = 0f; var den = 0f
            for (i in 0 until 5) {
                val dxS = src[i].x - srcCx
                val dyS = src[i].y - srcCy
                val dxD = dst[i][0] - dstCx
                val dyD = dst[i][1] - dstCy
                num += dxS * dyD - dyS * dxD
                den += dxS * dxD + dyS * dyD
            }
            val angle = atan2(num, den)
            val a = (scale * cos(angle)).toFloat()
            val b = (scale * sin(angle)).toFloat()
            val tx = (dstCx - a * srcCx - b * srcCy)
            val ty = (dstCy - b * srcCx + a * srcCy)
            return floatArrayOf(a, b, tx, ty, scale)
        }
    }
}
