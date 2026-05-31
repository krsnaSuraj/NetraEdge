package com.netraedge

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.WritableArray
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessor.Frame
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min

/**
 * FaceCropPlugin — extracts face region from camera frame and returns
 * normalized 112x112 RGB pixel data as Float32Array.
 *
 * This is the CRITICAL missing piece that bridges camera frames to the
 * face recognition pipeline. Without this, all face data is zero-filled.
 *
 * Flow:
 * 1. Receives Frame + face bounding box from JS
 * 2. Converts YUV_420_888 to RGB Bitmap
 * 3. Crops face region using bounding box (with padding)
 * 4. Resizes to 112x112 using bilinear interpolation
 * 5. Normalizes each pixel to 0.0-1.0
 * 6. Returns WritableArray of 37632 floats (112*112*3)
 *
 * Usage from JS frame processor:
 *   const plugin = VisionCameraProxy.initFrameProcessorPlugin('faceCrop', {});
 *   const result = plugin.call(frame, { x, y, width, height });
 *   // result.faceData = Float32Array(37632)
 */
class FaceCropPlugin : FrameProcessorPlugin {

    companion object {
        private const val INPUT_SIZE = 112
        private const val TAG = "FaceCropPlugin"
    }

    @OptIn(ExperimentalGetImage::class)
    override fun callback(frame: Frame, params: Map<String, Any>): Any? {
        return try {
            val imageProxy = frame.imageProxy ?: return null

            // Extract bounding box from params
            val x = (params["x"] as? Number)?.toDouble()?.toInt() ?: 0
            val y = (params["y"] as? Number)?.toDouble()?.toInt() ?: 0
            val width = (params["width"] as? Number)?.toDouble()?.toInt() ?: 0
            val height = (params["height"] as? Number)?.toDouble()?.toInt() ?: 0

            if (width <= 0 || height <= 0) return null

            // Convert YUV to Bitmap
            val bitmap = imageProxyToBitmap(imageProxy) ?: return null

            // Crop face region with padding
            val faceBitmap = cropFace(bitmap, x, y, width, height, bitmap.width, bitmap.height)

            // Resize to 112x112
            val resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true)

            // Normalize to 0.0-1.0 and extract RGB
            val faceData = bitmapToNormalizedFloats(resizedBitmap)

            // Clean up
            if (faceBitmap !== bitmap) faceBitmap.recycle()
            resizedBitmap.recycle()
            bitmap.recycle()

            // Return result
            val result = Arguments.createMap()
            val array = Arguments.createArray()
            for (v in faceData) {
                array.pushDouble(v.toDouble())
            }
            result.putArray("faceData", array)
            result.putInt("width", INPUT_SIZE)
            result.putInt("height", INPUT_SIZE)

            result
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Face crop failed: ${e.message}")
            null
        }
    }

    /**
     * Convert ImageProxy (YUV_420_888) to RGB Bitmap.
     * Uses JPEG compression as intermediary for reliable conversion.
     */
    @OptIn(ExperimentalGetImage::class)
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // U and V are swapped for NV21 format
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Crop face region from full frame bitmap.
     * Adds padding around the bounding box for better context.
     * Handles edge cases (face near frame boundaries).
     */
    private fun cropFace(
        bitmap: Bitmap,
        x: Int, y: Int, width: Int, height: Int,
        frameWidth: Int, frameHeight: Int
    ): Bitmap {
        // Add 20% padding around the face for better context
        val paddingX = (width * 0.2).toInt()
        val paddingY = (height * 0.2).toInt()

        // Calculate crop bounds with padding
        val cropLeft = max(0, x - paddingX)
        val cropTop = max(0, y - paddingY)
        val cropRight = min(frameWidth, x + width + paddingX)
        val cropBottom = min(frameHeight, y + height + paddingY)

        val cropWidth = cropRight - cropLeft
        val cropHeight = cropBottom - cropTop

        // Ensure minimum size
        val safeWidth = max(cropWidth, 10)
        val safeHeight = max(cropHeight, 10)

        return Bitmap.createBitmap(bitmap, cropLeft, cropTop, safeWidth, safeHeight)
    }

    /**
     * Convert Bitmap to normalized float array (0.0-1.0).
     * Returns Float32Array of size 112*112*3 = 37632.
     * Pixel order: R,G,B,R,G,B,... in row-major order.
     */
    private fun bitmapToNormalizedFloats(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val result = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            result[i * 3] = r
            result[i * 3 + 1] = g
            result[i * 3 + 2] = b
        }

        return result
    }
}
