package com.netraedge

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * NetraEdgeModule — TFLite inference engine for face recognition and liveness.
 *
 * Loads two TFLite models from app assets:
 * - face_recognition.tflite → 128-d L2-normalized embedding
 * - liveness_detector.tflite → 3-class softmax [real, print, screen]
 */
class NetraEdgeModule {
    companion object {
        private const val TAG = "NetraEdge"
        private const val INPUT_SIZE = 112
        private const val EMBEDDING_DIM = 128
        private const val LIVENESS_CLASSES = 3
    }

    private var recognitionInterpreter: Interpreter? = null
    private var livenessInterpreter: Interpreter? = null
    private var initialized = false

    fun isInitialized(): Boolean = initialized

    fun initialize(context: Context): Boolean {
        return try {
            recognitionInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "face_recognition.tflite")
            )
            livenessInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "liveness_detector.tflite")
            )
            initialized = true
            Log.i(TAG, "TFLite models loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite models: ${e.message}")
            false
        }
    }

    /**
     * Run face recognition inference.
     *
     * Input: 112×112×3 RGB image (normalized floats, 0–1)
     * Output: 128-dimensional L2-normalized embedding
     */
    fun runRecognition(inputData: FloatArray): FloatArray? {
        val interpreter = recognitionInterpreter ?: run {
            Log.e(TAG, "Recognition model not initialized")
            return null
        }
        if (!initialized) return null

        return try {
            val input = Array(1) { Array(INPUT_SIZE) { FloatArray(INPUT_SIZE * 3) } }
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val idx = (y * INPUT_SIZE + x) * 3
                    input[0][y][x * 3] = inputData[idx]
                    input[0][y][x * 3 + 1] = inputData[idx + 1]
                    input[0][y][x * 3 + 2] = inputData[idx + 2]
                }
            }

            val output = Array(1) { FloatArray(EMBEDDING_DIM) }
            interpreter.run(input, output)

            val embedding = output[0]
            var norm = 0f
            for (v in embedding) norm += v * v
            norm = Math.sqrt(norm.toDouble()).toFloat()
            if (norm > 0) {
                for (i in embedding.indices) embedding[i] /= norm
            }

            embedding
        } catch (e: Exception) {
            Log.e(TAG, "Recognition inference failed: ${e.message}")
            null
        }
    }

    /**
     * Run liveness detection inference.
     *
     * Input: 112×112×3 RGB image (normalized floats, 0–1)
     * Output: 3-class probabilities [real, print, screen]
     */
    fun runLiveness(inputData: FloatArray): FloatArray? {
        val interpreter = livenessInterpreter ?: run {
            Log.e(TAG, "Liveness model not initialized")
            return null
        }
        if (!initialized) return null

        return try {
            val input = Array(1) { Array(INPUT_SIZE) { FloatArray(INPUT_SIZE * 3) } }
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val idx = (y * INPUT_SIZE + x) * 3
                    input[0][y][x * 3] = inputData[idx]
                    input[0][y][x * 3 + 1] = inputData[idx + 1]
                    input[0][y][x * 3 + 2] = inputData[idx + 2]
                }
            }

            val output = Array(1) { FloatArray(LIVENESS_CLASSES) }
            interpreter.run(input, output)
            output[0]
        } catch (e: Exception) {
            Log.e(TAG, "Liveness inference failed: ${e.message}")
            null
        }
    }

    /**
     * Compute cosine similarity between two embeddings.
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Dimension mismatch" }
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return (-1f).coerceAtLeast(1f.coerceAtMost(dot))
    }

    /**
     * Process raw camera frame pixels: crop face region, resize to 112x112,
     * normalize to 0.0-1.0. This is the CRITICAL bridge between camera
     * frames and the face recognition pipeline.
     *
     * @param imageData Raw pixel data as float array (R,G,B,R,G,B,...)
     * @param frameWidth Width of the original camera frame
     * @param frameHeight Height of the original camera frame
     * @param faceX Face bounding box left coordinate
     * @param faceY Face bounding box top coordinate
     * @param faceWidth Face bounding box width
     * @param faceHeight Face bounding box height
     * @return Float array of 37632 values (112*112*3) or null on failure
     */
    fun cropFace(
        imageData: FloatArray,
        frameWidth: Int,
        frameHeight: Int,
        faceX: Int,
        faceY: Int,
        faceWidth: Int,
        faceHeight: Int
    ): FloatArray? {
        if (faceWidth <= 0 || faceHeight <= 0) return null

        // Add 20% padding around face for better context
        val paddingX = (faceWidth * 0.2).toInt()
        val paddingY = (faceHeight * 0.2).toInt()

        val cropLeft = maxOf(0, faceX - paddingX)
        val cropTop = maxOf(0, faceY - paddingY)
        val cropRight = minOf(frameWidth, faceX + faceWidth + paddingX)
        val cropBottom = minOf(frameHeight, faceY + faceHeight + paddingY)

        val cropWidth = cropRight - cropLeft
        val cropHeight = cropBottom - cropTop

        if (cropWidth <= 0 || cropHeight <= 0) return null

        // Extract and resize using bilinear interpolation
        val result = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                // Map output pixel to input crop region
                val srcX = cropLeft + (x.toFloat() / INPUT_SIZE * cropWidth).toInt()
                val srcY = cropTop + (y.toFloat() / INPUT_SIZE * cropHeight).toInt()

                val srcIdx = (srcY * frameWidth + srcX) * 3
                val dstIdx = (y * INPUT_SIZE + x) * 3

                if (srcIdx + 2 < imageData.size && dstIdx + 2 < result.size) {
                    result[dstIdx] = imageData[srcIdx]         // R
                    result[dstIdx + 1] = imageData[srcIdx + 1] // G
                    result[dstIdx + 2] = imageData[srcIdx + 2] // B
                }
            }
        }

        return result
    }

    /**
     * Convert YUV_420_888 byte data to normalized RGB float array.
     * This is used when receiving raw camera frame data.
     *
     * @param yuvData YUV_420_888 byte array
     * @param width Frame width
     * @param height Frame height
     * @return Normalized RGB float array or null
     */
    fun yuvToNormalizedFloats(yuvData: ByteArray, width: Int, height: Int): FloatArray? {
        return try {
            val argb = IntArray(width * height)
            decodeYUV420ToARGB(yuvData, argb, width, height)

            val result = FloatArray(width * height * 3)
            for (i in argb.indices) {
                val pixel = argb[i]
                result[i * 3] = ((pixel shr 16) and 0xFF) / 255.0f     // R
                result[i * 3 + 1] = ((pixel shr 8) and 0xFF) / 255.0f  // G
                result[i * 3 + 2] = (pixel and 0xFF) / 255.0f          // B
            }
            result
        } catch (e: Exception) {
            Log.e(TAG, "YUV conversion failed: ${e.message}")
            null
        }
    }

    /**
     * Decode YUV_420_888 to ARGB pixel array.
     */
    private fun decodeYUV420ToARGB(yuvData: ByteArray, argb: IntArray, width: Int, height: Int) {
        val ySize = width * height
        val uvSize = ySize / 4

        for (y in 0 until height) {
            for (x in 0 until width) {
                val yIndex = y * width + x
                val yValue = (yuvData[yIndex].toInt() and 0xFF) - 128

                val uvIndex = ySize + (y / 2) * (width / 2) + (x / 2)
                val uValue = (yuvData[uvIndex].toInt() and 0xFF) - 128
                val vValue = (yuvData[uvIndex + uvSize].toInt() and 0xFF) - 128

                val r = (yValue + 1.370705f * vValue).toInt().coerceIn(0, 255)
                val g = (yValue - 0.337633f * uValue - 0.698001f * vValue).toInt().coerceIn(0, 255)
                val b = (yValue + 1.732446f * uValue).toInt().coerceIn(0, 255)

                argb[yIndex] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    fun close() {
        recognitionInterpreter?.close()
        livenessInterpreter?.close()
        recognitionInterpreter = null
        livenessInterpreter = null
        initialized = false
        Log.i(TAG, "NetraEdge module closed")
    }
}
