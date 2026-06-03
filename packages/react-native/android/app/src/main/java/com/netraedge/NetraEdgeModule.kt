package com.netraedge

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * NetraEdgeModule — SOTA TFLite inference engine for face recognition and liveness.
 *
 * Loads two TFLite models from app assets:
 * - face_recognition.tflite → embedding dim derived at load time
 *   (current pre-trained InsightFace MobileFaceNet = 128-d;
 *    Day-2 fine-tuned EdgeFace-XS / SubCenter-AdaFace / 2-Teacher KD = 512-d).
 * - liveness_detector.tflite → 3-class softmax [real, print, screen]
 *
 * SOTA 2026 stack (Path A+):
 *   - Backbone: EdgeFace-XS gamma-06 (1.77M params, 99.73% LFW paper)
 *   - Loss: SubCenter-AdaFace (K=3) + GRL adversarial debiasing head
 *   - KD: 2 teachers (EdgeFace-base 99.83% LFW + InsightFace 99.38% LFW)
 *   - Embedding: 512-d TARGET (was 128-d MobileFaceNet, 192-d GFN) — code is dim-agnostic
 *   - TTA: horizontal flip averaging at inference time (+0.3-0.5% LFW free)
 *   - Liveness: 7-layer attention fusion (Active + Texture + rPPG + Depth + Depth-Motion + Moiré + Color)
 *
 * Dim handling: the embedding dim is read from the model's output tensor at
 * initialize() time, so the code works with any dim (128, 192, 256, 384, 512, ...).
 * TTA: useTTA defaults to true; set to false to skip the flip forward pass.
 */
class NetraEdgeModule {
    companion object {
        private const val TAG = "NetraEdge"
        private const val INPUT_SIZE = 112
        private const val LIVENESS_CLASSES = 3
        // SOTA TARGET dim for Day-2 fine-tune (Path A+: EdgeFace-XS / 2-Teacher KD).
        // Pre-trained model on disk is 128-d; this is the post-fine-tune target.
        const val SOTA_TARGET_EMBEDDING_DIM = 512
    }

    private var recognitionInterpreter: Interpreter? = null
    private var livenessInterpreter: Interpreter? = null
    private var initialized = false

    /** Embedding dim derived from the loaded recognition model's output tensor. */
    private var embeddingDim: Int = -1

    /** Liveness dim derived from the loaded liveness model's output tensor. */
    private var livenessDim: Int = LIVENESS_CLASSES

    /** Test-time augmentation: horizontal flip averaging at inference. Default ON. */
    private var useTTA: Boolean = true

    fun isInitialized(): Boolean = initialized

    /** Embedding dim of the currently loaded model (or -1 if not loaded). */
    fun getEmbeddingDim(): Int = embeddingDim

    /** Liveness output dim of the currently loaded model. */
    fun getLivenessDim(): Int = livenessDim

    /** Whether TTA (horizontal flip averaging) is enabled. */
    fun isUseTTA(): Boolean = useTTA

    /** Enable or disable TTA at runtime. */
    fun setUseTTA(enabled: Boolean) {
        useTTA = enabled
        Log.i(TAG, "TTA ${if (enabled) "enabled" else "disabled"}")
    }

    fun initialize(context: Context): Boolean {
        return try {
            val recInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "face_recognition.tflite")
            )
            // Derive output dim from the model's output tensor shape [1, N]
            val recOutShape = recInterpreter.getOutputTensor(0).shape()
            embeddingDim = if (recOutShape.size >= 2) recOutShape[1] else recOutShape.last()
            Log.i(TAG, "Recognition model output dim = $embeddingDim (target = $SOTA_TARGET_EMBEDDING_DIM)")

            val livInterpreter = Interpreter(
                FileUtil.loadMappedFile(context, "liveness_detector.tflite")
            )
            val livOutShape = livInterpreter.getOutputTensor(0).shape()
            livenessDim = if (livOutShape.size >= 2) livOutShape[1] else livOutShape.last()
            Log.i(TAG, "Liveness model output dim = $livenessDim")

            recognitionInterpreter = recInterpreter
            livenessInterpreter = livInterpreter
            initialized = true
            Log.i(TAG, "TFLite models loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite models: ${e.message}")
            false
        }
    }

    /**
     * Run face recognition inference with optional TTA.
     *
     * Input: 112×112×3 RGB image (normalized floats, 0–1)
     * Output: `embeddingDim`-dimensional L2-normalized embedding (128 for the
     * pre-trained MobileFaceNet, 512 for the Day-2 fine-tuned EdgeFace-XS).
     *
     * TTA (when enabled, default ON): runs the original input AND a horizontal
     * flip through the model, L2-normalizes each embedding, averages them,
     * then L2-normalizes the result. Empirically +0.3-0.5% LFW for free.
     */
    fun runRecognition(inputData: FloatArray): FloatArray? {
        val interpreter = recognitionInterpreter ?: run {
            Log.e(TAG, "Recognition model not initialized")
            return null
        }
        if (!initialized || embeddingDim <= 0) return null

        return try {
            // Build the 3D input tensor [1, 112, 112, 3] from flat RGB.
            val input = Array(1) { Array(INPUT_SIZE) { FloatArray(INPUT_SIZE * 3) } }
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val idx = (y * INPUT_SIZE + x) * 3
                    input[0][y][x * 3] = inputData[idx]
                    input[0][y][x * 3 + 1] = inputData[idx + 1]
                    input[0][y][x * 3 + 2] = inputData[idx + 2]
                }
            }

            // Forward pass on original input.
            val output = Array(1) { FloatArray(embeddingDim) }
            interpreter.run(input, output)
            l2NormalizeInPlace(output[0])

            if (!useTTA) return output[0]

            // TTA: forward pass on horizontal-flip.
            val flippedInput = Array(1) { Array(INPUT_SIZE) { FloatArray(INPUT_SIZE * 3) } }
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val srcX = INPUT_SIZE - 1 - x
                    val srcIdx = (y * INPUT_SIZE + srcX) * 3
                    val dstIdx = (y * INPUT_SIZE + x) * 3
                    flippedInput[0][y][dstIdx]     = inputData[srcIdx]
                    flippedInput[0][y][dstIdx + 1] = inputData[srcIdx + 1]
                    flippedInput[0][y][dstIdx + 2] = inputData[srcIdx + 2]
                }
            }
            val flippedOutput = Array(1) { FloatArray(embeddingDim) }
            interpreter.run(flippedInput, flippedOutput)
            l2NormalizeInPlace(flippedOutput[0])

            // Average the two L2-normalized embeddings, then re-normalize.
            val averaged = FloatArray(embeddingDim)
            for (i in 0 until embeddingDim) {
                averaged[i] = (output[0][i] + flippedOutput[0][i]) * 0.5f
            }
            l2NormalizeInPlace(averaged)
            averaged
        } catch (e: Exception) {
            Log.e(TAG, "Recognition inference failed: ${e.message}")
            null
        }
    }

    /** In-place L2 normalization. */
    private fun l2NormalizeInPlace(v: FloatArray) {
        var norm = 0f
        for (x in v) norm += x * x
        norm = Math.sqrt(norm.toDouble()).toFloat()
        if (norm > 0) {
            for (i in v.indices) v[i] /= norm
        }
    }

    /**
     * Run liveness detection inference.
     *
     * Input: 112×112×3 RGB image (normalized floats, 0–1)
     * Output: `livenessDim`-class probabilities (typically [real, print, screen] = 3)
     */
    fun runLiveness(inputData: FloatArray): FloatArray? {
        val interpreter = livenessInterpreter ?: run {
            Log.e(TAG, "Liveness model not initialized")
            return null
        }
        if (!initialized || livenessDim <= 0) return null

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

            val output = Array(1) { FloatArray(livenessDim) }
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
        require(a.size == b.size) { "Dimension mismatch: a=${a.size} b=${b.size}" }
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
