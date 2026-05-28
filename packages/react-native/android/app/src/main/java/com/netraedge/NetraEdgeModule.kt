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

    fun close() {
        recognitionInterpreter?.close()
        livenessInterpreter?.close()
        recognitionInterpreter = null
        livenessInterpreter = null
        initialized = false
        Log.i(TAG, "NetraEdge module closed")
    }
}
