package com.netraedge;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * NetraEdgeModule — React Native bridge for TFLite inference.
 *
 * Provides native methods for:
 * - Loading TFLite models from app assets
 * - Running face recognition inference (128-d embedding)
 * - Running liveness detection inference (3-class softmax)
 *
 * This module is loaded automatically by React Native's NativeModules system.
 */
public class NetraEdgeModule {
    private static final String TAG = "NetraEdge";

    private Interpreter recognitionInterpreter;
    private Interpreter livenessInterpreter;
    private boolean isInitialized = false;

    private static final int INPUT_SIZE = 112;
    private static final int EMBEDDING_DIM = 128;
    private static final int LIVENESS_CLASSES = 3;

    /**
     * Initialize the module with TFLite models.
     *
     * @param context Android context (for accessing assets)
     * @return true if initialization succeeded
     */
    public boolean initialize(Context context) {
        try {
            recognitionInterpreter = new Interpreter(
                FileUtil.loadMappedFile(context, "face_recognition.tflite")
            );
            livenessInterpreter = new Interpreter(
                FileUtil.loadMappedFile(context, "liveness_detector.tflite")
            );
            isInitialized = true;
            Log.i(TAG, "TFLite models loaded successfully");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load TFLite models: " + e.getMessage());
            return false;
        }
    }

    /**
     * Check if models are loaded and ready.
     */
    public boolean isInitialized() {
        return isInitialized;
    }

    /**
     * Run face recognition on a face crop.
     *
     * Input: 112x112x3 RGB image (normalized to 0-1)
     * Output: 128-dimensional L2-normalized embedding
     *
     * @param inputData Raw pixel data (112 * 112 * 3 floats)
     * @return 128-d embedding array, or null on failure
     */
    public float[] runRecognition(float[] inputData) {
        if (!isInitialized || recognitionInterpreter == null) {
            Log.e(TAG, "Recognition model not initialized");
            return null;
        }

        try {
            // Prepare input buffer
            float[][][] input = new float[1][INPUT_SIZE][INPUT_SIZE * 3];
            for (int y = 0; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    int idx = (y * INPUT_SIZE + x) * 3;
                    input[0][y][x * 3] = inputData[idx];
                    input[0][y][x * 3 + 1] = inputData[idx + 1];
                    input[0][y][x * 3 + 2] = inputData[idx + 2];
                }
            }

            // Run inference
            float[][] output = new float[1][EMBEDDING_DIM];
            recognitionInterpreter.run(input, output);

            // L2 normalize
            float[] embedding = output[0];
            float norm = 0;
            for (float v : embedding) norm += v * v;
            norm = (float) Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < embedding.length; i++) {
                    embedding[i] /= norm;
                }
            }

            return embedding;
        } catch (Exception e) {
            Log.e(TAG, "Recognition inference failed: " + e.getMessage());
            return null;
        }
    }

    /**
     * Run liveness detection on a face crop.
     *
     * Input: 112x112x3 RGB image (normalized to 0-1)
     * Output: 3-class probabilities [real, print, screen]
     *
     * @param inputData Raw pixel data (112 * 112 * 3 floats)
     * @return 3-element probability array, or null on failure
     */
    public float[] runLiveness(float[] inputData) {
        if (!isInitialized || livenessInterpreter == null) {
            Log.e(TAG, "Liveness model not initialized");
            return null;
        }

        try {
            float[][][] input = new float[1][INPUT_SIZE][INPUT_SIZE * 3];
            for (int y = 0; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    int idx = (y * INPUT_SIZE + x) * 3;
                    input[0][y][x * 3] = inputData[idx];
                    input[0][y][x * 3 + 1] = inputData[idx + 1];
                    input[0][y][x * 3 + 2] = inputData[idx + 2];
                }
            }

            float[][] output = new float[1][LIVENESS_CLASSES];
            livenessInterpreter.run(input, output);
            return output[0];
        } catch (Exception e) {
            Log.e(TAG, "Liveness inference failed: " + e.getMessage());
            return null;
        }
    }

    /**
     * Compute cosine similarity between two embeddings.
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Dimension mismatch");

        float dot = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return Math.max(-1, Math.min(1, dot));
    }

    /**
     * Release TFLite resources.
     */
    public void close() {
        if (recognitionInterpreter != null) {
            recognitionInterpreter.close();
            recognitionInterpreter = null;
        }
        if (livenessInterpreter != null) {
            livenessInterpreter.close();
            livenessInterpreter = null;
        }
        isInitialized = false;
        Log.i(TAG, "NetraEdge module closed");
    }
}
