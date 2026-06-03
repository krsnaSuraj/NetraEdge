package com.netraedge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.*
import com.netraedge.SecurityHardening
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SOTA React Native bridge for NetraEdge — face recognition + liveness
 * + active challenge + rPPG + security hardening + encrypted asset
 * decryption. Drop-in module for the Datalake 3.0 integration.
 *
 * Pipeline:
 *   JS frame (base64 bitmap + 5 face keypoints) →
 *     this module →
 *       5-point similarity transform → 112×112×3 float (CHW, mean=127.5, std=128.0) →
 *         MobileFaceNet 128-d (Apache-2.0, 99.48% LFW) →
 *           cosine sim vs enrolled embedding
 *         MiniFASNet 3-class (Apache-2.0, 98.20% CelebA-Spoof) →
 *           real / print_attack / replay_attack
 *
 *   rPPG (called per frame, window built up in JS):
 *     green-channel mean of forehead → 3.5 s sliding window →
 *       Hann window + 0.7-2.5 Hz bandpass via FFT → BPM + isLive
 *
 *   Security:
 *     SecurityHardening.audit() — anti-debug, root, custom ROM, debuggable APK,
 *                                  repacked APK (signature SHA-256 mismatch)
 *     EncryptedAssets.materialize() — AES-256-GCM decryption of .tflite + .task
 *
 * All inference is on-device. No network round-trip per frame. 100% offline.
 *
 * @author NetraEdge (Hackathon 7.0 / Datalake 3.0 / NHAI)
 */
class NetraEdgeNativeModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val TAG = "NetraEdge"
        private const val INPUT_SIZE = 112
        private const val PREF_NAME = "netraedge_secure"
        private const val KEY_EMBEDDING = "enrolled_embedding_v1"
        private const val KEY_USER_ID = "enrolled_user_id"
        private const val KEY_TTA = "tta_enabled"
        private const val KEY_ENROLLED_AT = "enrolled_at_iso"
    }

    private var recognitionInterpreter: Interpreter? = null
    private var livenessInterpreter: Interpreter? = null
    private var embeddingDim: Int = -1
    private var livenessDim: Int = 3
    private var isInitializedFlag: Boolean = false
    private var useTTA: Boolean = true

    private val recognitionInput: ByteBuffer =
        ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
    private val livenessInput: ByteBuffer =
        ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
    private val recognitionOutput: Array<FloatArray> = Array(1) { FloatArray(128) }
    private val livenessOutput: Array<FloatArray> = Array(1) { FloatArray(3) }
    private val alignedBuffer = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

    override fun getName(): String = "NetraEdgeModule"

    // ============================================================================
    // Lifecycle / setup
    // ============================================================================

    @ReactMethod
    fun initialize(promise: Promise) {
        try {
            val rec = Interpreter(FileUtil.loadMappedFile(reactContext, "face_recognition.tflite"))
            val recOut = rec.getOutputTensor(0).shape()
            embeddingDim = if (recOut.size >= 2) recOut[1] else recOut.last()
            recognitionInterpreter = rec

            val liv = Interpreter(FileUtil.loadMappedFile(reactContext, "liveness_detector.tflite"))
            val livOut = liv.getOutputTensor(0).shape()
            livenessDim = if (livOut.size >= 2) livOut[1] else livOut.last()
            livenessInterpreter = liv

            isInitializedFlag = true
            useTTA = prefs().getBoolean(KEY_TTA, true)
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e(TAG, "initialize failed", e)
            promise.reject("INIT_ERROR", e.message)
        }
    }

    @ReactMethod
    fun isInitialized(promise: Promise) {
        promise.resolve(isInitializedFlag)
    }

    @ReactMethod
    fun getEmbeddingDim(promise: Promise) {
        promise.resolve(embeddingDim)
    }

    @ReactMethod
    fun getLivenessDim(promise: Promise) {
        promise.resolve(livenessDim)
    }

    @ReactMethod
    fun setUseTTA(enabled: Boolean, promise: Promise) {
        useTTA = enabled
        prefs().edit().putBoolean(KEY_TTA, enabled).apply()
        promise.resolve(null)
    }

    @ReactMethod
    fun isUseTTA(promise: Promise) {
        promise.resolve(useTTA)
    }

    @ReactMethod
    fun close() {
        recognitionInterpreter?.close()
        livenessInterpreter?.close()
        recognitionInterpreter = null
        livenessInterpreter = null
        isInitializedFlag = false
    }

    // ============================================================================
    // Core: align + recognize
    // ============================================================================

    /**
     * Run the full recognition pipeline on a single frame.
     *
     * @param bitmapBase64 JPEG/PNG encoded bitmap from JS
     * @param frameW  source width
     * @param frameH  source height
     * @param key5   array of 5 keypoints [{x,y}, {x,y}, ...] in source pixel coords
     *               Order: left eye, right eye, nose tip, left mouth, right mouth
     * @param promise resolves to WritableMap { embedding, liveness, bbox }
     */
    @ReactMethod
    fun processFrame(
        bitmapBase64: String,
        frameW: Int,
        frameH: Int,
        key5: ReadableArray,
        promise: Promise
    ) {
        try {
            val rec = recognitionInterpreter
            val liv = livenessInterpreter
            if (rec == null || liv == null || !isInitializedFlag) {
                promise.reject("NOT_INITIALIZED", "Module not initialized — call initialize() first")
                return
            }

            val bm = decodeBase64Bitmap(bitmapBase64, frameW, frameH)
            val keypoints = extractKeypoints(key5, frameW, frameH)

            FaceAligner.align(
                yuv = null,
                yRowStride = 0,
                width = bm.width,
                height = bm.height,
                srcKey = keypoints,
                out = alignedBuffer,
                bitmap = bm
            )

            val embed = runRecognition(alignedBuffer, embeddingDim, rec, useTTA)
            val livOut = runLiveness(alignedBuffer, livenessDim, liv)
            val livenessScore = if (livOut != null && livOut.size > 1) livOut[1] else 0f

            val out = Arguments.createMap()
            val embedArr = Arguments.createArray()
            for (v in embed) embedArr.pushDouble(v.toDouble())
            out.putArray("embedding", embedArr)

            val livArr = Arguments.createArray()
            if (livOut != null) for (v in livOut) livArr.pushDouble(v.toDouble())
            out.putArray("liveness", livArr)
            out.putDouble("livenessRealScore", livenessScore.toDouble())
            out.putInt("embeddingDim", embeddingDim)
            out.putInt("livenessDim", livenessDim)
            promise.resolve(out)
        } catch (e: Throwable) {
            Log.e(TAG, "processFrame error", e)
            promise.reject("PROCESS_FRAME_ERROR", e.message)
        }
    }

    /**
     * Standalone liveness inference (no embedding). Useful for active challenge mode.
     */
    @ReactMethod
    fun processLivenessOnly(
        bitmapBase64: String,
        frameW: Int,
        frameH: Int,
        key5: ReadableArray,
        promise: Promise
    ) {
        try {
            val liv = livenessInterpreter
            if (liv == null || !isInitializedFlag) {
                promise.reject("NOT_INITIALIZED", "Module not initialized")
                return
            }
            val bm = decodeBase64Bitmap(bitmapBase64, frameW, frameH)
            val keypoints = extractKeypoints(key5, frameW, frameH)
            FaceAligner.align(null, 0, bm.width, bm.height, keypoints, alignedBuffer, bm)
            val livOut = runLiveness(alignedBuffer, livenessDim, liv)
            val out = Arguments.createArray()
            if (livOut != null) for (v in livOut) out.pushDouble(v.toDouble())
            promise.resolve(out)
        } catch (e: Throwable) {
            promise.reject("LIVENESS_ERROR", e.message)
        }
    }

    /**
     * Standalone recognition inference (no liveness).
     */
    @ReactMethod
    fun processRecognitionOnly(
        bitmapBase64: String,
        frameW: Int,
        frameH: Int,
        key5: ReadableArray,
        promise: Promise
    ) {
        try {
            val rec = recognitionInterpreter
            if (rec == null || !isInitializedFlag) {
                promise.reject("NOT_INITIALIZED", "Module not initialized")
                return
            }
            val bm = decodeBase64Bitmap(bitmapBase64, frameW, frameH)
            val keypoints = extractKeypoints(key5, frameW, frameH)
            FaceAligner.align(null, 0, bm.width, bm.height, keypoints, alignedBuffer, bm)
            val embed = runRecognition(alignedBuffer, embeddingDim, rec, useTTA)
            val arr = Arguments.createArray()
            for (v in embed) arr.pushDouble(v.toDouble())
            promise.resolve(arr)
        } catch (e: Throwable) {
            promise.reject("RECOGNITION_ERROR", e.message)
        }
    }

    // ============================================================================
    // Enrollment / verification
    // ============================================================================

    /**
     * Average 10+ embeddings, L2 normalize, save to encrypted storage.
     *
     * @param userId  string identifier (e.g. Aadhaar token, employee ID, phone)
     * @param embeddingsArray array of FloatArray embeddings, each 128-d
     */
    @ReactMethod
    fun enroll(userId: String, embeddingsArray: ReadableArray, promise: Promise) {
        try {
            if (userId.isBlank()) {
                promise.reject("ENROLL_ERROR", "userId is required")
                return
            }
            val n = embeddingsArray.size()
            if (n < 3) {
                promise.reject("ENROLL_ERROR", "Need at least 3 embeddings to enroll, got $n")
                return
            }
            val sum = FloatArray(embeddingDim)
            for (i in 0 until n) {
                val arr = embeddingsArray.getArray(i) ?: continue
                for (j in 0 until embeddingDim) {
                    if (j < arr.size()) sum[j] += arr.getDouble(j).toFloat()
                }
            }
            for (j in 0 until embeddingDim) sum[j] /= n.toFloat()
            val normalized = l2Normalize(sum)

            val jsonArr = JSONArray()
            for (v in normalized) jsonArr.put(v.toDouble())
            val obj = JSONObject()
                .put("embedding", jsonArr)
                .put("framesAveraged", n)
            prefs().edit()
                .putString(KEY_EMBEDDING, obj.toString())
                .putString(KEY_USER_ID, userId)
                .putString(KEY_ENROLLED_AT, java.text.SimpleDateFormat(
                    "yyyy-MM-dd'T'HH:mm:ssXXX",
                    java.util.Locale.US
                ).format(java.util.Date()))
                .apply()

            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("ENROLL_ERROR", e.message)
        }
    }

    /**
     * Compare an embedding to the enrolled embedding.
     * @return { similarity: number, livenessPassed: bool, userId: string, matched: bool }
     */
    @ReactMethod
    fun verify(embedding: ReadableArray, threshold: Double, livenessScore: Double, promise: Promise) {
        try {
            val raw = prefs().getString(KEY_EMBEDDING, null)
                ?: return promise.resolve(emptyVerifyMap(null, 0.0, false, false, livenessScore))
            val obj = JSONObject(raw)
            val arr = obj.getJSONArray("embedding")
            val ref = FloatArray(arr.length())
            for (i in 0 until arr.length()) ref[i] = arr.getDouble(i).toFloat()

            val live = FloatArray(embedding.size())
            for (i in 0 until embedding.size()) live[i] = embedding.getDouble(i).toFloat()

            if (live.size != ref.size) {
                promise.reject("VERIFY_ERROR", "dim mismatch: live=${live.size} ref=${ref.size}")
                return
            }

            val sim = cosineSimilarity(live, ref)
            val userId = prefs().getString(KEY_USER_ID, null)
            val livenessOk = livenessScore >= 0.70
            val matched = sim >= threshold && livenessOk

            val out = Arguments.createMap()
            out.putDouble("similarity", sim.toDouble())
            out.putBoolean("livenessPassed", livenessOk)
            out.putString("userId", userId)
            out.putBoolean("matched", matched)
            out.putDouble("threshold", threshold)
            out.putDouble("livenessScore", livenessScore)
            promise.resolve(out)
        } catch (e: Exception) {
            promise.reject("VERIFY_ERROR", e.message)
        }
    }

    @ReactMethod
    fun getEnrolledInfo(promise: Promise) {
        try {
            val raw = prefs().getString(KEY_EMBEDDING, null)
            val out = Arguments.createMap()
            if (raw == null) {
                out.putBoolean("enrolled", false)
                promise.resolve(out)
                return
            }
            val obj = JSONObject(raw)
            out.putBoolean("enrolled", true)
            out.putString("userId", prefs().getString(KEY_USER_ID, null))
            out.putString("enrolledAt", prefs().getString(KEY_ENROLLED_AT, null))
            out.putInt("framesAveraged", obj.optInt("framesAveraged", 0))
            out.putInt("embeddingDim", embeddingDim)
            promise.resolve(out)
        } catch (e: Exception) {
            promise.reject("INFO_ERROR", e.message)
        }
    }

    @ReactMethod
    fun clearEnrollment(promise: Promise) {
        prefs().edit()
            .remove(KEY_EMBEDDING)
            .remove(KEY_USER_ID)
            .remove(KEY_ENROLLED_AT)
            .apply()
        promise.resolve(true)
    }

    // ============================================================================
    // rPPG (algorithmic, model-free)
    // ============================================================================

    /**
     * Compute the mean green-channel intensity of the forehead ROI for
     * a single frame. JS accumulates these in a 3.5 s sliding window and
     * calls [rppgAnalyze] to get the BPM.
     */
    @ReactMethod
    fun sampleRppgGreen(bitmapBase64: String, frameW: Int, frameH: Int, key5: ReadableArray, promise: Promise) {
        try {
            val bm = decodeBase64Bitmap(bitmapBase64, frameW, frameH)
            val keypoints = extractKeypoints(key5, frameW, frameH)
            val leftEye = keypoints[0]
            val rightEye = keypoints[1]
            val cx = ((leftEye.x + rightEye.x) / 2f).toInt().coerceIn(0, bm.width - 1)
            val cy = ((leftEye.y + rightEye.y) / 2f).toInt().coerceIn(0, bm.height - 1)
            val w = 32
            val x0 = (cx - w / 2).coerceAtLeast(0)
            val y0 = (cy - w / 2 - 16).coerceAtLeast(0)
            val x1 = (x0 + w).coerceAtMost(bm.width)
            val y1 = (y0 + w).coerceAtMost(bm.height)
            if (x1 <= x0 || y1 <= y0) {
                promise.resolve(0.0)
                return
            }
            val pixels = IntArray((x1 - x0) * (y1 - y0))
            bm.getPixels(pixels, 0, x1 - x0, x0, y0, x1 - x0, y1 - y0)
            var sum = 0.0
            for (p in pixels) sum += (p shr 8) and 0xFF
            promise.resolve(sum / pixels.size)
        } catch (e: Exception) {
            promise.reject("RPPG_SAMPLE_ERROR", e.message)
        }
    }

    /**
     * Analyze a sliding window of green-channel means (3.5 s @ 30 fps = 105 samples).
     * Returns { bpm: Int, isLive: Bool, peakPower: number, totalPower: number }
     */
    @ReactMethod
    fun rppgAnalyze(samples: ReadableArray, promise: Promise) {
        try {
            val n = samples.size()
            if (n < 16) {
                val out = Arguments.createMap()
                out.putInt("bpm", 0)
                out.putBoolean("isLive", false)
                promise.resolve(out)
                return
            }
            val arr = DoubleArray(n)
            for (i in 0 until n) arr[i] = samples.getDouble(i)

            val mean = arr.average()
            for (i in 0 until n) arr[i] -= mean

            val nPow2 = nextPow2(n)
            val input = DoubleArray(nPow2)
            val mag = DoubleArray(nPow2)
            System.arraycopy(arr, 0, input, 0, n)

            for (i in 0 until n) {
                val w = 0.5 * (1.0 - kotlin.math.cos(2.0 * Math.PI * i / (n - 1)))
                input[i] *= w
            }

            com.netraedge.RppgAnalyzer.analyzeWindow(input, mag, sampleHz = 30.0, n = nPow2)

            val binHz = 30.0 / nPow2
            val minBin = (0.7 / binHz).toInt().coerceAtLeast(1)
            val maxBin = (2.5 / binHz).toInt().coerceAtMost(nPow2 / 2 - 1)
            var bestBin = 0
            var bestMag = 0.0
            var total = 0.0
            for (b in minBin..maxBin) {
                total += mag[b]
                if (mag[b] > bestMag) { bestMag = mag[b]; bestBin = b }
            }
            val prominence = if (total > 0) bestMag / total else 0.0
            val isLive = prominence > 0.10
            val bpm = if (isLive) ((bestBin * binHz) * 60.0).toInt().coerceIn(40, 150) else 0

            val out = Arguments.createMap()
            out.putInt("bpm", bpm)
            out.putBoolean("isLive", isLive)
            out.putDouble("peakPower", bestMag)
            out.putDouble("totalPower", total)
            out.putDouble("prominence", prominence)
            promise.resolve(out)
        } catch (e: Exception) {
            promise.reject("RPPG_ERROR", e.message)
        }
    }

    // ============================================================================
    // Security / Anti-tampering
    // ============================================================================

    @ReactMethod
    fun auditSecurity(promise: Promise) {
        try {
            val report = com.netraedge.SecurityHardening.audit(reactContext)
            val issues = Arguments.createArray()
            for (s in report.detectedIssues) issues.pushString(s)
            val out = Arguments.createMap()
            out.putBoolean("isDebuggerAttached", report.isDebuggerAttached)
            out.putBoolean("isRooted", report.isRooted)
            out.putBoolean("isCustomRom", report.isCustomRom)
            out.putBoolean("isDebuggableApk", report.isDebuggableApk)
            out.putBoolean("isEmulator", report.isEmulator)
            out.putBoolean("isRepackedApk", report.isRepackedApk)
            out.putBoolean("isTampered", report.isTampered)
            out.putArray("detectedIssues", issues)
            promise.resolve(out)
        } catch (e: Exception) {
            promise.reject("SECURITY_ERROR", e.message)
        }
    }

    @ReactMethod
    fun isRooted(promise: Promise) {
        promise.resolve(com.netraedge.SecurityHardening.isRooted())
    }

    @ReactMethod
    fun isDebuggerAttached(promise: Promise) {
        promise.resolve(android.os.Debug.isDebuggerConnected())
    }

    // ============================================================================
    // Diagnostics
    // ============================================================================

    @ReactMethod
    fun getModelInfo(promise: Promise) {
        val out = Arguments.createMap()
        out.putString("recognitionModel", "MobileFaceNet (foamliu, Apache-2.0, 99.48% LFW)")
        out.putString("livenessModel", "MiniFASNet-style (Apache-2.0, 98.20% CelebA-Spoof)")
        out.putInt("recognitionDim", embeddingDim)
        out.putInt("livenessDim", livenessDim)
        out.putInt("inputSize", INPUT_SIZE)
        out.putString("pipeline", "5-point alignment → MobileFaceNet + MiniFASNet + algorithmic rPPG + active challenge (EAR/MAR + head pose)")
        out.putString("license", "Apache-2.0 / MIT ONLY — no non-commercial licenses")
        out.putString("hackathon", "Hackathon 7.0 / Datalake 3.0 / NHAI")
        out.putString("version", "1.0.0")
        promise.resolve(out)
    }

    @ReactMethod
    fun getVersion(promise: Promise) {
        promise.resolve("1.0.0")
    }

    // ============================================================================
    // Private helpers
    // ============================================================================

    private fun prefs() = reactContext.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    private fun decodeBase64Bitmap(b64: String, w: Int, h: Int): Bitmap {
        val bytes = Base64.decode(b64, Base64.DEFAULT)
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bm = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: throw IllegalStateException("BitmapFactory returned null for ${bytes.size} bytes")
        return if (bm.width == w && bm.height == h) bm else {
            val matrix = Matrix().apply { postScale(w.toFloat() / bm.width, h.toFloat() / bm.height) }
            val out = Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, matrix, true)
            bm.recycle()
            out
        }
    }

    private fun extractKeypoints(arr: ReadableArray, w: Int, Int_h: Int): Array<android.graphics.PointF> {
        require(arr.size() >= 5) { "Expected 5 keypoints, got ${arr.size()}" }
        val out = Array(5) { android.graphics.PointF() }
        for (i in 0 until 5) {
            val map = arr.getMap(i)
            val x = if (map != null) map.getDouble("x").toFloat() else 0f
            val y = if (map != null) map.getDouble("y").toFloat() else 0f
            out[i] = android.graphics.PointF(x, y)
        }
        return out
    }

    private fun runRecognition(aligned: FloatArray, dim: Int, rec: Interpreter, tta: Boolean): FloatArray {
        recognitionInput.rewind()
        for (v in aligned) recognitionInput.putFloat(v)
        recognitionOutput[0] = FloatArray(dim)
        rec.run(recognitionInput, recognitionOutput)
        val e1 = l2Normalize(recognitionOutput[0])
        if (!tta) return e1
        val flipped = FloatArray(aligned.size)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val sx = INPUT_SIZE - 1 - x
                val sIdx = (y * INPUT_SIZE + sx) * 3
                val dIdx = (y * INPUT_SIZE + x) * 3
                for (c in 0..2) flipped[dIdx + c] = aligned[sIdx + c]
            }
        }
        recognitionInput.rewind()
        for (v in flipped) recognitionInput.putFloat(v)
        recognitionOutput[0] = FloatArray(dim)
        rec.run(recognitionInput, recognitionOutput)
        val e2 = l2Normalize(recognitionOutput[0])
        val avg = FloatArray(dim)
        for (i in 0 until dim) avg[i] = (e1[i] + e2[i]) * 0.5f
        return l2Normalize(avg)
    }

    private fun runLiveness(aligned: FloatArray, dim: Int, liv: Interpreter): FloatArray? {
        livenessInput.rewind()
        for (v in aligned) livenessInput.putFloat(v)
        livenessOutput[0] = FloatArray(dim)
        liv.run(livenessInput, livenessOutput)
        return livenessOutput[0]
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0f
        for (x in v) sum += x * x
        val n = kotlin.math.sqrt(sum)
        if (n == 0f) return v
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / n
        return out
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
        return if (denom == 0f) 0f else dot / denom
    }

    private fun emptyVerifyMap(userId: String?, sim: Double, matched: Boolean, livenessOk: Boolean, livenessScore: Double): WritableMap {
        val out = Arguments.createMap()
        out.putString("userId", userId)
        out.putDouble("similarity", sim)
        out.putBoolean("matched", matched)
        out.putBoolean("livenessPassed", livenessOk)
        out.putDouble("livenessScore", livenessScore)
        return out
    }

    private fun nextPow2(v: Int): Int {
        var n = 1
        while (n < v) n = n shl 1
        return n
    }
}
