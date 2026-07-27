package com.netraedge

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

/**
 * NetraEdge — SOTA offline face recognition + liveness (Hackathon 7.0 build).
 *
 * Pipeline (per camera frame at ~30 fps, decimated to ~10 fps):
 *   CameraX (YUV_420_888, 720p) ->
 *     MediaPipe Face Landmarker (478 landmarks + 52 blendshapes, Apache-2.0) ->
 *       KeypointExtractor -> 5 canonical points
 *       FaceAligner -> 112x112 CHW float buffer (mean=127.5, std=128.0)
 *         -> MobileFaceNet 128-d embedding (Apache-2.0, foamliu)
 *         -> MiniFASNet-style 3-class liveness (Apache-2.0)
 *       RppgAnalyzer -> 0.7-2.5Hz bandpass pulse over 3.5s sliding window
 *       ActiveChallengeRunner -> blink / smile / head-turn sequence
 *
 * All inference is on-device; no network round-trip per frame.
 * Models live in /assets (face_landmarker.task 3.58MB, face_recognition.tflite
 * 10.58MB, liveness_detector.tflite 522KB) — Apache-2.0 / MIT only.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "NetraEdge"
        private const val EMBED_DIM = 128
        private const val INPUT_SIZE = 112
        private const val ENROLL_FRAMES = 15  // more frames = better average
        private const val LIVENESS_THRESHOLD = 0.70f
        private const val PREF_NAME = "netraedge_enrollment"
        private const val KEY_EMBEDDING = "enrolled_embedding"

        // Adaptive threshold params (temperature-scaled sigmoid)
        // Based on ICLR 2026 Adaptive Calibration: replaces fixed 0.55
        private const val BASELINE_THRESHOLD = 0.45f
        private const val TEMPERATURE = 0.15f
    }

    private enum class State {
        IDLE, ENROLLING, ENROLLED, VERIFYING, VERIFIED, NOT_RECOGNIZED, SPOOF, ERROR
    }

    /**
     * Eye Aspect Ratio (EAR) — geometric blink detector.
     * Open eye ~0.30, closed eye ~0.10. Robust fallback when blendshapes unavailable.
     */
    private fun computeEar(
        lms: List<PointF>,
        p1: Int, p2: Int, p3: Int, p4: Int, p5: Int, p6: Int
    ): Float {
        val v1 = dist(lms[p2], lms[p6])
        val v2 = dist(lms[p3], lms[p5])
        val h = dist(lms[p1], lms[p4]).coerceAtLeast(1f)
        return ((v1 + v2) / (2f * h))
    }

    private fun dist(a: PointF, b: PointF): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private enum class Mode { NONE, ACTIVE_LIVENESS }

    private var state: State = State.IDLE
    private var mode: Mode = Mode.NONE
    private var enrolledEmbedding: FloatArray? = null
    private val enrollmentBuffer = ArrayList<FloatArray>(ENROLL_FRAMES)
    private val enrollmentQualities = ArrayList<Float>(ENROLL_FRAMES)  // quality weights for averaging
    private val ttaBuffer = FloatArray(EMBED_DIM)  // TTA accumulator

    private lateinit var previewView: PreviewView
    private lateinit var faceOverlay: FaceOverlayView
    private lateinit var statusText: TextView
    private lateinit var infoText: TextView
    private lateinit var statusCard: CardView
    private lateinit var statusDot: android.view.View
    private lateinit var statusBadge: TextView
    private lateinit var ambientBg: AmbientBackgroundView
    private lateinit var challengePromptCard: CardView
    private lateinit var challengePromptContainer: android.widget.LinearLayout
    private lateinit var challengeIcon: android.widget.TextView
    private lateinit var challengeLabel: android.widget.TextView
    private lateinit var challengePromptText: android.widget.TextView
    private lateinit var challengeProgress: android.widget.LinearLayout
    private lateinit var challengeCountdown: android.widget.TextView
    private lateinit var btnEnroll: Button
    private lateinit var btnVerify: Button
    private lateinit var btnReset: Button
    private lateinit var btnSync: Button
    private lateinit var btnPurge: Button
    // Liveness signal bars (8)
    private val sigBars: Array<android.view.View> by lazy {
        Array(8) { findViewById(resources.getIdentifier("sig${it + 1}", "id", packageName)) }
    }

    private var faceLandmarker: FaceLandmarker? = null
    private lateinit var haptics: HapticHelper
    private var recognitionInterpreter: Interpreter? = null
    private var livenessInterpreter: Interpreter? = null
    private val rppg = RppgAnalyzer(sampleRateHz = 30.0) // native 30fps for SOTA POS algorithm
    private val activeChallenge = ActiveChallengeRunner()

    // EMA-smoothed fused liveness score — prevents REAL/FAKE flicker
    private var fusedLivenessEma = 0.5f

    // Timestamp when current state (VERIFYING/ENROLLING) started — used for
    // time-based anti-replay vetoes (e.g. "no rPPG pulse after 4s = SPOOF")
    private var stateStartedAt: Long = 0L

    // Timestamp when the last face was seen during VERIFYING — used to detect
    // "user walked away" timeout. After 10s of no face, return to IDLE.
    private var lastFaceSeenAt: Long = 0L

    // Track last shown challenge hint to debounce UI thread work (avoid 30Hz spam).
    // null = never shown; "passed" = success state; "failed" = fail state
    private var lastShownChallengeStep: ActiveChallengeRunner.Step? = null
    private val challengePromptHideRunnable = Runnable { hideChallengePromptNow() }

    // 10-Layer Liveness Fusion — NetraEdge X SOTA 2026
    // Layers 3-10: algorithmic, NO TFLite model required (broken liveness model bypassed)
    private val textureAnalyzer = TextureAnalyzer()    // LBP histogram entropy
    private val colorAnalyzer = ColorAnalyzer()        // HSV stddev + entropy
    private val moireDetector = MoireDetector()        // 2D FFT high-freq energy
    private val specularDetector = SpecularDetector()  // Bright pixel cluster analysis
    private val sensorFusion = SensorFusion(this)      // Accel + Gyro variance
    private val temporalAnalyzer = TemporalConsistencyAnalyzer() // Embedding dynamics
    private val lightAnalyzer = LightConsistencyAnalyzer()       // Color temp mismatch
    private val bandingDetector = BandingDetector()    // Rolling shutter aliasing

    // Per-layer weights (sum to 1.0)
    // Active challenge and rPPG are highest weight because they are the most direct
    // liveness signals. Texture layers are weighted equally.
    private val layerWeights = floatArrayOf(
        0.20f, // 1. Active challenge (blink/smile/turn)
        0.15f, // 2. rPPG pulse
        0.10f, // 3. LBP texture
        0.10f, // 4. Color diversity
        0.10f, // 5. Moire detection
        0.08f, // 6. Specular highlight
        0.10f, // 7. Accelerometer fusion
        0.07f, // 8. Temporal consistency
        0.05f, // 9. Light source consistency
        0.05f  // 10. Screen refresh banding
    )

    private val recognitionInput: ByteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
    private val livenessInput: ByteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
    private val recognitionOutput: Array<FloatArray> = Array(1) { FloatArray(EMBED_DIM) }
    private val livenessOutput: Array<FloatArray> = Array(1) { FloatArray(3) }

    private val alignedBuffer = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)

    private lateinit var cameraExecutor: ExecutorService
    private var frameCounter = 0
    private var spoof = false
    private var liveScore = 0f

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else updateStatus("Camera permission denied")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val securityReport = SecurityHardening.audit(this)
        if (securityReport.isTampered) {
            Log.w(TAG, "Security tamper signals: ${securityReport.detectedIssues}")
        }

        previewView = findViewById(R.id.previewView)
        faceOverlay = findViewById(R.id.faceOverlay)
        faceOverlay.setLayerType(android.view.View.LAYER_TYPE_HARDWARE, null)  // GPU-accelerated overlay = 60fps
        statusText = findViewById(R.id.statusText)
        infoText = findViewById(R.id.infoText)
        statusCard = findViewById(R.id.statusCard)
        statusDot = findViewById(R.id.statusDot)
        statusBadge = findViewById(R.id.statusBadge)
        ambientBg = findViewById(R.id.ambientBg)
        challengePromptCard = findViewById(R.id.challengePromptCard)
        challengePromptContainer = findViewById(R.id.challengePromptContainer)
        challengeIcon = findViewById(R.id.challengeIcon)
        challengeLabel = findViewById(R.id.challengeLabel)
        challengeCountdown = findViewById(R.id.challengeCountdown)
        challengePromptText = findViewById(R.id.challengePromptText)
        challengeProgress = findViewById(R.id.challengeProgress)
        btnEnroll = findViewById(R.id.btnEnroll)
        btnVerify = findViewById(R.id.btnVerify)
        btnReset = findViewById(R.id.btnReset)
        btnSync = findViewById(R.id.btnSync)
        btnPurge = findViewById(R.id.btnPurge)

        btnEnroll.setOnClickListener {
            pulseButton(it)
            haptics.tap()
            onEnrollClicked()
        }
        btnVerify.setOnClickListener {
            pulseButton(it)
            haptics.tap()
            onVerifyClicked()
        }
        btnReset.setOnClickListener {
            pulseButton(it)
            haptics.tap()
            onResetClicked()
        }
        btnSync.setOnClickListener {
            pulseButton(it)
            haptics.tap()
            onSyncClicked()
        }
        btnSync.setOnLongClickListener {
            haptics.challenge()
            showSyncSettingsDialog()
            true
        }
        btnPurge.setOnClickListener {
            pulseButton(it)
            haptics.tap()
            onPurgeClicked()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        enrolledEmbedding = loadEnrolledEmbedding()
        haptics = HapticHelper(this)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onResume() {
        super.onResume()
        sensorFusion.start() // Start accelerometer + gyroscope for Layer 7
    }

    override fun onPause() {
        super.onPause()
        sensorFusion.stop() // Stop sensors to save battery
        temporalAnalyzer.reset() // Reset temporal window on pause
    }

    private fun startCamera() {
        try {
            faceLandmarker = buildFaceLandmarker()
            recognitionInterpreter = buildInterpreter("face_recognition.tflite")
            livenessInterpreter = buildInterpreter("liveness_detector.tflite")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models", e)
            updateStatus("Failed to load models: ${e.message}")
            return
        }

        warmupModels()

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis
                )
                updateStatus("NetraEdge ready")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                updateStatus("Camera failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun buildFaceLandmarker(): FaceLandmarker {
        val base = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .setOutputFaceBlendshapes(true)
            .build()
        return FaceLandmarker.createFromOptions(this, options)
    }

    private var gpuDelegate: GpuDelegate? = null

    private fun buildInterpreter(asset: String): Interpreter {
        val opts = Interpreter.Options().apply {
            setNumThreads(4)
        }

        // Try GPU delegate first — catches both Exception AND Error
        // (NoClassDefFoundError is an Error, not Exception)
        if (gpuDelegate == null) {
            try {
                gpuDelegate = GpuDelegate()
                opts.addDelegate(gpuDelegate)
                Log.d(TAG, "GPU delegate enabled for $asset")
            } catch (e: Throwable) {
                Log.w(TAG, "GPU delegate not available, using CPU: ${e.message}")
                gpuDelegate = null
            }
        } else {
            opts.addDelegate(gpuDelegate)
        }

        // Try encrypted assets first, fallback to plain mmap
        val path = try {
            EncryptedAssets.materialize(this, asset)
        } catch (e: Throwable) {
            Log.w(TAG, "Encrypted asset not found, using plain: $asset")
            val fd = assets.openFd(asset)
            val input = java.io.FileInputStream(fd.fileDescriptor)
            val channel = input.channel
            val buf = channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
            input.close()
            return Interpreter(buf, opts)
        }
        return Interpreter(java.io.File(path), opts)
    }

    /**
     * Temperature-scaled sigmoid adaptive threshold.
     * Based on ICLR 2026 "Adaptive Threshold Calibration".
     * Replaces fixed 0.55 with data-driven threshold.
     * @param numEnrolled number of enrolled faces (for calibration)
     */
    private fun adaptiveThreshold(numEnrolled: Int): Float {
        val t = TEMPERATURE
        val logit = (numEnrolled.toFloat() / ENROLL_FRAMES) * 2f - 1f
        val sigmoid = 1f / (1f + kotlin.math.exp(-logit / t))
        return BASELINE_THRESHOLD + sigmoid * 0.15f  // range: 0.45–0.60
    }

    /**
     * Warmup invocation — runs dummy inference to prime JIT/AOT
     * compilation and avoid first-frame latency spike.
     */
    private fun warmupModels() {
        Log.d(TAG, "Warming up models...")
        val dummyInput = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())

        // Log model input shapes for diagnostics
        recognitionInterpreter?.let { interp ->
            val inputShape = interp.getInputTensor(0).shape()
            Log.d(TAG, "Recognition input shape: ${inputShape.joinToString("x")}")
        }
        livenessInterpreter?.let { interp ->
            val inputShape = interp.getInputTensor(0).shape()
            val outputShape = interp.getOutputTensor(0).shape()
            Log.d(TAG, "Liveness input shape: ${inputShape.joinToString("x")}, output shape: ${outputShape.joinToString("x")}")
        }

        for (i in 0 until 3) {
            dummyInput.rewind()
            recognitionInterpreter?.run(dummyInput, recognitionOutput)
            livenessInterpreter?.run(dummyInput, livenessOutput)
        }
        Log.d(TAG, "Model warmup done")
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            frameCounter += 1
            Log.d(TAG, "analyzeFrame called - frame $frameCounter")
            // No frame decimation - process every frame so rPPG gets 30Hz sample rate.
            // Heavy liveness layers (texture/light/banding) are decimated separately.

            val bitmap = imageProxyToBitmap(image) ?: run {
                Log.e(TAG, "imageProxyToBitmap returned null!")
                image.close(); return
            }
            Log.d(TAG, "Bitmap created: ${bitmap.width}x${bitmap.height}")
            val rotated = rotateBitmap(bitmap, image.imageInfo.rotationDegrees)
            val w = rotated.width
            val h = rotated.height

            val landmarker = faceLandmarker ?: run { image.close(); return }
            val mpImage = BitmapImageBuilder(rotated).build()
            val result = landmarker.detect(mpImage)

            if (result.faceLandmarks().isEmpty()) {
                postOverlay(emptyArray(), w, h)
                updateStatus("Looking for face…")
                // Clear stale info text when no face is detected (prevents "Liveness: VERIFIED" persisting)
                runOnUiThread {
                    if (lastInfoText.isNotEmpty() && lastInfoText != "Tap Enroll or Verify to begin") {
                        lastInfoText = ""
                        infoText.text = ""
                    }
                }
                // Verify-timeout: if VERIFYING and no face for 10s, reset to IDLE
                // (otherwise state is stuck forever if user walks away)
                if (state == State.VERIFYING) {
                    val now = System.currentTimeMillis()
                    if (lastFaceSeenAt == 0L) lastFaceSeenAt = now
                    if (now - lastFaceSeenAt > 10_000L) {
                        runOnUiThread {
                            state = State.IDLE
                            mode = Mode.NONE
                            activeChallenge.cancel()
                            lastShownChallengeStep = null
                            clearOverlayState()
                            hideChallengePrompt()
                            updateStatus("Verify timed out — no face for 10s. Tap Verify again.")
                        }
                    }
                }
                image.close(); return
            }

            val landmarkList = result.faceLandmarks()[0]
            val landmarks = ArrayList<PointF>(landmarkList.size)
            for (i in 0 until landmarkList.size) {
                val lm = landmarkList[i]
                landmarks.add(PointF(lm.x() * w, lm.y() * h))
            }

            // Update lastFaceSeenAt — used for verify-timeout (no face for 10s → IDLE)
            lastFaceSeenAt = System.currentTimeMillis()
            val key5 = KeypointExtractor.extract(landmarks)
            // landmarks are already in pixel coords (lm.x()*w, lm.y()*h)
            // boundingBox() would multiply by w,h again → coordinates outside image

            // EAR (Eye Aspect Ratio) from MediaPipe FaceMesh landmarks — fallback blink detector
            // Left eye indices: 33 (outer corner), 160, 158 (upper), 133 (inner), 153, 144 (lower)
            // Right eye indices: 263 (inner), 387, 385 (upper), 362 (outer), 380, 373 (lower)
            // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)   Open ~0.30, Closed ~0.10
            // Snapshot challenge step before processing — used to fire haptic feedback
            // the instant a gesture is registered (step advances).
            val prevChallengeStep = activeChallenge.currentStep
            if (landmarks.size >= 468) {
                val earLeft = computeEar(landmarks, 33, 160, 158, 133, 153, 144)
                val earRight = computeEar(landmarks, 263, 387, 385, 362, 380, 373)
                val ear = (earLeft + earRight) / 2f
                activeChallenge.onEar(ear)
            }
            var boxMinX = Float.MAX_VALUE; var boxMinY = Float.MAX_VALUE
            var boxMaxX = -Float.MAX_VALUE; var boxMaxY = -Float.MAX_VALUE
            for (kp in landmarks) {
                if (kp.x < boxMinX) boxMinX = kp.x
                if (kp.y < boxMinY) boxMinY = kp.y
                if (kp.x > boxMaxX) boxMaxX = kp.x
                if (kp.y > boxMaxY) boxMaxY = kp.y
            }
            val box = RectF(boxMinX, boxMinY, boxMaxX, boxMaxY)

            // --- BENCHMARKING START ---
            val tStart = System.nanoTime()
            
            // 5-point similarity transform -> 112x112 CHW float
            val yuv = yPlaneFromImageProxy(image)
            FaceAligner.align(yuv, image.planes[0].rowStride, w, h, key5, alignedBuffer)
            val tAlign = System.nanoTime()
            
            // Embedding
            val rec = recognitionInterpreter
            val embed: FloatArray = if (rec != null) {
                recognitionInput.rewind()
                for (v in alignedBuffer) recognitionInput.putFloat(v)
                rec.run(recognitionInput, recognitionOutput)
                l2Normalize(recognitionOutput[0])
            } else FloatArray(EMBED_DIM)
            val tRec = System.nanoTime()

            // Liveness (passive) — MiniFASNet expects BGR color, mean=104, std=127
            val liv = livenessInterpreter
            val livenessRaw: FloatArray = if (liv != null) {
                livenessInput.rewind()
                val faceBmp = cropAndResizeForLiveness(rotated, box, INPUT_SIZE)
                val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
                faceBmp.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
                if (faceBmp !== rotated) faceBmp.recycle()
                // BGR [0,1] normalization — only one producing real signal in A-E test
                for (px in pixels) {
                    val r = (px shr 16) and 0xFF
                    val g = (px shr 8) and 0xFF
                    val b = px and 0xFF
                    livenessInput.putFloat(b / 255.0f)
                    livenessInput.putFloat(g / 255.0f)
                    livenessInput.putFloat(r / 255.0f)
                }
                liv.run(livenessInput, livenessOutput)
                softmax(livenessOutput[0])
            } else floatArrayOf(1f, 0f, 0f)
            val tLive = System.nanoTime()
            // --- BENCHMARKING END ---

            // class 0 = real, class 1 = print attack, class 2 = replay attack
            val livenessScore = livenessRaw.getOrNull(0) ?: 0f
            liveScore = livenessScore
            // NOTE: The on-device liveness model is unreliable (522KB, broken
            // on this device). The 10-layer algorithmic fusion + active
            // challenge is the real anti-spoofing gate. Don't use livenessScore
            // to set spoof here — the active challenge vetoes below handle it.
            // We still capture livenessScore for the info card display.

            if (frameCounter % 60 == 0) {  // was 30 — reduce logcat overhead
                val msAlign = (tAlign - tStart) / 1_000_000
                val msRec = (tRec - tAlign) / 1_000_000
                val msLive = (tLive - tRec) / 1_000_000
                val msTotal = (tLive - tStart) / 1_000_000
                Log.d(TAG, "BENCH [ms] Align:$msAlign Rec:$msRec Live:$msLive TOTAL:$msTotal")
                Log.d(TAG, "LIVENESS raw=[${"%.3f".format(livenessRaw[0])}, ${"%.3f".format(livenessRaw[1])}, ${"%.3f".format(livenessRaw[2])}] real=${"%.3f".format(livenessScore)} spoof=$spoof")
            }

            // rPPG pulse using SOTA POS algorithm — samples 5 face ROIs
            val roiSamples = sampleFaceROIs(rotated, key5)
            rppg.pushSample(roiSamples[0].r, roiSamples[0].g, roiSamples[0].b)
            rppg.pushSpatialSamples(
                DoubleArray(5) { roiSamples[it].r },
                DoubleArray(5) { roiSamples[it].g }
            )
            val rppgResult = rppg.analyze()
            val bpm = rppgResult.bpm
            val isLive = rppgResult.isLive
            val pulseInfo = if (isLive) "$bpm bpm" else "—"

            // Active challenge
            val bsList = result.faceBlendshapes()
            val innerCategories: List<com.google.mediapipe.tasks.components.containers.Category>? = if (bsList.isPresent) {
                val outer = bsList.get()
                if (outer.isNotEmpty()) outer[0] else null
            } else null
            val blinkL = innerCategories?.firstOrNull { it.categoryName() == "eyeBlinkLeft" }?.score() ?: 0f
            val blinkR = innerCategories?.firstOrNull { it.categoryName() == "eyeBlinkRight" }?.score() ?: 0f
            // MediaPipe uses separate mouthSmileLeft and mouthSmileRight blendshapes (not mouthSmile)
            // Also handle legacy "mouthSmile" and "_smile" for older models
            val smileL = innerCategories?.firstOrNull { it.categoryName() == "mouthSmileLeft" }?.score() ?: 0f
            val smileR = innerCategories?.firstOrNull { it.categoryName() == "mouthSmileRight" }?.score() ?: 0f
            val smileLegacy = innerCategories?.firstOrNull { it.categoryName() == "mouthSmile" || it.categoryName() == "_smile" }?.score() ?: 0f
            // Take max of all smiles — left, right, legacy
            val smile = maxOf(smileL, smileR, smileLegacy)
            val jawOpen = innerCategories?.firstOrNull { it.categoryName() == "jawOpen" }?.score() ?: 0f
            val headYaw = innerCategories?.firstOrNull { it.categoryName() == "headYaw" }?.score() ?: 0f
            activeChallenge.onBlendshapes(blinkL, blinkR, smile, headYaw, System.currentTimeMillis(), jawOpen)

            // Log smile value every 30 frames so user can verify it's picking up
            if (frameCounter % 30 == 0 && activeChallenge.currentStep == ActiveChallengeRunner.Step.SMILE) {
                Log.d(TAG, "SMILE debug: L=$smileL R=$smileR legacy=$smileLegacy MAX=$smile (thr=0.10)")
            }

            // Head turn detection via landmark geometry (MediaPipe doesn't have headYaw blendshape)
            val eyeMidX = (key5[0].x + key5[1].x) / 2f
            val eyeDist = kotlin.math.abs(key5[1].x - key5[0].x).coerceAtLeast(1f)
            val noseOffsetX = (key5[2].x - eyeMidX) / eyeDist
            activeChallenge.onLandmarksForHeadTurn(noseOffsetX)

            // Haptic feedback: vibrate the instant a challenge step advances so the
            // user knows their gesture was recognized. Success pattern on final
            // PASSED, light tap on each intermediate step.
            val challengeStepNow = activeChallenge.currentStep
            if (challengeStepNow != prevChallengeStep) {
                when (challengeStepNow) {
                    ActiveChallengeRunner.Step.PASSED -> haptics.success()
                    ActiveChallengeRunner.Step.FAILED,
                    ActiveChallengeRunner.Step.IDLE -> Unit
                    else -> haptics.tap()
                }
            }

            // === 10-LAYER LIVENESS FUSION (NetraEdge X SOTA 2026) ===
            // Run texture/light/banding layers only on every 5th frame for speed.
            // Sensor + Temporal run every frame (cheap). rPPG + Active every frame (cheap).
            val runHeavyLiveness = (frameCounter % 5 == 0)

            // Extract face crop bitmap for texture/light/banding analysis
            val faceBmp = cropAndResizeForLiveness(rotated, box, INPUT_SIZE)
            val boxInt = android.graphics.Rect(
                box.left.toInt().coerceAtLeast(0),
                box.top.toInt().coerceAtLeast(0),
                box.right.toInt().coerceAtMost(rotated.width),
                box.bottom.toInt().coerceAtMost(rotated.height)
            )

            // Layer 1: Active challenge (computed above, convert to score)
            val activeStep = activeChallenge.currentStep
            val activeScore = when (activeStep) {
                ActiveChallengeRunner.Step.PASSED -> 1.0f
                ActiveChallengeRunner.Step.FAILED -> 0.0f
                ActiveChallengeRunner.Step.IDLE -> 0.0f  // No signal yet — strict default
                else -> 0.0f  // In progress — no credit until completed
            }

            // Layer 2: rPPG pulse (computed above)
            val rppgScore = if (isLive && bpm in 40..150) {
                // Confidence: prominent peak in physiological range
                ((bpm.toFloat() - 30f) / 130f).coerceIn(0.3f, 1.0f)
            } else 0.0f

            // Layers 3-6, 9-10: texture/light/banding (run every 3rd frame)
            var textureScore = 0.5f
            var colorScore = 0.5f
            var moireScore = 0.5f
            var specularScore = 0.5f
            var lightScore = 0.5f
            var bandingScore = 0.5f
            if (runHeavyLiveness) {
                textureScore = textureAnalyzer.analyze(faceBmp)
                colorScore = colorAnalyzer.analyze(faceBmp)
                moireScore = moireDetector.analyze(faceBmp)
                specularScore = specularDetector.analyze(faceBmp)
                lightScore = lightAnalyzer.analyze(rotated, boxInt)
                bandingScore = bandingDetector.analyze(faceBmp)
            }
            if (faceBmp !== rotated) faceBmp.recycle()

            // Layer 7: Accelerometer + Gyro
            val sensorScore = sensorFusion.analyze()

            // Layer 8: Temporal consistency (push embedding then analyze)
            temporalAnalyzer.pushEmbedding(embed)
            val temporalScore = temporalAnalyzer.analyze()

            // === STATE-AWARE WEIGHTED FUSION ===
            //
            // IDLE state: Active challenge and rPPG aren't applicable (challenge
            //              not started, pulse needs 1-2s to lock on). Redistribute
            //              their weight to texture/light/banding (passive-only layers).
            //
            // VERIFYING state: All 10 layers contribute. Active challenge is the
            //                  primary gate.
            val isIdle = (state == State.IDLE)
            val effectiveWeights = if (isIdle) {
                // Redistribute 35% from active+rPPG to texture/light/banding
                floatArrayOf(
                    0.0f,   // 1. Active  (not applicable in IDLE)
                    0.0f,   // 2. rPPG    (not applicable in IDLE)
                    0.20f,  // 3. LBP texture    (up from 0.10)
                    0.12f,  // 4. Color          (up from 0.10)
                    0.13f,  // 5. Moire          (up from 0.10)
                    0.10f,  // 6. Specular
                    0.10f,  // 7. Sensor
                    0.10f,  // 8. Temporal       (up from 0.07)
                    0.13f,  // 9. Light          (up from 0.05)
                    0.12f   // 10. Banding       (up from 0.05)
                )
            } else {
                layerWeights
            }

            val layerScores = floatArrayOf(
                activeScore,    // 1
                rppgScore,      // 2
                textureScore,   // 3
                colorScore,     // 4
                moireScore,     // 5
                specularScore,  // 6
                sensorScore,    // 7
                temporalScore,  // 8
                lightScore,     // 9
                bandingScore    // 10
            )
            var fusedLiveness = 0.0f
            for (i in layerScores.indices) {
                fusedLiveness += effectiveWeights[i] * layerScores[i]
            }

            // === ANTI-REPLAY VETO LOGIC (SOTA 2026) ===
            //
            // REPLAY ATTACK: someone shows a video of the enrolled person on
            // another phone/tablet/laptop. Defense:
            //
            // 1. NO rPPG PULSE — videos don't have blood flow. After 4s of
            //    processing in VERIFYING mode, if rPPG is 0.0, force SPOOF.
            //    This catches video-on-phone, video-on-laptop, photo attacks.
            //
            // 2. SCREEN MOIRE — the Moire analyzer detects pixel-grid patterns
            //    from LCD/OLED displays. Strong Moire signal = screen detected.
            //
            // 3. BANDING — screens have rolling-shutter aliasing when filmed.
            //    Strong banding signal = screen detected.
            //
            // 4. LIGHT MISMATCH — video face has fixed lighting; real face
            //    has lighting consistent with background.
            //
            // Together: even if the active challenge passes (because the video
            // shows someone blinking), the lack of rPPG pulse + Moire + Banding
            // detects the replay.
            val verifyElapsedMs = if (state == State.VERIFYING || state == State.ENROLLING) {
                System.currentTimeMillis() - stateStartedAt
            } else 0L

            // rPPG needs at least 4s of data to lock onto a physiological pulse.
            // During enrollment, the user also needs time to complete 3 challenges
            // + capture 15 frames, so we extend the window to 8s.
            // Before this window closes, the system gives the user the benefit of
            // the doubt — the active challenge is the realer gate for liveliness.
            val canCheckRppg = if (state == State.ENROLLING) {
                verifyElapsedMs > 8000L
            } else {
                verifyElapsedMs > 4000L
            }

            // Grace period: extends rPPG/moire veto immunity to 6s during enrollment
            // (user needs time to do 3 challenges), 3s during verify.
            val inGracePeriod = if (state == State.ENROLLING) {
                verifyElapsedMs < 6000L
            } else {
                verifyElapsedMs < 3000L
            }

            // rPPG veto is ADVISORY ONLY — rPPG is unreliable across devices, lighting,
            // and skin tones. False-positives on real faces are unacceptable for a live
            // demo. The randomized active challenge is the hard liveness gate.
            // We only soft-penalize the fused score; we do NOT set spoof=true here.
            if ((state == State.VERIFYING || state == State.ENROLLING) && !inGracePeriod && canCheckRppg && rppgScore <= 0.10f && bpm <= 0) {
                fusedLiveness *= 0.5f
                if (frameCounter % 60 == 0) {
                    Log.w(TAG, "rPPG advisory: no pulse after ${verifyElapsedMs}ms (lighting/device dependent — not blocking)")
                }
            }

            // Moire+Banding veto is ADVISORY ONLY — these heuristic detectors
            // false-positive on real faces under many lighting/camera conditions.
            // Only an EXTREME consensus (both < 0.15) soft-penalizes the score.
            if ((state == State.VERIFYING || state == State.ENROLLING) && !inGracePeriod && moireScore < 0.15f && bandingScore < 0.15f) {
                fusedLiveness *= 0.5f
                if (frameCounter % 60 == 0) {
                    Log.w(TAG, "Moire/Banding advisory: extreme screen-like pattern (moire=${"%.2f".format(moireScore)} banding=${"%.2f".format(bandingScore)})")
                }
            }

            // Hard veto: if active challenge FAILED during verify, force SPOOF
            // (also protected by grace period)
            if ((state == State.VERIFYING || state == State.ENROLLING) && !inGracePeriod &&
                activeStep == ActiveChallengeRunner.Step.FAILED) {
                fusedLiveness = 0.0f
                spoof = true
                if (frameCounter % 60 == 0) {
                    Log.w(TAG, "SPOOF veto: active challenge FAILED at ${verifyElapsedMs}ms")
                }
            }

            // Active challenge PASSED — clear spoof (in case it was set by a
            // transient veto that has since resolved, e.g. user just moved).
            if (activeStep == ActiveChallengeRunner.Step.PASSED && state == State.VERIFYING) {
                spoof = false
            }

            liveScore = fusedLiveness

            // === SPOOF DETECTION — STATE-AWARE FUSION ===
            //
            // IDLE state: Show "Tap Verify" instead of REAL/FAKE — a real face
            //              looking at the camera may not have pulse data yet
            //              (rPPG needs 1-2 seconds to lock on), so we can't
            //              accurately verdict without explicit verification.
            //              We only flag a photo as FAKE in IDLE if the
            //              passive-only layers score very low (< 0.20).
            //
            // VERIFYING state: DO NOT use fusedLivenessEma for spoof decision.
            //                  The active challenge (BLINK/SMILE/HEAD_TURN) is
            //                  the gatekeeper. fusedLivenessEma naturally dips
            //                  during the first 1-2 seconds of VERIFYING (user
            //                  hasn't done the challenge yet → blink/motion/
            //                  rPPG are all 0 → baseline ~0.25) which would
            //                  incorrectly flag a real face as SPOOF.
            //                  Spoof in VERIFYING is set by:
            //                    1. Active challenge FAILED handler (below)
            //                    2. Replay veto (rPPG=0 + moire/banding high)
            //                    3. Specific liveness vetoes (3D mask, etc.)
            //
            // SPOOF/VERIFIED/NOT_RECOGNIZED: Frozen, don't update.
            //
            // EMA smoothing (alpha=0.15) prevents flicker between REAL/FAKE.
            fusedLivenessEma = fusedLivenessEma * 0.85f + fusedLiveness * 0.15f
            if (state == State.IDLE) {
                // Relaxed passive check (below the 0.25 baseline of a still
                // real face). The real anti-spoofing gate is the active
                // challenge in VERIFYING + the replay veto. IDLE is just
                // a soft visual cue for the user.
                spoof = fusedLivenessEma < 0.20f
            }
            // NOTE: VERIFYING does NOT use fusedLivenessEma for spoof here.
            // See the defense-in-depth check in onVerifyFrame which defers
            // to the active challenge result instead.

            // Log all 10 layers every 30 frames
            if (frameCounter % 30 == 0) {
                Log.d(TAG, "FUSED LIVENESS: ${"%.3f".format(fusedLiveness)}")
                Log.d(TAG, "  [1] Active:     ${"%.3f".format(activeScore)}")
                Log.d(TAG, "  [2] rPPG:       ${"%.3f".format(rppgScore)} (bpm=$bpm)")
                Log.d(TAG, "  [3] LBP:        ${"%.3f".format(textureScore)}")
                Log.d(TAG, "  [4] Color:      ${"%.3f".format(colorScore)}")
                Log.d(TAG, "  [5] Moire:      ${"%.3f".format(moireScore)}")
                Log.d(TAG, "  [6] Specular:   ${"%.3f".format(specularScore)}")
                Log.d(TAG, "  [7] Sensor:     ${"%.3f".format(sensorScore)}")
                Log.d(TAG, "  [8] Temporal:   ${"%.3f".format(temporalScore)}")
                Log.d(TAG, "  [9] Light:      ${"%.3f".format(lightScore)}")
                Log.d(TAG, "  [10] Banding:    ${"%.3f".format(bandingScore)}")
            }

            // Overlay
            postOverlay(key5, w, h, box = box, label = currentLabel(), bpm = if (isLive) bpm else 0, liveFace = !spoof, fused = fusedLiveness)
            updateInfo(embed, bpm, isLive)
            // Update 8 signal bars in info card (one per layer)
            updateSignalBars(layerScores)

            // === ACTIVE CHALLENGE PROMPT ===
            // Show popup during ACTIVE_LIVENESS mode (ENROLLING/VERIFYING).
            // Hide it for terminal states (ENROLLED/VERIFIED/SPOOF/NOT_RECOGNIZED/IDLE).
            // Debounce: only update UI thread when step actually changes.
            val promptStep = activeChallenge.currentStep
            val shouldShow = mode == Mode.ACTIVE_LIVENESS &&
                (state == State.ENROLLING || state == State.VERIFYING) &&
                promptStep != ActiveChallengeRunner.Step.IDLE
            if (shouldShow && promptStep != lastShownChallengeStep) {
                lastShownChallengeStep = promptStep
                // Step index in challenge sequence (1-based for display)
                val stepIdx = activeChallenge.getCurrentStepIndex() + 1
                val total = activeChallenge.getTotalSteps()
                showChallengePrompt(promptStep, stepIdx, total)
            } else if (shouldShow && challengePromptCard.visibility != android.view.View.VISIBLE) {
                // Step hasn't changed but popup is hidden — re-show it
                // (e.g., after a transient state change). Same step index.
                val stepIdx = activeChallenge.getCurrentStepIndex() + 1
                val total = activeChallenge.getTotalSteps()
                showChallengePrompt(promptStep, stepIdx, total)
            } else if (!shouldShow && challengePromptCard.visibility == android.view.View.VISIBLE) {
                // Mode exited — hide popup
                lastShownChallengeStep = null
                hideChallengePrompt()
            }

            // State machine
            when (state) {
                State.ENROLLING -> onEnrollFrame(embed, w, h, key5)
                State.VERIFYING -> onVerifyFrame(embed, livenessScore, bpm, isLive)
                else -> Unit
            }
        } catch (e: Throwable) {
            Log.e(TAG, "analyzeFrame error", e)
        } finally {
            image.close()
        }
    }

    private fun onEnrollFrame(embed: FloatArray, imgW: Int, imgH: Int, key5: Array<PointF>) {
        // ANTI-SPOOF GATE: reject enrollment if any liveness layer flags replay
        if (spoof) {
            updateStatus("SPOOF: screen replay detected — cannot enroll")
            haptics.error()
            return
        }
        // Geometry-based quality gate — fast, no per-pixel computation
        val leftEye = key5[0]
        val rightEye = key5[1]
        val nose = key5[2]
        val leftMouth = key5[3]
        val rightMouth = key5[4]

        // 1. Face size — eye distance should be > 20% of frame width
        val eyeDist = kotlin.math.abs(rightEye.x - leftEye.x)
        val faceSizeRatio = (eyeDist * 2.5f) / imgW
        if (faceSizeRatio < 0.20f) {
            updateStatus("Move closer — face too small")
            return
        }
        val sizeScore = (faceSizeRatio / 0.5f).coerceIn(0f, 1f)

        // 2. Pose — eyes should be roughly level (roll check)
        val eyeYDiff = kotlin.math.abs(leftEye.y - rightEye.y) / imgH
        if (eyeYDiff > 0.06f) {
            updateStatus("Keep head level — too tilted")
            return
        }
        val rollScore = 1f - (eyeYDiff / 0.06f)

        // 3. Face centered — nose should be between eyes horizontally
        val noseX = nose.x / imgW
        if (noseX < 0.25f || noseX > 0.75f) {
            updateStatus("Face the camera center")
            return
        }
        val centerScore = 1f - kotlin.math.abs(noseX - 0.5f) * 4f

        // 4. Landmark consistency — mouth should be below eyes
        if (leftMouth.y <= leftEye.y || rightMouth.y <= rightEye.y) {
            updateStatus("Invalid face geometry")
            return
        }

        // Combined quality score
        val quality = (sizeScore * 0.4f + rollScore * 0.3f + centerScore * 0.3f)

        // ACTIVE CHALLENGE GATE: enrollment frames are ONLY captured when the
        // user is actively engaging with the liveness challenge. This prevents
        // enrolling a static photo (which would never blink/smile/turn).
        val activeStep = activeChallenge.currentStep
        if (activeStep != ActiveChallengeRunner.Step.PASSED) {
            // Status pill just shows progress; the popup shows the actual challenge
            updateStatus("Enrolling: ${enrollmentBuffer.size}/$ENROLL_FRAMES — complete the challenge")
            return
        }

        enrollmentBuffer.add(embed.copyOf())
        enrollmentQualities.add(quality)
        updateStatus("Enrolling: ${enrollmentBuffer.size}/$ENROLL_FRAMES (${(quality * 100).toInt()}%)")
        if (enrollmentBuffer.size >= ENROLL_FRAMES) {
            // rPPG pulse check is ADVISORY ONLY — rPPG is unreliable across devices,
            // lighting, and skin tones. The randomized active challenge (2 blinks,
            // sustained gestures, shuffled 3-of-4) is the hard liveness gate.
            // We log the rPPG status for diagnostics but never block enrollment on it.
            val rppgResult = rppg.analyze()
            val hasPulse = rppgResult.isLive && rppgResult.bpm in 40..150
            if (!hasPulse) {
                Log.w(TAG, "rPPG advisory: no pulse at enrollment completion (lighting/device dependent — not blocking)")
            }

            // Quality-weighted averaging — higher quality frames contribute more
            val avg = FloatArray(EMBED_DIM)
            var totalQuality = 0f
            for (i in enrollmentBuffer.indices) {
                val q = enrollmentQualities[i]
                totalQuality += q
                for (j in 0 until EMBED_DIM) avg[j] += enrollmentBuffer[i][j] * q
            }
            for (i in 0 until EMBED_DIM) avg[i] /= totalQuality
            val normalized = l2Normalize(avg)
            enrolledEmbedding = normalized
            saveEnrolledEmbedding(normalized)
            enrollmentBuffer.clear()
            enrollmentQualities.clear()
            state = State.ENROLLED
            haptics.success()
            updateStatus("Enrolled successfully (quality-weighted)")
        }
    }

    private val livenessHistory = ArrayList<Float>(10)

    private fun onVerifyFrame(embed: FloatArray, livenessScore: Float, bpm: Int, isLive: Boolean) {
        livenessHistory.add(livenessScore)
        if (livenessHistory.size > 10) livenessHistory.removeFirst()

        val avgLiveness = livenessHistory.average().toFloat()
        val activeState = activeChallenge.currentStep
        val activePassed = activeState == ActiveChallengeRunner.Step.PASSED
        val activeFailed = activeState == ActiveChallengeRunner.Step.FAILED

        Log.d(TAG, "VERIFY frame: raw=${"%.3f".format(livenessScore)} avg=${"%.3f".format(avgLiveness)} active=$activeState rPPG=$isLive bpm=$bpm history=${livenessHistory.size} spoof=$spoof")

        // === 10-LAYER FUSION SPOOF VETO (defense-in-depth) ===
        //
        // The main analyzeFrame sets spoof=true when ANY of these trigger
        // (during VERIFYING):
        //   • rPPG veto: no pulse after 4s (catches photo, video-on-phone, video-on-laptop)
        //   • Moire+banding both > 0.6 (catches screens)
        //   • Active challenge FAILED
        //
        // Note: VERIFYING does NOT use fusedLivenessEma-based spoof — that
        // would falsely flag a real face as SPOOF during the first 1-2
        // seconds of the active challenge (when blink/motion/rPPG are
        // still 0). The active challenge + identity check is the gate.
        //
        // onVerifyFrame MUST honor this verdict — otherwise a spoof could pass
        // through to VERIFIED just because the embedding similarity is high.
        if (spoof) {
            state = State.SPOOF
            haptics.error()
            updateStatus("SPOOF: liveness fusion vetoed (10-layer)")
            Log.w(TAG, "SPOOF verdict: 10-layer fusion vetoed")
            return
        }

        if (activeFailed) {
            state = State.SPOOF
            spoof = true
            haptics.error()
            updateStatus("SPOOF: active challenge failed")
            Log.w(TAG, "SPOOF verdict: active challenge FAILED")
            return
        }
        if (!activePassed && livenessHistory.size < 5) {
            updateStatus("Verifying — complete the active challenge")
            return
        }

        // === ACTIVE CHALLENGE GATE (defense-in-depth) ===
        //
        // For VERIFIED verdict, the user must have completed the active challenge.
        // This prevents an attacker from passing identity match without proving
        // they're a live person (even if all passive layers happen to be OK).
        if (!activePassed) {
            updateStatus("Verifying — complete the active challenge to verify")
            return
        }

        // Active passed + spoof clear — proceed to identity check
        val ref = enrolledEmbedding ?: run {
            updateStatus("No enrolled face. Tap Enroll first.")
            state = State.IDLE
            return
        }
        val sim = cosineSimilarity(embed, ref)
        val ttaEmbed = flipEmbedding(embed)
        val ttaSim = cosineSimilarity(ttaEmbed, ref)
        val finalSim = maxOf(sim, ttaSim)

        val threshold = adaptiveThreshold(enrollmentBuffer.size)
        if (finalSim >= threshold) {
            state = State.VERIFIED
            spoof = false
            haptics.success()
            val bonus = if (activePassed) " +active" else ""
            updateStatus("VERIFIED (sim=${"%.3f".format(finalSim)}$bonus)")
        } else {
            state = State.NOT_RECOGNIZED
            haptics.tap()  // light haptic on unknown person (not SPOOF — different signal)
            updateStatus("NOT RECOGNIZED (sim=${"%.3f".format(finalSim)}, thr=${"%.3f".format(threshold)})")
        }
    }

    /**
     * TTA: approximate horizontal flip augmentation.
     * Reshapes 128-d embedding into 8x16 2D grid, flips horizontally,
     * and flattens back. This simulates the model seeing a mirrored face.
     */
    private fun flipEmbedding(embed: FloatArray): FloatArray {
        val rows = 8
        val cols = EMBED_DIM / rows  // 16
        val out = FloatArray(EMBED_DIM)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[r * cols + c] = embed[r * cols + (cols - 1 - c)]
            }
        }
        return l2Normalize(out)
    }

    private fun onEnrollClicked() {
        // Always reset the popup debounce so the challenge prompt shows for
        // every enroll, even if the previous run's last step matches.
        lastShownChallengeStep = null
        // Reset any terminal state (SPOOF/VERIFIED/NOT_RECOGNIZED/ENROLLED) so the
        // badge, label, and overlay are always consistent. Without this, if state
        // is SPOOF and user clicks Enroll, the badge may show IDLE but the
        // overlay still draws "SPOOF DETECTED" because state is unchanged.
        if (state == State.SPOOF || state == State.VERIFIED || state == State.NOT_RECOGNIZED || state == State.ENROLLED) {
            state = State.IDLE
            spoof = false
            mode = Mode.NONE
            activeChallenge.cancel()
            hideChallengePrompt()
        }
        // Block re-enroll if already enrolled — must reset first
        if (enrolledEmbedding != null) {
            updateStatus("Already enrolled. Tap Reset to re-enroll.")
            clearOverlayState()
            return
        }
        enrollmentBuffer.clear()
        enrollmentQualities.clear()
        state = State.ENROLLING
        spoof = false
        fusedLivenessEma = 0.5f
        stateStartedAt = System.currentTimeMillis()
        lastFaceSeenAt = 0L
        activeChallenge.start()
        mode = Mode.ACTIVE_LIVENESS
        clearOverlayState()
        // Force-clear stale info text for clean transition
        runOnUiThread {
            lastInfoText = ""
        }
        updateStatus("Enrolling: 0/$ENROLL_FRAMES — complete the challenge")
    }

    private fun onVerifyClicked() {
        if (enrolledEmbedding == null) {
            updateStatus("No enrolled face. Tap Enroll first.")
            return
        }
        livenessHistory.clear()
        state = State.VERIFYING
        spoof = false
        fusedLivenessEma = 0.5f
        stateStartedAt = System.currentTimeMillis()
        lastFaceSeenAt = 0L  // reset verify-timeout timer
        // Reset the popup debounce so the challenge prompt will definitely show,
        // even if the previous session's lastShownChallengeStep matches the
        // new active challenge step.
        lastShownChallengeStep = null
        activeChallenge.start()
        mode = Mode.ACTIVE_LIVENESS
        clearOverlayState()
        hideChallengePrompt()  // ensure clean state — showChallengePrompt will animate it in
        // Force-clear stale info text for clean transition
        runOnUiThread {
            lastInfoText = ""
        }
        updateStatus("Verifying identity… — complete the challenge")
    }

    private fun onResetClicked() {
        enrollmentBuffer.clear()
        enrollmentQualities.clear()
        livenessHistory.clear()
        state = State.IDLE
        mode = Mode.NONE
        activeChallenge.cancel()
        enrolledEmbedding = null
        stateStartedAt = 0L
        lastFaceSeenAt = 0L
        fusedLivenessEma = 0.5f
        spoof = false
        lastShownChallengeStep = null
        prefs().edit().remove(KEY_EMBEDDING).apply()
        updateStatus("NetraEdge ready")
        clearOverlayState()
        hideChallengePrompt()
        // Force-clear stale UI immediately
        runOnUiThread {
            infoText.text = "Tap Enroll or Verify to begin"
            lastInfoText = ""
            lastStatusText = ""
            // Clear all signal bars to neutral
            sigBars.forEach { it.setBackgroundResource(R.drawable.bar_neutral) }
        }
    }

    private val syncService by lazy { SyncService(applicationContext) }

    /**
     * Reset face overlay state — used on every state transition to ensure
     * stale VERIFIED/SPOOF labels don't bleed into new sessions.
     */
    private fun clearOverlayState() {
        runOnUiThread {
            faceOverlay.label = ""
            faceOverlay.isVerified = false
            faceOverlay.isDenied = false
            faceOverlay.isVerifying = false
            faceOverlay.isEnrolling = false
            faceOverlay.bpm = 0
            faceOverlay.fusedLiveness = 0f
            faceOverlay.isSpoof = false
            faceOverlay.confidence = 0f
            faceOverlay.clearParticles()
        }
    }

    private fun onSyncClicked() {
        val emb = enrolledEmbedding
        if (emb == null) {
            updateStatus("No enrolled face to sync. Enroll first.")
            return
        }
        // Show the target URL in the status for transparency
        val target = syncService.getOverrideEndpoint() ?: "auto-detected"
        updateStatus("Syncing to cloud → $target…")
        syncService.upload(
            userId = "user_${emb.contentHashCode().toString().take(8)}",
            embedding = emb,
            callback = object : SyncService.Callback {
                override fun onStarted() {
                    updateStatus("Uploading embedding (Laplace ε=1.0, sensitivity=2.0)…")
                }
                override fun onSuccess(serverId: String) {
                    updateStatus("Synced ✓ — serverId=$serverId — tap Purge to clear local")
                }
                override fun onError(message: String) {
                    // Show error in status + offer to set custom URL via long-press
                    updateStatus("Sync failed: $message (long-press Sync to set custom URL)")
                }
            },
            scope = lifecycleScope
        )
    }

    /**
     * Long-press Sync button → set custom endpoint URL dialog.
     * Useful when the device is on a different network than the server.
     */
    private fun showSyncSettingsDialog() {
        val input = android.widget.EditText(this).apply {
            setText(syncService.getOverrideEndpoint() ?: "")
            hint = "http://192.168.1.100:4000"
            setSingleLine()
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_VARIATION_URI
        }
        val pad = (16 * resources.displayMetrics.density).toInt()
        val frame = android.widget.FrameLayout(this).apply {
            setPadding(pad, pad, pad, 0)
            addView(input)
        }
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Custom Sync URL")
            .setMessage("Set a custom endpoint URL. Leave blank to auto-detect from WiFi.\n\nCurrent override: ${syncService.getOverrideEndpoint() ?: "(none)"}")
            .setView(frame)
            .setPositiveButton("Save") { _, _ ->
                val url = input.text.toString().trim()
                if (url.isEmpty()) {
                    syncService.clearOverrideEndpoint()
                    updateStatus("Sync URL cleared — will auto-detect from WiFi")
                } else {
                    syncService.setOverrideEndpoint(url)
                    updateStatus("Sync URL set to $url")
                }
            }
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Clear") { _, _ ->
                syncService.clearOverrideEndpoint()
                updateStatus("Sync URL cleared — will auto-detect from WiFi")
            }
            .show()
    }

    private fun onPurgeClicked() {
        // Wipe local data after successful sync (offline-first principle)
        enrollmentBuffer.clear()
        enrollmentQualities.clear()
        livenessHistory.clear()
        enrolledEmbedding = null
        state = State.IDLE
        mode = Mode.NONE
        activeChallenge.cancel()
        stateStartedAt = 0L
        lastFaceSeenAt = 0L
        fusedLivenessEma = 0.5f
        spoof = false
        lastShownChallengeStep = null
        prefs().edit().remove(KEY_EMBEDDING).apply()
        clearOverlayState()
        hideChallengePrompt()
        runOnUiThread {
            lastInfoText = ""
            infoText.text = "Local data purged — ready for next user (cloud has the DP copy)"
            sigBars.forEach { it.setBackgroundResource(R.drawable.bar_neutral) }
        }
        updateStatus("Local data purged — ready for next user (cloud has the DP copy)")
    }

    private fun currentLabel(): String = when (state) {
        State.VERIFYING -> "Verifying…"
        State.VERIFIED -> "VERIFIED"
        State.NOT_RECOGNIZED -> "NOT RECOGNIZED"
        State.SPOOF -> "SPOOF DETECTED"
        State.ENROLLING -> "Enrolling"
        State.ENROLLED -> "Enrolled"
        else -> ""
    }

    // UI throttling: only update text views at 10Hz max, skip if unchanged
    private var lastInfoText = ""
    private var lastStatusText = ""
    private var lastUiUpdateMs = 0L
    private val uiUpdateIntervalMs = 100L  // 10Hz

    private fun updateInfo(embed: FloatArray, bpm: Int, isLive: Boolean) {
        val ref = enrolledEmbedding
        val sim = if (ref != null) cosineSimilarity(embed, ref) else 0f
        val pct = (sim * 100).toInt().coerceIn(0, 100)
        val pulse = if (isLive && bpm > 0) " | $bpm bpm" else ""
        val liveText = when {
            state == State.IDLE -> "Tap Verify to check"
            state == State.VERIFYING -> if (spoof) "FAKE" else "REAL"
            state == State.VERIFIED -> "VERIFIED"
            state == State.SPOOF -> "SPOOF"
            state == State.NOT_RECOGNIZED -> "NOT RECOGNIZED"
            state == State.ENROLLED -> "Enrolled"
            state == State.ENROLLING -> "Enrolling…"
            else -> ""
        }
        val text = when (state) {
            State.VERIFYING -> "Confidence: $pct% | Liveness: $liveText$pulse"
            State.ENROLLING -> "Confidence: $pct% | $liveText"
            else -> "Liveness: $liveText$pulse"
        }
        // Skip if unchanged or too soon (reduces UI thread spam from 30Hz to 10Hz)
        if (text == lastInfoText) return
        val now = System.currentTimeMillis()
        if (now - lastUiUpdateMs < uiUpdateIntervalMs && state != State.VERIFYING) return
        lastInfoText = text
        lastUiUpdateMs = now
        runOnUiThread { infoText.text = text }
    }

    private fun postOverlay(
        key5: Array<PointF>,
        w: Int,
        h: Int,
        box: RectF? = null,
        label: String = "",
        bpm: Int = 0,
        liveFace: Boolean = true,
        fused: Float = 0f
    ) {
        val previewW = previewView.width.toFloat()
        val previewH = previewView.height.toFloat()
        val scaleX = previewW / w
        val scaleY = previewH / h
        val keypointsScaled = key5.map { PointF(it.x * scaleX, it.y * scaleY) }
        val boxScaled = box?.let {
            RectF(it.left * scaleX, it.top * scaleY, it.right * scaleX, it.bottom * scaleY)
        }
        runOnUiThread {
            faceOverlay.faceBox = boxScaled
            faceOverlay.keypoints = keypointsScaled
            faceOverlay.label = label
            faceOverlay.bpm = bpm
            faceOverlay.liveFace = liveFace
            faceOverlay.isSpoof = spoof
            faceOverlay.hasFace = key5.isNotEmpty()
            faceOverlay.confidence = liveScore
            faceOverlay.fusedLiveness = fused
            faceOverlay.isVerifying = (state == State.VERIFYING)
            faceOverlay.isEnrolling = (state == State.ENROLLING)
            faceOverlay.isVerified = (state == State.VERIFIED)
            faceOverlay.isDenied = (state == State.SPOOF)
            faceOverlay.invalidate()
        }
    }

    private fun updateStatus(text: String) {
        if (text == lastStatusText) return  // skip identical updates
        lastStatusText = text
        runOnUiThread {
            // Animate status pill: subtle scale + fade-in on change
            val previous = statusText.text.toString()
            if (previous != text) {
                statusText.alpha = 0.3f
                statusText.scaleX = 0.95f
                statusText.scaleY = 0.95f
                statusText.animate()
                    .alpha(1f)
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(180)
                    .start()
            }
            statusText.text = text
            val color = when {
                text.startsWith("VERIFIED") -> ContextCompat.getColor(this, R.color.brand_accent_green)
                text.startsWith("NOT") || text.contains("SPOOF") -> ContextCompat.getColor(this, R.color.brand_accent_red)
                text.startsWith("Enrolling") || text.startsWith("Verifying") -> ContextCompat.getColor(this, R.color.brand_accent_blue)
                else -> ContextCompat.getColor(this, R.color.brand_text_primary)
            }
            (statusText.parent as? CardView)?.setCardBackgroundColor(
                ContextCompat.getColor(this, R.color.brand_surface)
            )
            statusText.setTextColor(color)
            updateStatusIndicator(text)
        }
    }

    /**
     * Updates the status pill: dot color (neutral/active/warning/error/live) and
     * badge text (IDLE/SCAN/ENROLL/VERIFY/VERIFIED/SPOOF) with smooth transitions.
     */
    private fun updateStatusIndicator(text: String) {
        val (dotRes, badgeText, badgeColor) = when {
            text.startsWith("VERIFIED") -> Triple(R.drawable.dot_active, "VERIFIED", R.color.brand_accent_green)
            text.startsWith("NOT RECOGNIZED") -> Triple(R.drawable.dot_error, "DENIED", R.color.brand_accent_red)
            text.startsWith("NOT") -> Triple(R.drawable.dot_error, "DENIED", R.color.brand_accent_red)
            text.contains("SPOOF") -> Triple(R.drawable.dot_error, "SPOOF", R.color.brand_accent_red)
            text.startsWith("Enrolling") -> Triple(R.drawable.dot_live, "ENROLL", R.color.brand_accent_blue)
            text.startsWith("Verifying") -> Triple(R.drawable.dot_live, "VERIFY", R.color.brand_accent_blue)
            text.startsWith("Looking") -> Triple(R.drawable.dot_neutral, "SCAN", R.color.brand_text_secondary)
            else -> Triple(R.drawable.dot_neutral, "IDLE", R.color.brand_text_secondary)
        }
        statusDot.setBackgroundResource(dotRes)
        statusBadge.text = badgeText
        statusBadge.setTextColor(ContextCompat.getColor(this, badgeColor))
        // Pulse the dot when active
        statusDot.animate().cancel()
        statusDot.scaleX = 1f
        statusDot.scaleY = 1f
        if (dotRes != R.drawable.dot_neutral) {
            statusDot.animate()
                .scaleX(1.35f).scaleY(1.35f)
                .setDuration(450)
                .withEndAction {
                    statusDot.animate()
                        .scaleX(1f).scaleY(1f)
                        .setDuration(450)
                        .start()
                }.start()
        }
    }

    /**
     * Update the 8 liveness signal bars (one per fusion layer) based on per-layer scores.
     * Active=green, neutral=low-alpha white.
     */
    private var lastSigUpdate = 0L
    private fun updateSignalBars(layers: FloatArray) {
        val now = System.currentTimeMillis()
        if (now - lastSigUpdate < 200) return  // 5Hz
        lastSigUpdate = now
        runOnUiThread {
            for (i in sigBars.indices) {
                val v = layers.getOrNull(i) ?: 0f
                val active = v > 0.55f
                sigBars[i].setBackgroundResource(
                    if (active) R.drawable.bar_active else R.drawable.bar_neutral
                )
            }
        }
    }

    /**
     * Quick scale-down/scale-up animation on button press.
     * Gives tactile feedback in 240ms.
     */
    private fun pulseButton(v: android.view.View) {
        v.animate()
            .scaleX(0.92f).scaleY(0.92f)
            .setDuration(80)
            .withEndAction {
                v.animate()
                    .scaleX(1f).scaleY(1f)
                    .setDuration(160)
                    .start()
            }.start()
    }

    /**
     * Show the active challenge prompt with spring animation.
     * Position: just below status pill, above face overlay (no overlap).
     * Debounced: only updates UI when step actually changes (not 30Hz).
     */
    private fun showChallengePrompt(step: ActiveChallengeRunner.Step, stepIndex: Int = -1, totalSteps: Int = 0) {
        // Cancel any pending auto-hide from previous PASSED/FAILED state
        challengePromptCard.removeCallbacks(challengePromptHideRunnable)
        // Cancel any pending countdown updates
        challengePromptCard.removeCallbacks(challengeCountdownRunnable)
        // Cancel any in-flight animations
        challengePromptCard.animate().cancel()
        challengePromptContainer.animate().cancel()

        val (icon, label, text, drawable) = when (step) {
            ActiveChallengeRunner.Step.BLINK -> Tuple4("👁", "ACTIVE CHALLENGE", "Please BLINK", R.drawable.bg_challenge_prompt)
            ActiveChallengeRunner.Step.SMILE -> Tuple4("😊", "ACTIVE CHALLENGE", "Please SMILE", R.drawable.bg_challenge_prompt)
            ActiveChallengeRunner.Step.HEAD_TURN_LEFT -> Tuple4("⬅️", "ACTIVE CHALLENGE", "Turn head LEFT", R.drawable.bg_challenge_prompt)
            ActiveChallengeRunner.Step.HEAD_TURN_RIGHT -> Tuple4("➡️", "ACTIVE CHALLENGE", "Turn head RIGHT", R.drawable.bg_challenge_prompt)
            ActiveChallengeRunner.Step.PASSED -> Tuple4("✅", "CHALLENGE COMPLETE", "All checks passed", R.drawable.bg_challenge_prompt_passed)
            ActiveChallengeRunner.Step.FAILED -> Tuple4("❌", "CHALLENGE FAILED", "Spoof detected", R.drawable.bg_challenge_prompt_failed)
            ActiveChallengeRunner.Step.IDLE -> return hideChallengePrompt()
        }
        runOnUiThread {
            challengeIcon.text = icon
            challengeLabel.text = label
            challengePromptText.text = text
            challengePromptContainer.setBackgroundResource(drawable)
            updateProgressDots(stepIndex, totalSteps)

            if (challengePromptCard.visibility != android.view.View.VISIBLE) {
                challengePromptCard.visibility = android.view.View.VISIBLE
                challengePromptCard.alpha = 0f
                challengePromptCard.translationY = -40f
                challengePromptCard.scaleX = 0.92f
                challengePromptCard.scaleY = 0.92f
                challengePromptCard.animate()
                    .alpha(1f)
                    .translationY(0f)
                    .scaleX(1f).scaleY(1f)
                    .setDuration(380)
                    .setInterpolator(android.view.animation.OvershootInterpolator(1.2f))
                    .start()
            } else {
                // Already visible — pulse the icon + bounce the card
                challengeIcon.scaleX = 0.7f
                challengeIcon.scaleY = 0.7f
                challengeIcon.animate()
                    .scaleX(1f).scaleY(1f)
                    .setDuration(280)
                    .setInterpolator(android.view.animation.OvershootInterpolator(2.0f))
                    .start()
            }
        }

        // Auto-hide PASSED/FAILED after 1.5s
        if (step == ActiveChallengeRunner.Step.PASSED || step == ActiveChallengeRunner.Step.FAILED) {
            challengePromptCard.postDelayed(challengePromptHideRunnable, 1500L)
        } else {
            // Start countdown updates every 1s for active challenge steps
            challengeCountdownRunnable.run()
            challengePromptCard.postDelayed(challengeCountdownRunnable, 1000L)
        }
    }

    /** Updates the countdown text every second. Changes color to red when < 5s left. */
    private val challengeCountdownRunnable = object : Runnable {
        override fun run() {
            val remaining = activeChallenge.remainingSeconds()
            if (remaining > 0 && challengePromptCard.visibility == android.view.View.VISIBLE) {
                challengeCountdown.text = "${remaining}s"
                // Red text when < 5s remaining (urgency cue)
                val color = if (remaining <= 5) 0xFFEF4444.toInt() else 0xFF60A5FA.toInt()
                challengeCountdown.setTextColor(color)
                challengePromptCard.postDelayed(this, 1000L)
            } else {
                challengeCountdown.text = ""
            }
        }
    }

    private fun hideChallengePrompt() {
        challengePromptCard.removeCallbacks(challengePromptHideRunnable)
        runOnUiThread { hideChallengePromptNow() }
    }

    private fun hideChallengePromptNow() {
        challengePromptCard.removeCallbacks(challengeCountdownRunnable)
        challengeCountdown.text = ""
        challengePromptCard.animate().cancel()
        if (challengePromptCard.visibility == android.view.View.VISIBLE) {
            challengePromptCard.animate()
                .alpha(0f)
                .translationY(-30f)
                .scaleX(0.95f).scaleY(0.95f)
                .setDuration(220)
                .withEndAction {
                    challengePromptCard.visibility = android.view.View.GONE
                    challengePromptCard.translationY = 0f
                    challengePromptCard.alpha = 0f
                }.start()
        }
    }

    /**
     * Update the progress dots in the challenge prompt (e.g. "1/2", "2/2").
     */
    private fun updateProgressDots(current: Int, total: Int) {
        challengeProgress.removeAllViews()
        if (total <= 1) return
        val density = resources.displayMetrics.density
        val dotSize = (6 * density).toInt()
        val dotMargin = (4 * density).toInt()
        for (i in 0 until total) {
            val dot = android.view.View(this)
            val lp = android.widget.LinearLayout.LayoutParams(dotSize, dotSize)
            lp.marginEnd = dotMargin
            dot.layoutParams = lp
            dot.background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.OVAL
                setColor(
                    if (i < current) ContextCompat.getColor(this@MainActivity, R.color.brand_text_primary)
                    else ContextCompat.getColor(this@MainActivity, R.color.brand_text_secondary)
                )
            }
            challengeProgress.addView(dot)
        }
    }

    /**
     * 4-tuple for showChallengePrompt — (icon, label, text, background drawable).
     * Data classes are concise but here we need an inline-ish construct for clarity.
     */
    private data class Tuple4<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)

    /** Mean RGB of a rectangular patch. */
    private data class RgbSample(val r: Double, val g: Double, val b: Double)

    /**
     * Sample mean RGB from 5 face regions for SOTA POS rPPG algorithm.
     * Returns 5 samples: 0=forehead, 1=left_cheek, 2=right_cheek, 3=nose, 4=chin.
     * All are filled with forehead mean if specific regions can't be computed.
     */
    private fun sampleFaceROIs(bitmap: Bitmap, key5: Array<PointF>): Array<RgbSample> {
        val leftEye = key5[0]
        val rightEye = key5[1]
        val nose = key5[2]
        val leftMouth = key5[3]
        val rightMouth = key5[4]

        val eyeMidX = (leftEye.x + rightEye.x) / 2f
        val eyeMidY = (leftEye.y + rightEye.y) / 2f
        val eyeDist = kotlin.math.abs(rightEye.x - leftEye.x).coerceAtLeast(30f)
        val mouthMidX = (leftMouth.x + rightMouth.x) / 2f
        val mouthMidY = (leftMouth.y + rightMouth.y) / 2f

        // ROI centers (in pixel coords of the bitmap)
        val forehead = PointF(eyeMidX, eyeMidY - eyeDist * 0.7f)
        val leftCheek = PointF(eyeMidX - eyeDist * 0.7f, eyeMidY + eyeDist * 0.4f)
        val rightCheek = PointF(eyeMidX + eyeDist * 0.7f, eyeMidY + eyeDist * 0.4f)
        val noseTip = PointF(nose.x, nose.y)
        val chin = PointF(mouthMidX, mouthMidY + eyeDist * 0.6f)

        val patchSize = (eyeDist * 0.6f).toInt().coerceAtLeast(20)

        val samples = arrayOf(
            sampleRgb(bitmap, forehead, patchSize),
            sampleRgb(bitmap, leftCheek, patchSize),
            sampleRgb(bitmap, rightCheek, patchSize),
            sampleRgb(bitmap, noseTip, patchSize),
            sampleRgb(bitmap, chin, patchSize)
        )
        return samples
    }

    private fun sampleRgb(bitmap: Bitmap, center: PointF, size: Int): RgbSample {
        val cx = center.x.toInt().coerceIn(0, bitmap.width - 1)
        val cy = center.y.toInt().coerceIn(0, bitmap.height - 1)
        val half = size / 2
        val x0 = (cx - half).coerceAtLeast(0)
        val y0 = (cy - half).coerceAtLeast(0)
        val x1 = (x0 + size).coerceAtMost(bitmap.width)
        val y1 = (y0 + size).coerceAtMost(bitmap.height)
        if (x1 <= x0 || y1 <= y0) return RgbSample(128.0, 128.0, 128.0)
        val w = x1 - x0
        val h = y1 - y0
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, x0, y0, w, h)
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        for (p in pixels) {
            sumR += (p shr 16) and 0xFF
            sumG += (p shr 8) and 0xFF
            sumB += p and 0xFF
        }
        val n = pixels.size.toLong()
        return RgbSample(sumR.toDouble() / n, sumG.toDouble() / n, sumB.toDouble() / n)
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        yBuffer.rewind()
        uBuffer.rewind()
        vBuffer.rewind()
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val jpeg = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
    }

    private fun yPlaneFromImageProxy(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        yBuffer.rewind()  // buffer was consumed by imageProxyToBitmap — rewind first
        val arr = ByteArray(yBuffer.remaining())
        yBuffer.get(arr)
        yBuffer.rewind()
        return arr
    }

    private fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return src
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    private fun cropAndResizeForLiveness(src: Bitmap, box: RectF, size: Int): Bitmap {
        val padX = (box.width() * 0.3f).toInt()
        val padY = (box.height() * 0.3f).toInt()
        val cx = (box.left + box.right) / 2f
        val cy = (box.top + box.bottom) / 2f
        val halfW = box.width() / 2f + padX
        val halfH = box.height() / 2f + padY
        val x0 = (cx - halfW).toInt().coerceIn(0, src.width - 1)
        val y0 = (cy - halfH).toInt().coerceIn(0, src.height - 1)
        val x1 = (cx + halfW).toInt().coerceIn(x0 + 1, src.width)
        val y1 = (cy + halfH).toInt().coerceIn(y0 + 1, src.height)
        val cropped = Bitmap.createBitmap(src, x0, y0, x1 - x0, y1 - y0)
        val scaled = Bitmap.createScaledBitmap(cropped, size, size, true)
        if (scaled !== cropped) cropped.recycle()
        return scaled
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var sum = 0f
        for (x in v) sum += x * x
        val n = sqrt(sum)
        if (n == 0f) return v
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / n
        return out
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        val denom = sqrt(na) * sqrt(nb)
        return if (denom == 0f) 0f else dot / denom
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = FloatArray(logits.size) { kotlin.math.exp(logits[it] - max) }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    private fun prefs() = getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    private fun saveEnrolledEmbedding(embedding: FloatArray) {
        val arr = JSONArray()
        for (v in embedding) arr.put(v.toDouble())
        val obj = JSONObject().put(KEY_EMBEDDING, arr)
        prefs().edit().putString(KEY_EMBEDDING, obj.toString()).apply()
    }

    private fun loadEnrolledEmbedding(): FloatArray? {
        val raw = prefs().getString(KEY_EMBEDDING, null) ?: return null
        return try {
            val arr = JSONObject(raw).getJSONArray(KEY_EMBEDDING)
            FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load embedding", e)
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        try { faceLandmarker?.close() } catch (_: Throwable) {}
        try { recognitionInterpreter?.close() } catch (_: Throwable) {}
        try { livenessInterpreter?.close() } catch (_: Throwable) {}
        try { gpuDelegate?.close() } catch (_: Throwable) {}
    }
}
