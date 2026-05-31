package com.netraedge

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Size
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var cameraPreview: PreviewView
    private lateinit var statusText: TextView
    private lateinit var infoText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var recInterpreter: Interpreter? = null
    private var livInterpreter: Interpreter? = null
    private var enrolledEmbedding: FloatArray? = null
    private var enrolledName: String = ""
    private var isEnrolling = false
    private var enrollFrames = 0
    private val ENROLL_TARGET = 10
    private val MATCH_THRESHOLD = 0.75f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraPreview = findViewById(R.id.camera_preview)
        statusText = findViewById(R.id.status_text)
        infoText = findViewById(R.id.info_text)
        cameraExecutor = Executors.newSingleThreadExecutor()

        try {
            recInterpreter = Interpreter(loadModelFile("face_recognition.tflite"))
            livInterpreter = Interpreter(loadModelFile("liveness_detector.tflite"))
            infoText.text = "Models loaded | Ready"
        } catch (e: Exception) {
            infoText.text = "Error: ${e.message}"
        }

        findViewById<Button>(R.id.btn_enroll).setOnClickListener {
            isEnrolling = true
            enrollFrames = 0
            statusText.text = "Enrolling... Look at camera"
        }

        findViewById<Button>(R.id.btn_verify).setOnClickListener {
            isEnrolling = false
            statusText.text = "Verifying... Look at camera"
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 10)
        }
    }

    private fun loadModelFile(name: String): MappedByteBuffer {
        val fd = assets.openFd(name)
        val stream = FileInputStream(fd.fileDescriptor)
        return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == 10 && results.isNotEmpty() && results[0] == PackageManager.PERMISSION_GRANTED) startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(cameraPreview.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(cameraExecutor) { frame -> processFrame(frame) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(frame: ImageProxy) {
        val bitmap = frame.toBitmap() ?: run { frame.close(); return }
        val resized = Bitmap.createScaledBitmap(bitmap, 112, 112, true)
        val input = bitmapToBuffer(resized)

        val livOutput = Array(1) { FloatArray(3) }
        livInterpreter?.run(input, livOutput)
        val liveScore = livOutput[0][0]
        val isLive = liveScore > 0.5f

        val recOutput = Array(1) { FloatArray(128) }
        recInterpreter?.run(input, recOutput)
        val embedding = recOutput[0]
        l2Normalize(embedding)

        runOnUiThread {
            if (isEnrolling) {
                enrollFrames++
                statusText.text = "Enrolling: $enrollFrames/$ENROLL_TARGET"
                if (enrollFrames >= ENROLL_TARGET) {
                    enrolledEmbedding = embedding.copyOf()
                    enrolledName = "User_${System.currentTimeMillis() % 1000}"
                    isEnrolling = false
                    statusText.text = "Enrolled: $enrolledName"
                    infoText.text = "Liveness: ${(liveScore * 100).toInt()}% | Ready to verify"
                }
            } else {
                if (!isLive) {
                    statusText.text = "SPOOF DETECTED"
                    infoText.text = "Use a real face"
                } else if (enrolledEmbedding != null) {
                    val sim = cosineSimilarity(embedding, enrolledEmbedding!!)
                    if (sim >= MATCH_THRESHOLD) {
                        statusText.text = "VERIFIED: $enrolledName"
                        infoText.text = "Confidence: ${(sim * 100).toInt()}% | Liveness: OK"
                    } else {
                        statusText.text = "NOT RECOGNIZED"
                        infoText.text = "Confidence: ${(sim * 100).toInt()}%"
                    }
                } else {
                    statusText.text = "No user enrolled"
                    infoText.text = "Tap Enroll first"
                }
            }
        }
        frame.close()
    }

    private fun bitmapToBuffer(bmp: Bitmap): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(112 * 112 * 3 * 4)
        buf.order(ByteOrder.nativeOrder())
        val pixels = IntArray(112 * 112)
        bmp.getPixels(pixels, 0, 112, 0, 0, 112, 112)
        for (p in pixels) {
            buf.putFloat(((p shr 16) and 0xFF) / 255f)
            buf.putFloat(((p shr 8) and 0xFF) / 255f)
            buf.putFloat((p and 0xFF) / 255f)
        }
        buf.rewind()
        return buf
    }

    private fun l2Normalize(v: FloatArray) {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm > 0) for (i in v.indices) v[i] /= norm
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot.coerceIn(-1f, 1f)
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val y = planes[0].buffer
        val u = planes[1].buffer
        val v = planes[2].buffer
        val ySize = y.remaining()
        val uvSize = u.remaining()
        val nv21 = ByteArray(ySize + uvSize)
        y.get(nv21, 0, ySize)
        v.get(nv21, ySize, uvSize)
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val bytes = out.toByteArray()
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        recInterpreter?.close()
        livInterpreter?.close()
    }
}
