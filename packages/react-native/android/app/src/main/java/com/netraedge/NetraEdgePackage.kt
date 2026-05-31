package com.netraedge

import android.content.Context
import com.facebook.react.bridge.*

/**
 * React Native bridge for NetraEdge native module.
 *
 * Exposes TFLite inference to JavaScript:
 * - initialize() — load models from assets
 * - runRecognition(pixels) — face recognition
 * - runLiveness(pixels) — liveness detection
 * - cosineSimilarity(a, b) — embedding comparison
 */
class NetraEdgePackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        return listOf(NetraEdgeNativeModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}

class NetraEdgeNativeModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val module = NetraEdgeModule()

    override fun getName(): String = "NetraEdgeModule"

    @ReactMethod
    fun initialize(promise: Promise) {
        try {
            val success = module.initialize(reactContext)
            if (success) {
                promise.resolve(true)
            } else {
                promise.reject("INIT_FAILED", "Failed to load TFLite models")
            }
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", e.message)
        }
    }

    @ReactMethod
    fun isInitialized(promise: Promise) {
        promise.resolve(module.isInitialized)
    }

    @ReactMethod
    fun runRecognition(pixels: ReadableArray, promise: Promise) {
        try {
            val floatArray = FloatArray(pixels.size())
            for (i in 0 until pixels.size()) {
                floatArray[i] = pixels.getDouble(i).toFloat()
            }

            val result = module.runRecognition(floatArray)
            if (result != null) {
                val array = Arguments.createArray()
                for (v in result) {
                    array.pushDouble(v.toDouble())
                }
                promise.resolve(array)
            } else {
                promise.reject("RECOGNITION_FAILED", "Inference failed")
            }
        } catch (e: Exception) {
            promise.reject("RECOGNITION_ERROR", e.message)
        }
    }

    @ReactMethod
    fun runLiveness(pixels: ReadableArray, promise: Promise) {
        try {
            val floatArray = FloatArray(pixels.size())
            for (i in 0 until pixels.size()) {
                floatArray[i] = pixels.getDouble(i).toFloat()
            }

            val result = module.runLiveness(floatArray)
            if (result != null) {
                val array = Arguments.createArray()
                for (v in result) {
                    array.pushDouble(v.toDouble())
                }
                promise.resolve(array)
            } else {
                promise.reject("LIVENESS_FAILED", "Inference failed")
            }
        } catch (e: Exception) {
            promise.reject("LIVENESS_ERROR", e.message)
        }
    }

    @ReactMethod
    fun cosineSimilarity(a: ReadableArray, b: ReadableArray, promise: Promise) {
        try {
            val aFloat = FloatArray(a.size())
            val bFloat = FloatArray(b.size())
            for (i in 0 until a.size()) aFloat[i] = a.getDouble(i).toFloat()
            for (i in 0 until b.size()) bFloat[i] = b.getDouble(i).toFloat()

            promise.resolve(module.cosineSimilarity(aFloat, bFloat).toDouble())
        } catch (e: Exception) {
            promise.reject("SIMILARITY_ERROR", e.message)
        }
    }

    /**
     * Crop face from raw frame data, resize to 112x112, normalize to 0-1.
     * This is the CRITICAL bridge between camera and face pipeline.
     *
     * @param imageData Raw RGB pixel data (flat float array)
     * @param frameWidth Width of camera frame
     * @param frameHeight Height of camera frame
     * @param faceX Face bounding box X
     * @param faceY Face bounding box Y
     * @param faceWidth Face bounding box width
     * @param faceHeight Face bounding box height
     */
    @ReactMethod
    fun cropFace(
        imageData: ReadableArray,
        frameWidth: Int,
        frameHeight: Int,
        faceX: Int,
        faceY: Int,
        faceWidth: Int,
        faceHeight: Int,
        promise: Promise
    ) {
        try {
            val floatArray = FloatArray(imageData.size())
            for (i in 0 until imageData.size()) {
                floatArray[i] = imageData.getDouble(i).toFloat()
            }

            val result = module.cropFace(
                floatArray, frameWidth, frameHeight,
                faceX, faceY, faceWidth, faceHeight
            )

            if (result != null) {
                val array = Arguments.createArray()
                for (v in result) {
                    array.pushDouble(v.toDouble())
                }
                promise.resolve(array)
            } else {
                promise.reject("CROP_FAILED", "Face crop failed")
            }
        } catch (e: Exception) {
            promise.reject("CROP_ERROR", e.message)
        }
    }

    /**
     * Convert YUV_420_888 byte data to normalized RGB float array.
     */
    @ReactMethod
    fun yuvToNormalized(yuvData: ReadableArray, width: Int, height: Int, promise: Promise) {
        try {
            val byteArray = ByteArray(yuvData.size())
            for (i in 0 until yuvData.size()) {
                byteArray[i] = yuvData.getDouble(i).toInt().toByte()
            }

            val result = module.yuvToNormalizedFloats(byteArray, width, height)
            if (result != null) {
                val array = Arguments.createArray()
                for (v in result) {
                    array.pushDouble(v.toDouble())
                }
                promise.resolve(array)
            } else {
                promise.reject("YUV_FAILED", "YUV conversion failed")
            }
        } catch (e: Exception) {
            promise.reject("YUV_ERROR", e.message)
        }
    }

    @ReactMethod
    fun close() {
        module.close()
    }
}
    }

    @ReactMethod
    fun close() {
        module.close()
    }
}
