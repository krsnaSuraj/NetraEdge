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

            promise.resolve(NetraEdgeModule.cosineSimilarity(aFloat, bFloat).toDouble())
        } catch (e: Exception) {
            promise.reject("SIMILARITY_ERROR", e.message)
        }
    }

    @ReactMethod
    fun close() {
        module.close()
    }
}
