/**
 * NetraEdgeModule — iOS native bridge for TFLite inference.
 *
 * Provides the same API as the Android Kotlin module:
 * - initialize: loads TFLite models from the app bundle
 * - runRecognition: face recognition (112×112×3 → 128-d embedding)
 * - runLiveness: liveness detection (112×112×3 → 3-class softmax)
 * - cosineSimilarity: embedding comparison utility
 * - close: resource cleanup
 */

import Foundation
import TensorFlowLite

@objc(NetraEdgeModule)
class NetraEdgeModule: NSObject {

  private var recognitionInterpreter: Interpreter?
  private var livenessInterpreter: Interpreter?
  private var isInitialized = false

  private let inputSize = 112
  private let embeddingDim = 128
  private let livenessClasses = 3

  // MARK: - React Native bridge methods

  @objc func initialize(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    do {
      guard let recPath = Bundle.main.path(
        forResource: "face_recognition",
        ofType: "tflite"
      ) else {
        reject("INIT_FAILED", "face_recognition.tflite not found in bundle", nil)
        return
      }

      guard let livPath = Bundle.main.path(
        forResource: "liveness_detector",
        ofType: "tflite"
      ) else {
        reject("INIT_FAILED", "liveness_detector.tflite not found in bundle", nil)
        return
      }

      recognitionInterpreter = try Interpreter(modelPath: recPath)
      try recognitionInterpreter?.allocateTensors()

      livenessInterpreter = try Interpreter(modelPath: livPath)
      try livenessInterpreter?.allocateTensors()

      isInitialized = true
      resolve(true)
    } catch {
      reject("INIT_ERROR", "Failed to load TFLite models: \(error.localizedDescription)", error)
    }
  }

  @objc func isInitialized(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    resolve(isInitialized)
  }

  @objc func runRecognition(
    _ pixels: NSArray,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    guard isInitialized, let interpreter = recognitionInterpreter else {
      reject("RECOGNITION_FAILED", "Recognition model not initialized", nil)
      return
    }

    do {
      let floatPixels = try convertToFloatArray(pixels)
      let inputData = Data(bytes: floatPixels, count: floatPixels.count * MemoryLayout<Float>.stride)

      try interpreter.copy(inputData, toInputAt: 0)
      try interpreter.invoke()

      let outputTensor = try interpreter.output(at: 0)
      let outputData = outputTensor.data
      let embedding: [Float] = outputData.withUnsafeBytes {
        Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: self.embeddingDim))
      }

      // L2 normalize
      let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
      let normalized = norm > 0 ? embedding.map { $0 / norm } : embedding

      resolve(normalized)
    } catch {
      reject("RECOGNITION_ERROR", "Inference failed: \(error.localizedDescription)", error)
    }
  }

  @objc func runLiveness(
    _ pixels: NSArray,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    guard isInitialized, let interpreter = livenessInterpreter else {
      reject("LIVENESS_FAILED", "Liveness model not initialized", nil)
      return
    }

    do {
      let floatPixels = try convertToFloatArray(pixels)
      let inputData = Data(bytes: floatPixels, count: floatPixels.count * MemoryLayout<Float>.stride)

      try interpreter.copy(inputData, toInputAt: 0)
      try interpreter.invoke()

      let outputTensor = try interpreter.output(at: 0)
      let outputData = outputTensor.data
      let probabilities: [Float] = outputData.withUnsafeBytes {
        Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: self.livenessClasses))
      }

      resolve(probabilities)
    } catch {
      reject("LIVENESS_ERROR", "Inference failed: \(error.localizedDescription)", error)
    }
  }

  @objc func cosineSimilarity(
    _ a: NSArray,
    b: NSArray,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    do {
      let aFloats = try convertToFloatArray(a)
      let bFloats = try convertToFloatArray(b)

      guard aFloats.count == bFloats.count else {
        reject("SIMILARITY_ERROR", "Dimension mismatch", nil)
        return
      }

      let dot = zip(aFloats, bFloats).reduce(Float(0)) { $0 + $1.0 * $1.1 }
      let clamped = max(-1, min(1, dot))
      resolve(Double(clamped))
    } catch {
      reject("SIMILARITY_ERROR", error.localizedDescription, error)
    }
  }

  @objc func close() {
    recognitionInterpreter = nil
    livenessInterpreter = nil
    isInitialized = false
  }

  // MARK: - React Native setup

  @objc static func requiresMainQueueSetup() -> Bool {
    return false
  }

  // MARK: - Private helpers

  private func convertToFloatArray(_ nsArray: NSArray) throws -> [Float] {
    var floats = [Float]()
    floats.reserveCapacity(nsArray.count)
    for item in nsArray {
      guard let number = item as? NSNumber else {
        throw NSError(
          domain: "NetraEdge",
          code: -1,
          userInfo: [NSLocalizedDescriptionKey: "Expected numeric array"]
        )
      }
      floats.append(number.floatValue)
    }
    return floats
  }
}
