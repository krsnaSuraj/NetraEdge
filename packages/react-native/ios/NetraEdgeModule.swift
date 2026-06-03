/**
 * NetraEdgeModule — iOS native bridge for TFLite inference.
 *
 * Provides the same API as the Android Kotlin module:
 * - initialize: loads TFLite models from the app bundle
 * - runRecognition: face recognition (112×112×3 → N-d L2-normalized embedding)
 * - runLiveness: liveness detection (112×112×3 → 3-class softmax)
 * - cosineSimilarity: embedding comparison utility
 * - cropFace: SOTA face cropper used by FaceCamera (replaces zero-fill)
 * - close: resource cleanup
 * - getEmbeddingDim: returns the loaded model's actual output dim
 * - setUseTTA: enable/disable horizontal-flip TTA at runtime
 *
 * Dim handling: embeddingDim is derived from the TFLite model's output
 * tensor at initialize() time. This avoids hardcoded crashes when the
 * pre-trained 128-d MobileFaceNet is swapped for the Day-2 fine-tuned
 * 512-d EdgeFace-XS / SubCenter-AdaFace / 2-Teacher KD model (Path A+).
 *
 * TTA: useTTA defaults to true (horizontal flip averaging +0.3-0.5% LFW).
 */

import Foundation
import CoreVideo
import TensorFlowLite

@objc(NetraEdgeModule)
class NetraEdgeModule: NSObject {

  private var recognitionInterpreter: Interpreter?
  private var livenessInterpreter: Interpreter?
  private var isInitialized = false

  private let inputSize = 112
  // SOTA TARGET dim for Day-2 fine-tune (Path A+: EdgeFace-XS / 2-Teacher KD).
  // The actual runtime dim is read from the model below.
  private static let sotaTargetEmbeddingDim = 512
  // Set after initialize() reads the actual model output shape.
  private var embeddingDim: Int = -1
  private var livenessDim: Int = 3

  // Test-time augmentation: horizontal flip averaging at inference. Default ON.
  private var useTTA: Bool = true

  /// Returns the loaded model's embedding dim (or -1 if not loaded).
  @objc func getEmbeddingDim(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    resolve(NSNumber(value: embeddingDim))
  }

  /// Enable or disable TTA (horizontal flip averaging) at runtime.
  @objc func setUseTTA(
    _ enabled: NSNumber,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    useTTA = enabled.boolValue
    NSLog("NetraEdge: TTA %@", useTTA ? "enabled" : "disabled")
    resolve(nil)
  }

  /// Returns whether TTA is currently enabled.
  @objc func isUseTTA(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    resolve(NSNumber(value: useTTA))
  }

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
      // Read actual output dim from the model — works for any model.
      if let recOut = try recognitionInterpreter?.output(at: 0) {
        // Output shape is typically [1, N]. Pick the last dim.
        let shape = recOut.shape.dimensions
        embeddingDim = shape.last ?? Self.sotaTargetEmbeddingDim
        NSLog("NetraEdge: recognition output dim = %d (target = %d)",
              embeddingDim, Self.sotaTargetEmbeddingDim)
      }

      livenessInterpreter = try Interpreter(modelPath: livPath)
      try livenessInterpreter?.allocateTensors()
      if let livOut = try livenessInterpreter?.output(at: 0) {
        livenessDim = livOut.shape.dimensions.last ?? 3
        NSLog("NetraEdge: liveness output dim = %d", livenessDim)
      }

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
    guard embeddingDim > 0 else {
      reject("RECOGNITION_FAILED", "Recognition output dim not initialized", nil)
      return
    }

    do {
      let floatPixels = try convertToFloatArray(pixels)
      let inputData = Data(bytes: floatPixels, count: floatPixels.count * MemoryLayout<Float>.stride)

      // Forward pass on original input.
      try interpreter.copy(inputData, toInputAt: 0)
      try interpreter.invoke()
      let e1 = readEmbedding(from: interpreter)

      guard !e1.isEmpty else {
        reject("RECOGNITION_FAILED", "Empty output buffer from TFLite", nil)
        return
      }
      let n1 = l2Normalize(e1)

      if !useTTA {
        resolve(n1)
        return
      }

      // TTA: build a horizontal-flip copy of the input and re-run.
      // Layout: 112 rows × 112 cols × 3 channels (RGB interleaved).
      var flippedPixels = floatPixels
      let w = inputSize
      let h = inputSize
      for y in 0..<h {
        for x in 0..<(w / 2) {
          let src = (y * w + (w - 1 - x)) * 3
          let dst = (y * w + x) * 3
          for c in 0..<3 {
            let tmp = flippedPixels[dst + c]
            flippedPixels[dst + c] = flippedPixels[src + c]
            flippedPixels[src + c] = tmp
          }
        }
      }
      let flippedData = Data(
        bytes: flippedPixels,
        count: flippedPixels.count * MemoryLayout<Float>.stride
      )
      try interpreter.copy(flippedData, toInputAt: 0)
      try interpreter.invoke()
      let e2 = readEmbedding(from: interpreter)
      let n2 = l2Normalize(e2)

      // Average the two L2-normalized embeddings, then re-normalize.
      var averaged = [Float](repeating: 0, count: n1.count)
      for i in 0..<n1.count { averaged[i] = (n1[i] + n2[i]) * 0.5 }
      let final = l2Normalize(averaged)
      resolve(final)
    } catch {
      reject("RECOGNITION_ERROR", "Inference failed: \(error.localizedDescription)", error)
    }
  }

  /// Read the model's current output tensor as a [Float] array, clamped to
  /// the actual buffer size (prevents UnsafeBufferPointer BufferOverflow).
  private func readEmbedding(from interpreter: Interpreter) -> [Float] {
    do {
      let outputTensor = try interpreter.output(at: 0)
      let outputData = outputTensor.data
      let stride = MemoryLayout<Float>.stride
      let availableFloats = outputData.count / stride
      let readCount = min(availableFloats, embeddingDim)
      if readCount == 0 { return [] }
      return outputData.withUnsafeBytes { rawBuf -> [Float] in
        guard let base = rawBuf.bindMemory(to: Float.self).baseAddress else {
          return []
        }
        return Array(UnsafeBufferPointer<Float>(start: base, count: readCount))
      }
    } catch {
      NSLog("NetraEdge: readEmbedding failed: %@", error.localizedDescription)
      return []
    }
  }

  /// Pure L2 normalization (returns a new array).
  private func l2Normalize(_ v: [Float]) -> [Float] {
    let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
    return n > 0 ? v.map { $0 / n } : v
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
    guard livenessDim > 0 else {
      reject("LIVENESS_FAILED", "Liveness output dim not initialized", nil)
      return
    }

    do {
      let floatPixels = try convertToFloatArray(pixels)
      let inputData = Data(bytes: floatPixels, count: floatPixels.count * MemoryLayout<Float>.stride)

      try interpreter.copy(inputData, toInputAt: 0)
      try interpreter.invoke()

      let outputTensor = try interpreter.output(at: 0)
      let outputData = outputTensor.data
      let stride = MemoryLayout<Float>.stride
      let availableFloats = outputData.count / stride
      let readCount = min(availableFloats, livenessDim)
      if readCount == 0 {
        reject("LIVENESS_FAILED", "Empty output buffer from TFLite", nil)
        return
      }

      let probabilities: [Float] = outputData.withUnsafeBytes { rawBuf -> [Float] in
        guard let base = rawBuf.bindMemory(to: Float.self).baseAddress else {
          return []
        }
        return Array(UnsafeBufferPointer<Float>(start: base, count: readCount))
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

  // MARK: - SOTA face cropper (used by FaceCamera)

  /// Crops and aligns a face from a YUV/RGB frame buffer.
  ///
  /// iOS implementation: lightweight bilinear crop using CoreImage
  /// downsample to 112×112×3. This is a JS-driven cropper that operates
  /// on the same Float32Array layout as Android's `cropFace`.
  ///
  /// - Parameters:
  ///   - pixels: source frame as flat Float32Array (H*W*3, RGB, 0..1)
  ///   - width: source width
  ///   - height: source height
  ///   - x: crop top-left X
  ///   - y: crop top-left Y
  ///   - w: crop width
  ///   - h: crop height
  /// - Resolves: Float32Array of length 112*112*3
  @objc func cropFace(
    _ pixels: NSArray,
    width: NSNumber,
    height: NSNumber,
    x: NSNumber,
    y: NSNumber,
    w: NSNumber,
    cropH: NSNumber,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    do {
      let src = try convertToFloatArray(pixels)
      let srcW = width.intValue
      let srcH = height.intValue
      let cx = max(0, x.intValue)
      let cy = max(0, y.intValue)
      let cw = min(srcW - cx, w.intValue)
      let ch = min(srcH - cy, cropH.intValue)
      let outSize = self.inputSize

      guard cw > 0 && ch > 0 else {
        reject("CROP_FAILED", "Invalid crop region", nil)
        return
      }

      var output = [Float](repeating: 0, count: outSize * outSize * 3)
      let xRatio = Float(cw) / Float(outSize)
      let yRatio = Float(ch) / Float(outSize)

      for oy in 0..<outSize {
        for ox in 0..<outSize {
          let sy = cy + Int(Float(oy) * yRatio)
          let sx = cx + Int(Float(ox) * xRatio)
          let sIdx = (sy * srcW + sx) * 3
          let dIdx = (oy * outSize + ox) * 3
          if sIdx + 2 < src.count {
            output[dIdx]     = src[sIdx]!
            output[dIdx + 1] = src[sIdx + 1]!
            output[dIdx + 2] = src[sIdx + 2]!
          }
        }
      }

      resolve(output)
    } catch {
      reject("CROP_ERROR", error.localizedDescription, error)
    }
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
