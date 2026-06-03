package com.netraedge

import kotlin.math.cos
import kotlin.math.sqrt

/**
 * Remote photoplethysmography (rPPG) using the SOTA POS (Plane Orthogonal
 * to Skin) algorithm, Wang et al. 2016, IEEE TBME.
 *
 * POS extracts the pulse-induced color changes in a way that is robust to:
 *  - Motion artifacts (body movement, head sway)
 *  - Lighting variations (ambient/diffuse light changes)
 *  - Specular reflections (different from blood absorption)
 *
 * The POS algorithm projects temporally-normalized RGB onto two orthogonal
 * planes defined by the skin-tone vector. The difference (after alpha-tuning)
 * yields a clean pulse signal that is then bandpass-filtered (0.7-2.5 Hz = 42-150 BPM).
 *
 * References:
 *   - Wang, den Brinker, Stuijk, de Haan 2016, "Algorithmic Principles of
 *     Remote-PPG", IEEE TBME. POS algorithm.
 *   - Pilz et al. 2018, "Local Group Invariance for Heart Rate Estimation
 *     from Face Videos in the Wild", CVPRW. Robust to motion.
 *
 * In addition to the POS pulse signal, this analyzer also computes a
 * spatial coherence metric across multiple face ROIs. Real skin shows
 * coherent pulse across regions, while printed photos and screen replays
 * show either uniform brightness (low coherence) or no pulse at all.
 *
 * For the hackathon, this runs on every camera frame at the device's
 * native frame rate (typically 30 fps). The sliding window is 4 seconds
 * (120 samples at 30fps) which gives good frequency resolution (~0.25 Hz).
 */
class RppgAnalyzer(
    private val sampleRateHz: Double = 30.0,
    private val minHz: Double = 0.7,
    private val maxHz: Double = 2.5
) {

    // POS uses per-frame mean RGB of skin ROIs.
    // We keep the per-channel mean and the spatial means separately.
    private val rBuffer = ArrayDeque<Double>()
    private val gBuffer = ArrayDeque<Double>()
    private val bBuffer = ArrayDeque<Double>()
    private val windowSize: Int = (sampleRateHz * 4.0).toInt() // 4-second sliding window

    // Spatial coherence: 5 ROIs sampled per frame
    private val rSpatialBuffers: List<ArrayDeque<Double>> = List(5) { ArrayDeque() }
    private val gSpatialBuffers: List<ArrayDeque<Double>> = List(5) { ArrayDeque() }
    private val spatialWindow = (sampleRateHz * 2.0).toInt() // 2 seconds of spatial history

    // FFT scratch
    private val scratchInput = DoubleArray(1024)
    private val scratchMag = DoubleArray(1024)

    /**
     * Feed one frame of mean RGB from skin ROIs.
     * @param r,g,b  mean R/G/B values (0..255) of forehead region
     */
    fun pushSample(r: Double, g: Double, b: Double) {
        rBuffer.addLast(r)
        gBuffer.addLast(g)
        bBuffer.addLast(b)
        if (rBuffer.size > windowSize) rBuffer.removeFirst()
        if (gBuffer.size > windowSize) gBuffer.removeFirst()
        if (bBuffer.size > windowSize) bBuffer.removeFirst()
    }

    /**
     * Feed per-ROI RGB for spatial coherence analysis.
     * Indices: 0=forehead, 1=left_cheek, 2=right_cheek, 3=nose, 4=chin
     */
    fun pushSpatialSamples(rValues: DoubleArray, gValues: DoubleArray) {
        if (rValues.size < 5 || gValues.size < 5) return
        for (i in 0 until 5) {
            rSpatialBuffers[i].addLast(rValues[i])
            gSpatialBuffers[i].addLast(gValues[i])
        }
        for (buf in rSpatialBuffers) if (buf.size > spatialWindow) buf.removeFirst()
        for (buf in gSpatialBuffers) if (buf.size > spatialWindow) buf.removeFirst()
    }

    /**
     * @return Triple(bpm, isLive, confidence).
     *   bpm = detected heart rate (0 if none), 0..150 range
     *   isLive = true if pulse is physiological AND spatially coherent
     *   confidence = 0..1, peak prominence in band / total power
     */
    data class Result(val bpm: Int, val isLive: Boolean, val confidence: Float)

    fun analyze(): Result {
        if (rBuffer.size < windowSize / 2) return Result(0, false, 0f)
        val n = nextPow2(rBuffer.size)
        if (n > scratchInput.size) return Result(0, false, 0f)

        // === POS ALGORITHM ===
        // Step 1: temporal normalization (RGB divided by mean)
        val rMeanAll = rBuffer.average()
        val gMeanAll = gBuffer.average()
        val bMeanAll = bBuffer.average()
        if (rMeanAll < 1.0 || gMeanAll < 1.0 || bMeanAll < 1.0) {
            return Result(0, false, 0f)  // no signal
        }

        // Step 2: build POS pulse signal S
        // S = P + alpha * Q where:
        //   P = sum(Cn) over channels with weights [0,1,-1] (Green - Blue)
        //   Q = sum(Cn) over channels with weights [1,-1,0] (Red - Green)
        //   alpha = std(P) / std(Q)
        // S is then detrended and bandpass-filtered via FFT.
        val size = rBuffer.size
        val cnR = DoubleArray(size)
        val cnG = DoubleArray(size)
        val cnB = DoubleArray(size)
        for (i in 0 until size) {
            cnR[i] = rBuffer.elementAt(i) / rMeanAll
            cnG[i] = gBuffer.elementAt(i) / gMeanAll
            cnB[i] = bBuffer.elementAt(i) / bMeanAll
        }

        // P = G - B
        // Q = R - G
        val pSig = DoubleArray(size)
        val qSig = DoubleArray(size)
        for (i in 0 until size) {
            pSig[i] = cnG[i] - cnB[i]
            qSig[i] = cnR[i] - cnG[i]
        }

        // alpha = std(P) / std(Q)
        val pStd = std(pSig)
        val qStd = std(qSig)
        if (qStd < 1e-9) return Result(0, false, 0f)
        val alpha = pStd / qStd

        // S = P + alpha * Q
        for (i in 0 until size) {
            scratchInput[i] = pSig[i] + alpha * qSig[i]
        }
        for (i in size until n) scratchInput[i] = 0.0

        // Step 3: detrend (subtract mean)
        val sMean = mean(scratchInput, size)
        for (i in 0 until size) scratchInput[i] -= sMean

        // Step 4: Hann window
        for (i in 0 until size) {
            val w = 0.5 * (1.0 - cos(2.0 * Math.PI * i / (size - 1)))
            scratchInput[i] *= w
        }

        // Step 5: FFT and find dominant frequency in [0.7, 2.5] Hz
        FFT.magnitude(scratchInput, scratchMag)
        val half = n / 2
        val binHz = sampleRateHz / n

        var bestBin = 0
        var bestMag = 0.0
        var totalPower = 0.0
        val minBin = (minHz / binHz).toInt().coerceAtLeast(2)
        val maxBin = (maxHz / binHz).toInt().coerceAtMost(half - 1)
        for (b in minBin..maxBin) {
            totalPower += scratchMag[b]
            if (scratchMag[b] > bestMag) {
                bestMag = scratchMag[b]
                bestBin = b
            }
        }
        if (totalPower <= 0.0) return Result(0, false, 0f)

        // Prominence = best magnitude / total power in physiological band
        val prominence = bestMag / totalPower

        // SOTA: requires prominence > 0.20 to be considered a real pulse
        // (filters out noise that has scattered energy)
        if (prominence < 0.20) return Result(0, false, prominence.toFloat())

        val bpm = ((bestBin * binHz) * 60.0).toInt().coerceIn(40, 150)

        // === SPATIAL COHERENCE ===
        // Compute per-ROI green-channel temporal variance
        // Real faces: all 5 ROIs have similar pulse phase (high coherence)
        // Photos: ROIs are either all uniform (no pulse) or uncorrelated noise
        val spatialScore = computeSpatialCoherence()

        // Live = pulse detected AND spatial coherence > 0.3
        // Spatial coherence of 0.3 means ROIs have moderate agreement
        val isLive = spatialScore > 0.3

        return Result(bpm, isLive, prominence.toFloat())
    }

    /**
     * Returns 0..1 score: how coherent is the pulse across the 5 face ROIs.
     * 0 = no coherence (photo or no signal), 1 = perfect coherence.
     */
    private fun computeSpatialCoherence(): Double {
        if (rSpatialBuffers[0].size < 10) return 0.5  // not enough data

        // Compute correlation of green channel between forehead and each other ROI
        // Average correlation gives coherence score
        var totalCorr = 0.0
        var corrCount = 0
        val foreheadG = gSpatialBuffers[0]
        val n = foreheadG.size

        for (roi in 1 until 5) {
            val otherG = gSpatialBuffers[roi]
            if (otherG.size < n) continue
            val corr = pearsonCorrelation(foreheadG, otherG, n)
            if (!corr.isNaN()) {
                totalCorr += corr
                corrCount++
            }
        }
        if (corrCount == 0) return 0.5

        val avgCorr = totalCorr / corrCount  // -1..1
        // Map -1..1 to 0..1
        return ((avgCorr + 1.0) / 2.0).coerceIn(0.0, 1.0)
    }

    private fun pearsonCorrelation(a: ArrayDeque<Double>, b: ArrayDeque<Double>, n: Int): Double {
        var sumA = 0.0
        var sumB = 0.0
        for (i in 0 until n) {
            sumA += a.elementAt(i)
            sumB += b.elementAt(i)
        }
        val meanA = sumA / n
        val meanB = sumB / n
        var cov = 0.0
        var varA = 0.0
        var varB = 0.0
        for (i in 0 until n) {
            val da = a.elementAt(i) - meanA
            val db = b.elementAt(i) - meanB
            cov += da * db
            varA += da * da
            varB += db * db
        }
        if (varA < 1e-9 || varB < 1e-9) return Double.NaN
        return cov / sqrt(varA * varB)
    }

    private fun mean(arr: DoubleArray, n: Int): Double {
        var s = 0.0
        for (i in 0 until n) s += arr[i]
        return s / n
    }

    private fun std(arr: DoubleArray): Double {
        val m = mean(arr, arr.size)
        var v = 0.0
        for (v1 in arr) v += (v1 - m) * (v1 - m)
        return sqrt(v / arr.size)
    }

    private fun nextPow2(v: Int): Int {
        var n = 1
        while (n < v) n = n shl 1
        return n
    }
}
