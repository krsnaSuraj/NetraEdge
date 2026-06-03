package com.netraedge

import kotlin.math.cos

/**
 * Remote photoplethysmography (rPPG) — recovers a heart-rate signal from
 * the mean green-channel intensity of the face region across a sliding
 * window, then bandpass-filters 0.7–2.5 Hz and reports the dominant
 * frequency as BPM.
 *
 * Reference: Poh, McDuff & Picard 2010, "Advancements in Noncontact,
 * Multiparameter Physiological Measurements Using a Webcam".
 *
 * Two usage patterns are supported:
 *   1) Instance API (CameraX demo): pushSample() per frame, analyze() once
 *      the window is full.
 *   2) Static API (React Native bridge): rppgAnalyze() on a JS-built
 *      array of green-channel means.
 */
class RppgAnalyzer(
    private val sampleRateHz: Double = 30.0,
    private val minHz: Double = 0.7,
    private val maxHz: Double = 2.5
) {

    private val window = ArrayDeque<Double>()
    private val windowSize: Int = (sampleRateHz * 3.5).toInt()
    private val scratchInput = DoubleArray(128)
    private val scratchMag = DoubleArray(128)

    fun pushSample(greenValue: Double) {
        window.addLast(greenValue)
        if (window.size > windowSize) window.removeFirst()
    }

    /** Instance analyze — used by the standalone CameraX demo. */
    fun analyze(): Pair<Int, Boolean> {
        if (window.size < windowSize / 2) return 0 to false
        val n = nextPow2(window.size)
        if (n > scratchInput.size) return 0 to false

        val mean = window.average()
        for (i in 0 until window.size) scratchInput[i] = window.elementAt(i) - mean
        for (i in window.size until n) scratchInput[i] = 0.0

        for (i in 0 until window.size) {
            val w = 0.5 * (1.0 - cos(2.0 * Math.PI * i / (window.size - 1)))
            scratchInput[i] *= w
        }
        FFT.magnitude(scratchInput, scratchMag)
        return findBpm(scratchMag, n)
    }

    companion object {
        /**
         * Static analyze — used by the React Native bridge.
         *
         * @param input   Hann-windowed, detrended signal (length = n)
         * @param mag     pre-allocated output magnitude buffer (length = n)
         * @param sampleHz sampling rate (typically 30 fps for camera)
         * @param n       number of samples in the input
         */
        fun analyzeWindow(input: DoubleArray, mag: DoubleArray, sampleHz: Double, n: Int) {
            if (n > input.size) throw IllegalArgumentException("n > input.size")
            if (n > mag.size) throw IllegalArgumentException("n > mag.size")
            FFT.magnitude(input, mag)
        }

        /** Find the dominant frequency in 0.7-2.5 Hz from a magnitude spectrum. */
        fun findBpm(mag: DoubleArray, n: Int, sampleHz: Double = 30.0): Pair<Int, Boolean> {
            val binHz = sampleHz / n
            val half = n / 2
            val minBin = (0.7 / binHz).toInt().coerceAtLeast(1)
            val maxBin = (2.5 / binHz).toInt().coerceAtMost(half - 1)
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
            return bpm to isLive
        }

        private fun nextPow2(v: Int): Int {
            var n = 1
            while (n < v) n = n shl 1
            return n
        }
    }
}
