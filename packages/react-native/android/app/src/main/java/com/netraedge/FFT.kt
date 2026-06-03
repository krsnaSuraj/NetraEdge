package com.netraedge

import kotlin.math.cos
import kotlin.math.sin

/**
 * Iterative in-place radix-2 Cooley-Tukey FFT over DoubleArray pairs.
 * Lengths must be a power of two. No allocations after construction
 * (the caller reuses pre-sized buffers).
 */
object FFT {

    /** Compute the magnitude spectrum of the real-valued input signal. */
    fun magnitude(input: DoubleArray, out: DoubleArray) {
        val n = input.size
        require(n == out.size) { "size mismatch" }
        require(n > 0 && (n and (n - 1)) == 0) { "size must be power of two" }

        val re = DoubleArray(n)
        val im = DoubleArray(n)
        System.arraycopy(input, 0, re, 0, n)

        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * Math.PI / len
            val wRe = cos(angle)
            val wIm = sin(angle)
            var i = 0
            while (i < n) {
                var curRe = 1.0
                var curIm = 0.0
                for (k in 0 until halfLen) {
                    val a = i + k
                    val b = i + k + halfLen
                    val tRe = re[b] * curRe - im[b] * curIm
                    val tIm = re[b] * curIm + im[b] * curRe
                    re[b] = re[a] - tRe
                    im[b] = im[a] - tIm
                    re[a] = re[a] + tRe
                    im[a] = im[a] + tIm
                    val nRe = curRe * wRe - curIm * wIm
                    val nIm = curRe * wIm + curIm * wRe
                    curRe = nRe
                    curIm = nIm
                }
                i += len
            }
            len = len shl 1
        }

        for (k in 0 until n) {
            out[k] = sqrt(re[k] * re[k] + im[k] * im[k])
        }
    }
}

private fun sqrt(x: Double): Double = kotlin.math.sqrt(x)
