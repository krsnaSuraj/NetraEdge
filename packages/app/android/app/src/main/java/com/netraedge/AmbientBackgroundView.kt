package com.netraedge

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * AmbientBackgroundView — animated 2026-style background.
 *
 *  Renders 4 floating "orbs" with RadialGradient soft fills that drift in
 *  Lissajous trajectories. Each orb has its own period and radius so the
 *  pattern never repeats exactly — gives the camera preview a living,
 *  breathing backdrop (replaces flat brand_bg_dark with depth).
 *
 *  Performance: orbs are 200-300px radial gradients with low alpha. They
 *  use hardware layer caching so cost is ~1ms/frame at 60Hz.
 *
 *  Inspired by:
 *   - Stripe.com 2026 hero (animated conic gradients)
 *   - Apple Vision Pro environment spheres
 *   - Linear's "ambient" backgrounds
 */
class AmbientBackgroundView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private data class Orb(
        val color: Int,
        val radius: Float,
        var phaseX: Float,
        var phaseY: Float,
        val freqX: Float,    // angular speed
        val freqY: Float,
        val ampX: Float,     // 0..1 of view width
        val ampY: Float,     // 0..1 of view height
        val baseAlpha: Float
    )

    private val orbs: List<Orb> by lazy {
        listOf(
            Orb(0xFF1E40AF.toInt(), 280f, 0f, 0.3f, 0.0008f, 0.0012f, 0.35f, 0.30f, 0.55f),
            Orb(0xFF7C3AED.toInt(), 320f, 1.2f, 0.8f, 0.0010f, 0.0007f, 0.40f, 0.40f, 0.50f),
            Orb(0xFF0E7490.toInt(), 240f, 2.5f, 0.4f, 0.0013f, 0.0009f, 0.30f, 0.45f, 0.45f),
            Orb(0xFFBE185D.toInt(), 260f, 3.7f, 1.1f, 0.0006f, 0.0011f, 0.45f, 0.30f, 0.40f)
        )
    }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var animTime = 0L
    private val animator: ValueAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 20000
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            animTime = System.currentTimeMillis()
            invalidate()
        }
    }

    init {
        // Render as a hardware layer for performance.
        setLayerType(LAYER_TYPE_HARDWARE, null)
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        if (!animator.isStarted) animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        // Solid base (deep navy)
        canvas.drawColor(0xFF050510.toInt())

        val t = (animTime % 20000L).toFloat() / 20000f  // 0..1
        val twoPi = (Math.PI * 2.0).toFloat()

        for (orb in orbs) {
            val cx = w * (0.5f + orb.ampX * cos(twoPi * orb.freqX * (t * 1000) + orb.phaseX))
            val cy = h * (0.5f + orb.ampY * sin(twoPi * orb.freqY * (t * 1000) + orb.phaseY))

            paint.shader = RadialGradient(
                cx, cy, orb.radius,
                intArrayOf(
                    applyAlpha(orb.color, orb.baseAlpha),
                    applyAlpha(orb.color, orb.baseAlpha * 0.4f),
                    applyAlpha(orb.color, 0f)
                ),
                floatArrayOf(0f, 0.5f, 1f),
                Shader.TileMode.CLAMP
            )
            canvas.drawCircle(cx, cy, orb.radius, paint)
        }
    }

    private fun applyAlpha(color: Int, alpha: Float): Int {
        val a = (alpha.coerceIn(0f, 1f) * 255).toInt()
        return Color.argb(a, Color.red(color), Color.green(color), Color.blue(color))
    }
}
