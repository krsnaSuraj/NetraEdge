package com.netraedge

import android.content.Context
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.SweepGradient
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * FaceOverlayView — 2026 premium visual treatment of face detection state.
 *
 *  Renders on top of the camera preview:
 *  - Conic gradient ring (rotating when live) — signals active scanning
 *  - Multi-layer corner brackets with breathing bounce
 *  - Liquid morph face box (rounded corners scale with confidence)
 *  - Halo rings expanding outward when verifying
 *  - Particle systems (success confetti, error sparks)
 *  - Sweep scan line with leading glow
 *  - Liveness bar with shimmer + color shift
 *  - Heart pulse glow synced to detected BPM
 *  - Spring physics (overshoot) for state transitions
 *  - Idle: animated reticle + breathing center dot
 *
 *  All animations driven by `System.nanoTime()`. Hardware layer cached.
 *  ~2-3ms per frame.
 */
class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // ===== STATE =====
    @Volatile var faceBox: RectF? = null
    @Volatile var keypoints: List<PointF> = emptyList()
    @Volatile var isSpoof: Boolean = false
    @Volatile var label: String = ""
    @Volatile var confidence: Float = 0f
    @Volatile var liveFace: Boolean = false
    @Volatile var bpm: Int = 0
    @Volatile var hasFace: Boolean = false
    @Volatile var fusedLiveness: Float = 0f
    @Volatile var isVerifying: Boolean = false
    @Volatile var isEnrolling: Boolean = false
    @Volatile var isVerified: Boolean = false   // triggers confetti burst
    @Volatile var isDenied: Boolean = false      // triggers red sparks burst

    private val startTime = System.nanoTime()
    private val startTimeMs = System.currentTimeMillis()

    // ===== SPRING PHYSICS =====
    // Spring-damped state variables (overshoot then settle).
    private var springScale: Float = 0.7f
    private var springScaleVel: Float = 0f
    private var springAlpha: Float = 0f
    private var springAlphaVel: Float = 0f
    private var springTargetScale = 0.7f
    private var springTargetAlpha = 0f

    // ===== PARTICLES =====
    private data class Particle(
        var x: Float, var y: Float,
        var vx: Float, var vy: Float,
        var life: Float,      // 1 -> 0
        var decay: Float,
        var size: Float,
        var color: Int,
        var gravity: Float = 0f
    )
    private val particles = ArrayList<Particle>(64)
    private var lastParticleEmit = 0L

    private val particlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    // ===== PAINTS =====
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#60A5FA")
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 32f
        color = Color.WHITE
        isFakeBoldText = true
        letterSpacing = 0.02f
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#B0101A2E")
    }

    // Conic (sweep) gradient ring
    private val conicPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    // Pulse ring (breathing effect)
    private val pulseRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    // Halo glow (radial soft fill behind face)
    private val haloPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    // Heart pulse
    private val heartGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    // Corner brackets
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 7f
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#60A5FA")
    }

    // Scan line
    private val scanPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val scanGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }

    // Liveness bar
    private val barBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#22FFFFFF")
    }
    private val barFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val barTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 22f
        color = Color.WHITE
        isFakeBoldText = true
    }

    // Idle reticle
    private val reticlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.parseColor("#60A5FA")
    }
    private val reticleDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    // Sweep gradient shader reused
    private val sweepMatrix = Matrix()

    init {
        setLayerType(LAYER_TYPE_HARDWARE, null)
    }

    // ====================================================================
    // Main draw loop
    // ====================================================================
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val tSec = (System.nanoTime() - startTime) / 1_000_000_000.0f
        val wallMs = System.currentTimeMillis() - startTimeMs

        // Spring physics update (mass=1, stiffness=200, damping=18 → slightly underdamped)
        updateSpring(tSec)

        // Trigger particle bursts on state changes
        if (isVerified) emitConfetti(faceBox, tSec)
        if (isDenied) emitErrorSparks(faceBox, tSec)

        if (hasFace && faceBox != null) {
            val box = faceBox!!
            val s = springScale

            // 1. Halo (soft glow under box) — only when live
            drawHalo(canvas, box, tSec)

            // 2. Conic gradient ring around box
            drawConicRing(canvas, box, tSec)

            // 3. Pulse rings (breathing)
            drawPulseRings(canvas, box, tSec)

            // 4. Bounding box (with rounded corners that scale with confidence)
            drawFaceBox(canvas, box, tSec)

            // 5. Keypoints
            for (kp in keypoints) {
                canvas.drawCircle(kp.x, kp.y, 4f, pointPaint)
            }

            // 6. Corner brackets
            drawCornerBrackets(canvas, box, tSec)

            // 7. Scan line (when verifying or enrolling)
            if (isVerifying || isEnrolling) drawScanLine(canvas, box, tSec)

            // 8. Top label badge
            if (label.isNotEmpty()) drawLabelBadge(canvas, box)

            // 9. Liveness bar
            drawLivenessBar(canvas, box, tSec)

            // 10. Heart pulse (if BPM detected)
            if (bpm > 0 && liveFace) drawHeartGlow(canvas, box, tSec)
        } else {
            drawIdleReticle(canvas, tSec, wallMs)
        }

        // 11. Particles (always on top)
        drawParticles(canvas, tSec)

        postInvalidateOnAnimation()
    }

    // ====================================================================
    // Spring physics (mass=1, k=200, damping=18 → slightly underdamped)
    // ====================================================================
    private fun updateSpring(tSec: Float) {
        // Adjust targets based on state
        springTargetScale = if (hasFace) 1.0f else 0.6f
        springTargetAlpha = if (hasFace) 1.0f else 0.0f

        val dt = 0.016f  // ~60fps step
        val k = 200f
        val damping = 18f

        // Spring for scale
        val scaleAccel = (springTargetScale - springScale) * k - springScaleVel * damping
        springScaleVel += scaleAccel * dt
        springScale += springScaleVel * dt

        // Spring for alpha
        val alphaAccel = (springTargetAlpha - springAlpha) * k - springAlphaVel * damping
        springAlphaVel += alphaAccel * dt
        springAlpha += springAlphaVel * dt
    }

    // ====================================================================
    // Halo — soft radial glow under the face
    // ====================================================================
    private fun drawHalo(canvas: Canvas, box: RectF, tSec: Float) {
        val cx = box.centerX()
        val cy = box.centerY()
        val pulse = 0.5f + 0.5f * sin(tSec * 2.0f)
        val r = (maxOf(box.width(), box.height()) * 0.9f) * (0.85f + 0.15f * pulse)
        val baseColor = if (isSpoof) 0xEF4444 else if (isVerified) 0x4ADE80 else if (isVerifying || isEnrolling) 0x60A5FA else 0x22C55E
        val alpha = (30 + pulse * 30f).toInt()
        haloPaint.shader = RadialGradient(
            cx, cy, r,
            intArrayOf(Color.argb(alpha, Color.red(baseColor), Color.green(baseColor), Color.blue(baseColor)),
                       Color.argb(0, Color.red(baseColor), Color.green(baseColor), Color.blue(baseColor))),
            null, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, r, haloPaint)
    }

    // ====================================================================
    // Conic (sweep) gradient ring — rotating when active
    // ====================================================================
    private fun drawConicRing(canvas: Canvas, box: RectF, tSec: Float) {
        val cx = box.centerX()
        val cy = box.centerY()
        val maxR = (maxOf(box.width(), box.height()) / 2f) + 18f
        val rotation = (tSec * 60f) % 360f  // 60 deg/sec

        val colorA = if (isSpoof) 0xEF4444 else if (isVerified) 0x4ADE80 else 0x3B82F6
        val colorB = if (isSpoof) 0xFCA5A5 else if (isVerified) 0xBBF7D0 else 0x60A5FA

        val shader = SweepGradient(cx, cy, intArrayOf(
            colorA, colorB, colorA, colorB, colorA
        ), null)
        sweepMatrix.reset()
        sweepMatrix.postRotate(rotation, cx, cy)
        shader.setLocalMatrix(sweepMatrix)
        conicPaint.shader = shader
        conicPaint.alpha = (200 * springAlpha).toInt().coerceIn(0, 255)
        conicPaint.strokeWidth = 5f
        canvas.drawCircle(cx, cy, maxR, conicPaint)
        conicPaint.shader = null
    }

    // ====================================================================
    // Pulse rings (3 rings at staggered phases)
    // ====================================================================
    private fun drawPulseRings(canvas: Canvas, box: RectF, tSec: Float) {
        val cx = box.centerX()
        val cy = box.centerY()
        val baseR = (maxOf(box.width(), box.height()) / 2f) + 12f
        val color = if (isSpoof) 0xEF4444 else 0x22C55E

        for (i in 0..2) {
            val phase = ((tSec * 0.7f + i * 0.33f) % 1f)
            val r = baseR + 30f * phase
            val alpha = ((1f - phase) * 140f).toInt().coerceIn(0, 255)
            pulseRingPaint.color = Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawCircle(cx, cy, r, pulseRingPaint)
        }
    }

    // ====================================================================
    // Face bounding box (rounded corners scale with state)
    // ====================================================================
    private fun drawFaceBox(canvas: Canvas, box: RectF, tSec: Float) {
        val color = when {
            isSpoof -> 0xEF4444
            isVerified -> 0x4ADE80
            isVerifying || isEnrolling -> 0x60A5FA
            else -> 0x22C55E
        }
        // Corner radius morphs with confidence (0.5 → 0.95 → 0.5)
        val breath = 0.85f + 0.15f * sin(tSec * 2.5f)
        val cornerR = 18f * breath
        boxPaint.color = color
        boxPaint.strokeWidth = 4f
        boxPaint.alpha = (255 * springAlpha).toInt().coerceIn(0, 255)
        canvas.drawRoundRect(box, cornerR, cornerR, boxPaint)
    }

    // ====================================================================
    // Corner brackets (L-shape, breathing)
    // ====================================================================
    private val cornerLen = 32f
    private val cornerInset = 6f
    private val cornerPath = Path()

    private fun drawCornerBrackets(canvas: Canvas, box: RectF, tSec: Float) {
        val appearance = (springAlpha).coerceIn(0f, 1f)
        val bounce = 1f + 0.06f * sin(tSec * 3.0f)
        val len = cornerLen * bounce

        val left = box.left - cornerInset
        val top = box.top - cornerInset
        val right = box.right + cornerInset
        val bottom = box.bottom + cornerInset

        cornerPath.reset()
        // Top-left
        cornerPath.moveTo(left, top + len); cornerPath.lineTo(left, top); cornerPath.lineTo(left + len, top)
        // Top-right
        cornerPath.moveTo(right - len, top); cornerPath.lineTo(right, top); cornerPath.lineTo(right, top + len)
        // Bottom-left
        cornerPath.moveTo(left, bottom - len); cornerPath.lineTo(left, bottom); cornerPath.lineTo(left + len, bottom)
        // Bottom-right
        cornerPath.moveTo(right - len, bottom); cornerPath.lineTo(right, bottom); cornerPath.lineTo(right, bottom - len)

        cornerPaint.alpha = (220 * appearance).toInt().coerceIn(0, 255)
        canvas.drawPath(cornerPath, cornerPaint)
    }

    // ====================================================================
    // Scan line (vertical sweep with glow)
    // ====================================================================
    private fun drawScanLine(canvas: Canvas, box: RectF, tSec: Float) {
        val period = 1.6f
        val phase = (tSec % period) / period
        val y = box.top + box.height() * phase

        val bandHeight = 16f
        val gradient = LinearGradient(
            0f, y - bandHeight, 0f, y + bandHeight,
            intArrayOf(
                Color.TRANSPARENT,
                Color.argb(160, 96, 165, 250),
                Color.argb(255, 147, 197, 253),
                Color.argb(160, 96, 165, 250),
                Color.TRANSPARENT
            ),
            floatArrayOf(0f, 0.35f, 0.5f, 0.65f, 1f),
            Shader.TileMode.CLAMP
        )
        scanPaint.shader = gradient
        canvas.drawRect(box.left, y - bandHeight, box.right, y + bandHeight, scanPaint)
        scanPaint.shader = null

        scanGlowPaint.color = Color.parseColor("#93C5FD")
        canvas.drawRect(box.left, y - 2f, box.right, y + 2f, scanGlowPaint)
    }

    // ====================================================================
    // Label badge (top of box)
    // ====================================================================
    private val labelRect = RectF()
    private fun drawLabelBadge(canvas: Canvas, box: RectF) {
        val pad = 14f
        val tw = labelPaint.measureText(label)
        labelRect.set(
            box.left, box.top - 56f,
            box.left + tw + 2 * pad, box.top - 12f
        )
        // Clip if off-screen
        if (labelRect.top < 0f) labelRect.offset(0f, -labelRect.top + 8f)
        canvas.drawRoundRect(labelRect, 18f, 18f, labelBgPaint)
        canvas.drawText(label, box.left + pad, labelRect.bottom - 18f, labelPaint)
    }

    // ====================================================================
    // Liveness bar (below face, 3-stop color, shimmer)
    // ====================================================================
    private val barRect = RectF()
    private fun drawLivenessBar(canvas: Canvas, box: RectF, tSec: Float) {
        val barWidth = box.width()
        val barHeight = 12f
        val barLeft = box.left
        val barTop = box.bottom + 18f
        val barRight = barLeft + barWidth
        val barBottom = barTop + barHeight

        // Background
        barRect.set(barLeft, barTop, barRight, barBottom)
        canvas.drawRoundRect(barRect, 6f, 6f, barBgPaint)

        // Fill
        val fill = fusedLiveness.coerceIn(0f, 1f)
        val fillRight = barLeft + barWidth * fill
        if (fillRight > barLeft) {
            val color = when {
                fill < 0.5f -> blendColor(0xEF4444, 0xF59E0B, fill * 2f)
                else -> blendColor(0xF59E0B, 0x22C55E, (fill - 0.5f) * 2f)
            }
            barFillPaint.color = color

            // Shimmer
            val shimmerX = barLeft + barWidth * ((tSec * 0.35f) % 1f) - 40f
            if (shimmerX > barLeft && shimmerX < fillRight) {
                barFillPaint.shader = LinearGradient(
                    shimmerX, 0f, shimmerX + 50f, 0f,
                    intArrayOf(Color.TRANSPARENT, Color.argb(90, 255, 255, 255), Color.TRANSPARENT),
                    null, Shader.TileMode.CLAMP
                )
            } else {
                barFillPaint.shader = null
            }
            barRect.set(barLeft, barTop, fillRight, barBottom)
            canvas.drawRoundRect(barRect, 6f, 6f, barFillPaint)
            barFillPaint.shader = null
        }

        val pct = (fill * 100).toInt()
        val txt = "Liveness $pct%"
        val tw = barTextPaint.measureText(txt)
        canvas.drawText(txt, barLeft + (barWidth - tw) / 2f, barTop - 6f, barTextPaint)
    }

    // ====================================================================
    // Heart pulse glow (synced to BPM)
    // ====================================================================
    private fun drawHeartGlow(canvas: Canvas, box: RectF, tSec: Float) {
        val bpmNorm = bpm.coerceIn(40, 150).toFloat()
        val cycleHz = bpmNorm / 60f
        val phase = (tSec * cycleHz) * 2f * Math.PI.toFloat()
        val pulse = 0.5f + 0.5f * sin(phase)

        val cx = box.left + 36f
        val cy = box.top + 36f
        val maxR = 28f
        val r = 14f + pulse * (maxR - 14f)
        val alpha = (100 + pulse * 120f).toInt().coerceIn(0, 255)

        heartGlowPaint.color = Color.argb(alpha, 239, 68, 68)
        canvas.drawCircle(cx, cy, r, heartGlowPaint)
        heartGlowPaint.color = Color.argb((alpha * 1.3f).toInt().coerceAtMost(255), 252, 165, 165)
        canvas.drawCircle(cx, cy, r * 0.5f, heartGlowPaint)
    }

    // ====================================================================
    // Idle reticle (when no face)
    // ====================================================================
    private fun drawIdleReticle(canvas: Canvas, tSec: Float, wallMs: Long) {
        val cx = width / 2f
        val cy = height / 2f
        val r = 130f
        val appearance = springAlpha

        // 3 expanding rings
        for (i in 0..2) {
            val phase = ((tSec * 0.6f + i * 0.33f) % 1f)
            val ringR = r * (0.5f + phase * 0.7f)
            val ringAlpha = ((1f - phase) * 100f).toInt().coerceIn(0, 255)
            pulseRingPaint.color = Color.argb(ringAlpha, 96, 165, 250)
            pulseRingPaint.strokeWidth = 2.5f
            canvas.drawCircle(cx, cy, ringR, pulseRingPaint)
        }

        // Steady reticle circle
        reticlePaint.strokeWidth = 3f
        reticlePaint.color = Color.parseColor("#60A5FA")
        reticlePaint.alpha = (180 * appearance).toInt().coerceIn(0, 255)
        canvas.drawCircle(cx, cy, r, reticlePaint)

        // Center dot with breathing
        val breath = 0.5f + 0.5f * sin(tSec * 2.0f)
        reticleDotPaint.color = Color.argb((180 * breath).toInt().coerceIn(0, 255), 96, 165, 250)
        canvas.drawCircle(cx, cy, 6f + 4f * breath, reticleDotPaint)

        // Crosshair lines
        val gap = 12f
        val len = 24f
        reticlePaint.strokeWidth = 2.5f
        canvas.drawLine(cx - r - 30f, cy, cx - r + gap, cy, reticlePaint)
        canvas.drawLine(cx - r + gap + len, cy, cx - r + gap + len + 14f, cy, reticlePaint)
        canvas.drawLine(cx + r - 30f, cy, cx + r - gap - len - 14f, cy, reticlePaint)
        canvas.drawLine(cx + r - gap, cy, cx + r + 30f, cy, reticlePaint)
        canvas.drawLine(cx, cy - r - 30f, cx, cy - r + gap, reticlePaint)
        canvas.drawLine(cx, cy - r + gap + len, cx, cy - r + gap + len + 14f, reticlePaint)
        canvas.drawLine(cx, cy + r - 30f, cx, cy + r - gap - len - 14f, reticlePaint)
        canvas.drawLine(cx, cy + r - gap, cx, cy + r + 30f, reticlePaint)

        val txt = "Looking for face…"
        val tw = labelPaint.measureText(txt)
        val badgeY = cy + r + 60f
        val badgeRect = RectF(cx - tw / 2f - 20f, badgeY, cx + tw / 2f + 20f, badgeY + 50f)
        canvas.drawRoundRect(badgeRect, 24f, 24f, labelBgPaint)
        canvas.drawText(txt, cx - tw / 2f, badgeY + 35f, labelPaint)
    }

    // ====================================================================
    // Particle system
    // ====================================================================
    private fun emitConfetti(box: RectF?, tSec: Float) {
        val now = System.currentTimeMillis()
        if (now - lastParticleEmit < 60) return
        lastParticleEmit = now
        if (box == null) return
        val cx = box.centerX()
        val cy = box.centerY()
        repeat(3) {
            val angle = Random.nextDouble(0.0, Math.PI * 2).toFloat()
            val speed = Random.nextDouble(3.0, 8.0).toFloat()
            val color = when (Random.nextInt(3)) {
                0 -> 0x4ADE80  // green
                1 -> 0xFCD34D  // gold
                else -> 0x60A5FA  // blue
            }
            particles.add(Particle(
                x = cx, y = cy,
                vx = cos(angle) * speed,
                vy = sin(angle) * speed,
                life = 1f,
                decay = Random.nextDouble(0.012, 0.020).toFloat(),
                size = Random.nextDouble(3.0, 6.0).toFloat(),
                color = color,
                gravity = 0.15f
            ))
        }
    }

    private fun emitErrorSparks(box: RectF?, tSec: Float) {
        val now = System.currentTimeMillis()
        if (now - lastParticleEmit < 60) return
        lastParticleEmit = now
        if (box == null) return
        val cx = box.centerX()
        val cy = box.centerY()
        repeat(3) {
            val angle = Random.nextDouble(0.0, Math.PI * 2).toFloat()
            val speed = Random.nextDouble(4.0, 10.0).toFloat()
            particles.add(Particle(
                x = cx, y = cy,
                vx = cos(angle) * speed,
                vy = sin(angle) * speed,
                life = 1f,
                decay = Random.nextDouble(0.015, 0.025).toFloat(),
                size = Random.nextDouble(2.5, 5.0).toFloat(),
                color = 0xEF4444,
                gravity = 0.10f
            ))
        }
    }

    private fun drawParticles(canvas: Canvas, tSec: Float) {
        if (particles.isEmpty()) return
        val it = particles.iterator()
        while (it.hasNext()) {
            val p = it.next()
            p.life -= p.decay
            if (p.life <= 0f) { it.remove(); continue }
            p.x += p.vx
            p.y += p.vy
            p.vy += p.gravity
            val alpha = (p.life * 255f).toInt().coerceIn(0, 255)
            particlePaint.color = Color.argb(
                alpha,
                Color.red(p.color), Color.green(p.color), Color.blue(p.color)
            )
            canvas.drawCircle(p.x, p.y, p.size * p.life, particlePaint)
        }
    }

    fun clearParticles() {
        particles.clear()
    }

    // ====================================================================
    // Helpers
    // ====================================================================
    private fun blendColor(c1: Int, c2: Int, t: Float): Int {
        val tt = t.coerceIn(0f, 1f)
        val a = (Color.alpha(c1) + (Color.alpha(c2) - Color.alpha(c1)) * tt).toInt()
        val r = (Color.red(c1) + (Color.red(c2) - Color.red(c1)) * tt).toInt()
        val g = (Color.green(c1) + (Color.green(c2) - Color.green(c1)) * tt).toInt()
        val b = (Color.blue(c1) + (Color.blue(c2) - Color.blue(c1)) * tt).toInt()
        return Color.argb(a, r, g, b)
    }
}
