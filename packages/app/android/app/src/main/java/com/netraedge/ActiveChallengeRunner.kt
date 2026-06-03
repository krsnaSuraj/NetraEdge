package com.netraedge

import android.util.Log
import kotlin.random.Random

/**
 * State machine for randomized active liveness challenges
 * (blink, smile, head turn). Reads MediaPipe blendshape scores
 * to detect the requested gesture and advances to the next step.
 *
 * Challenge order is randomized each time to prevent
 * automated replay attacks and pre-computed bypasses.
 *
 *   IDLE -> [random step 1] -> [random step 2] -> [random step 3] -> PASSED
 */
class ActiveChallengeRunner {

    enum class Step { IDLE, BLINK, SMILE, HEAD_TURN_LEFT, HEAD_TURN_RIGHT, PASSED, FAILED }
    enum class GestureResult { PENDING, OK, MISSED }

    var currentStep: Step = Step.IDLE
        private set

    private var blinkCount = 0
    private var lastBlinkAt = 0L
    private var startedAt = 0L
    private val challengeTimeoutMs = 25_000L  // 25s — generous for demo, less likely to time out

    // Detection thresholds tuned for MediaPipe Face Landmarker blendshapes (V2 model).
    // Range: 0.0 (no expression) to 1.0 (full expression).
    // Per MediaPipe documentation:
    //   Smile blendshapes (mouthSmileLeft/Right) typically peak at 0.2-0.5 for a natural smile
    //   (the V2 model is conservative — exaggerated smiles reach 0.7+).
    //   Blink (eyeBlinkLeft/Right) peaks at 0.6-0.95 for a full blink.
    //   jawOpen is reliable: 0.3+ means clearly talking/smiling with mouth open.
    // Thresholds are intentionally LOW so a natural smile/blink registers.
    private val blinkThreshold = 0.15f  // lowered from 0.20 for natural blinks
    private val smileThreshold = 0.10f  // lowered from 0.15 for natural smiles
    private val jawOpenThreshold = 0.25f  // used as backup for "open mouth smile"

    // EAR (Eye Aspect Ratio) for fallback blink detection from landmarks
    // EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    // Open eye: ~0.25-0.35, Closed eye: ~0.05-0.15
    // Threshold 0.22 is a balance — catches natural blinks without
    // false-positives on partial squints.
    private val earThreshold = 0.22f

    // Head turn is detected by landmark geometry (left/right eye position change
    // and nose offset from face center) — NOT blendshapes, since MediaPipe does
    // not provide a "headYaw" blendshape. Detection happens in MainActivity.

    private var pendingGesture = GestureResult.PENDING
    private var lastBlinkLeft = 0f
    private var lastBlinkRight = 0f
    private var lastEar = 0.3f
    private var challengeOrder = listOf(Step.BLINK, Step.SMILE)
    private var stepIndex = 0

    fun start() {
        // 4 challenges available, randomly pick 2 per session. Includes
        // head turn for variety but uses LOWER threshold (0.10 instead of
        // 0.15) so a small head turn registers — user doesn't need to
        // rotate so far that the face exits the frame.
        val all = listOf(Step.BLINK, Step.SMILE, Step.HEAD_TURN_LEFT, Step.HEAD_TURN_RIGHT)
        challengeOrder = all.shuffled().take(2)
        stepIndex = 0
        currentStep = challengeOrder[0]
        blinkCount = 0
        startedAt = System.currentTimeMillis()
        pendingGesture = GestureResult.PENDING
        Log.i(TAG, "Active challenge started — order: ${challengeOrder.map { it.name }}")
    }

    /** Seconds remaining in this challenge session (0 if expired). */
    fun remainingSeconds(): Int {
        if (currentStep == Step.IDLE || currentStep == Step.PASSED || currentStep == Step.FAILED) return 0
        val elapsed = System.currentTimeMillis() - startedAt
        val remaining = ((challengeTimeoutMs - elapsed) / 1000L).toInt()
        return remaining.coerceAtLeast(0)
    }

    fun cancel() {
        currentStep = Step.IDLE
    }

    fun onBlendshapes(
        blinkLeft: Float,
        blinkRight: Float,
        smile: Float,
        headYaw: Float,
        nowMs: Long,
        jawOpen: Float = 0f
    ) {
        if (currentStep == Step.IDLE || currentStep == Step.PASSED || currentStep == Step.FAILED) return

        if (System.currentTimeMillis() - startedAt > challengeTimeoutMs) {
            currentStep = Step.FAILED
            Log.w(TAG, "Active challenge timed out")
            return
        }

        when (currentStep) {
            Step.BLINK -> {
                val lClosed = blinkLeft > blinkThreshold
                val rClosed = blinkRight > blinkThreshold
                val justBlinked = (lClosed || rClosed) &&
                    (lastBlinkLeft < blinkThreshold && lastBlinkRight < blinkThreshold)
                if (justBlinked && (nowMs - lastBlinkAt) > 200) {
                    blinkCount += 1
                    lastBlinkAt = nowMs
                    Log.i(TAG, "Blink detected #$blinkCount (L=$blinkLeft R=$blinkRight)")
                }
                lastBlinkLeft = blinkLeft
                lastBlinkRight = blinkRight
                if (blinkCount >= 1) {  // 1 blink is enough (was 2 - too slow)
                    advanceStep()
                }
            }
            Step.SMILE -> {
                // Detect smile via: mouthSmileLeft/Right OR jawOpen (for big open smiles)
                val smileOk = smile > smileThreshold
                val jawOk = jawOpen > jawOpenThreshold
                if (smileOk) {
                    Log.i(TAG, "Smile detected (smile=$smile jawOpen=$jawOpen), advancing")
                    advanceStep()
                } else if (jawOk) {
                    Log.i(TAG, "Open-mouth smile detected (jawOpen=$jawOpen smile=$smile), advancing")
                    advanceStep()
                }
            }
            else -> Unit
        }
    }

    /**
     * Fallback blink detection using Eye Aspect Ratio (EAR) from landmarks.
     * More reliable than blendshapes when MediaPipe doesn't output them.
     *
     * @param ear Eye aspect ratio: vertical_eye_height / horizontal_eye_width
     *            Open ~0.30, Closed ~0.10. Pass 0 to skip.
     */
    fun onEar(ear: Float) {
        if (currentStep != Step.BLINK) return
        if (ear <= 0f) return  // not provided
        if (System.currentTimeMillis() - startedAt > challengeTimeoutMs) return

        val justClosed = ear < earThreshold && lastEar >= earThreshold
        val justOpened = ear >= earThreshold && lastEar < earThreshold
        lastEar = ear

        if (justClosed && (System.currentTimeMillis() - lastBlinkAt) > 200) {
            blinkCount += 1
            lastBlinkAt = System.currentTimeMillis()
            Log.i(TAG, "EAR blink detected #$blinkCount (ear=$ear)")
        }
        if (blinkCount >= 1) {
            advanceStep()
        }
    }

    /**
     * Landmark-based head turn detection. MediaPipe does NOT provide a headYaw
     * blendshape — only facial expression blendshapes. To detect head turns, we
     * measure the horizontal position of the nose relative to the midpoint of
     * the two eyes.
     *
     * When user turns head to their LEFT:
     *   - Nose moves to the right of the eye midpoint in image space (parallax)
     *   - noseOffsetX = (noseX - eyeMidX) / eyeDist becomes POSITIVE (> 0.15)
     *
     * When user turns head to their RIGHT:
     *   - Nose moves to the left of the eye midpoint
     *   - noseOffsetX becomes NEGATIVE (< -0.15)
     *
     * Straight ahead: noseOffsetX ≈ 0
     */
    fun onLandmarksForHeadTurn(noseOffsetX: Float) {
        if (currentStep == Step.IDLE || currentStep == Step.PASSED || currentStep == Step.FAILED) return

        when (currentStep) {
            Step.HEAD_TURN_LEFT -> {
                // Lowered from 0.15 to 0.10 — easier to register, no need
                // to over-rotate (which would push face out of frame).
                if (noseOffsetX > 0.10f) {
                    Log.i(TAG, "Head turn LEFT detected (noseOffsetX=$noseOffsetX)")
                    advanceStep()
                }
            }
            Step.HEAD_TURN_RIGHT -> {
                if (noseOffsetX < -0.10f) {
                    Log.i(TAG, "Head turn RIGHT detected (noseOffsetX=$noseOffsetX)")
                    advanceStep()
                }
            }
            else -> Unit
        }
    }

    private fun advanceStep() {
        stepIndex++
        if (stepIndex >= challengeOrder.size) {
            currentStep = Step.PASSED
            Log.i(TAG, "All challenges PASSED")
        } else {
            currentStep = challengeOrder[stepIndex]
            Log.i(TAG, "Next challenge: ${currentStep.name} (${stepIndex + 1}/${challengeOrder.size})")
        }
    }

    /** Current step index (0-based) for UI progress display. */
    fun getCurrentStepIndex(): Int = stepIndex

    /** Total number of challenges in this session. */
    fun getTotalSteps(): Int = challengeOrder.size

    companion object {
        private const val TAG = "ActiveChallenge"
    }
}
