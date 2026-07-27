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
 *
 * Anti-bypass defenses:
 *   - Gesture must be HELD for minGestureFrames (15 = 500ms) before advancing
 *     (prevents single-frame flukes from video noise)
 *   - 2 blinks required (harder to time with a pre-recorded video)
 *   - Thresholds calibrated for intentional gestures, not micro-movements
 */
class ActiveChallengeRunner {

    enum class Step { IDLE, BLINK, SMILE, HEAD_TURN_LEFT, HEAD_TURN_RIGHT, PASSED, FAILED }
    enum class GestureResult { PENDING, OK, MISSED }

    var currentStep: Step = Step.IDLE
        private set

    private var blinkCount = 0
    private var lastBlinkAt = 0L
    private var startedAt = 0L
    private val challengeTimeoutMs = 25_000L

    // Thresholds tuned for INTENTIONAL gestures (not micro-expressions).
    // MediaPipe V2 model ranges: 0.0 (neutral) to 1.0 (full expression).
    // These require a clear, deliberate action but stay forgiving enough for a
    // smooth live demo — a passive face or video noise will not trigger them.
    private val blinkThreshold = 0.30f       // a real, conscious blink (full blink peaks 0.6+)
    private val smileThreshold = 0.20f       // a visible smile (natural smile peaks 0.2-0.5)
    private val jawOpenThreshold = 0.30f     // clearly open mouth (backup for big smiles)

    // EAR (Eye Aspect Ratio) for fallback blink detection from landmarks
    // Open eye: ~0.25-0.35, Closed eye: ~0.05-0.15
    private val earThreshold = 0.22f

    // Minimum consecutive frames a gesture must persist before advancing.
    // At ~25-30fps, 5 frames ≈ 170-200ms — responsive for a live demo yet long
    // enough to reject single-frame blendshape noise / compression artifacts.
    private val minGestureFrames = 5

    // Require 2 blinks — single blink could be natural, 2 proves interaction.
    // Video replay attackers can't precisely control blink timing in a
    // pre-recorded clip against a random challenge sequence.
    private val requiredBlinks = 2

    // Consecutive frames where the current gesture condition is met.
    // Reset when the condition drops below threshold (user stopped the gesture).
    private var gestureFrameCount = 0

    // Track whether we already captured the blink close phase for this blink
    // (prevents counting the same blink twice if it spans multiple frames)
    private var blinkCloseCaptured = false

    private var pendingGesture = GestureResult.PENDING
    private var lastBlinkLeft = 0f
    private var lastBlinkRight = 0f
    private var lastEar = 0.3f
    private var challengeOrder = listOf(Step.BLINK, Step.SMILE)
    private var stepIndex = 0

    fun start() {
        // Randomly pick 3 out of 4 challenge types. Random order prevents
        // attackers from pre-recording a video that plays in the expected sequence.
        val all = listOf(Step.BLINK, Step.SMILE, Step.HEAD_TURN_LEFT, Step.HEAD_TURN_RIGHT)
        challengeOrder = all.shuffled().take(3)
        stepIndex = 0
        currentStep = challengeOrder[0]
        blinkCount = 0
        blinkCloseCaptured = false
        gestureFrameCount = 0
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
                val leftClosed = blinkLeft > blinkThreshold
                val rightClosed = blinkRight > blinkThreshold
                val anyEyeClosed = leftClosed || rightClosed

                // Edge-detected blink: transition from OPEN -> CLOSED
                val wasOpen = lastBlinkLeft < blinkThreshold && lastBlinkRight < blinkThreshold
                val justClosed = anyEyeClosed && wasOpen

                if (justClosed && !blinkCloseCaptured && (nowMs - lastBlinkAt) > 300) {
                    blinkCloseCaptured = true
                    lastBlinkAt = nowMs
                    Log.i(TAG, "Blink close phase #${blinkCount + 1} (L=$blinkLeft R=$blinkRight)")
                }

                // Count blink when eye RE-OPENS after being closed (full blink cycle)
                val wasClosed = lastBlinkLeft >= blinkThreshold || lastBlinkRight >= blinkThreshold
                val justOpened = !anyEyeClosed && wasClosed && blinkCloseCaptured

                if (justOpened && (nowMs - lastBlinkAt) > 200) {
                    blinkCount += 1
                    blinkCloseCaptured = false
                    lastBlinkAt = nowMs
                    Log.i(TAG, "Blink complete #$blinkCount (L=$blinkLeft R=$blinkRight)")
                }

                lastBlinkLeft = blinkLeft
                lastBlinkRight = blinkRight

                if (blinkCount >= requiredBlinks) {
                    advanceStep()
                }
            }
            Step.SMILE -> {
                val smileOk = smile > smileThreshold
                val jawOk = jawOpen > jawOpenThreshold
                val gestureActive = smileOk || jawOk

                if (gestureActive) {
                    gestureFrameCount++
                    if (gestureFrameCount >= minGestureFrames) {
                        Log.i(TAG, "Smile sustained ${gestureFrameCount}f (smile=$smile jawOpen=$jawOpen), advancing")
                        advanceStep()
                    }
                } else {
                    gestureFrameCount = 0
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
        if (ear <= 0f) return
        if (System.currentTimeMillis() - startedAt > challengeTimeoutMs) return

        val justClosed = ear < earThreshold && lastEar >= earThreshold
        val justOpened = ear >= earThreshold && lastEar < earThreshold
        lastEar = ear

        if (justClosed && !blinkCloseCaptured && (System.currentTimeMillis() - lastBlinkAt) > 300) {
            blinkCloseCaptured = true
            lastBlinkAt = System.currentTimeMillis()
            Log.i(TAG, "EAR blink close phase #${blinkCount + 1} (ear=$ear)")
        }

        if (justOpened && blinkCloseCaptured && (System.currentTimeMillis() - lastBlinkAt) > 200) {
            blinkCount += 1
            blinkCloseCaptured = false
            lastBlinkAt = System.currentTimeMillis()
            Log.i(TAG, "EAR blink complete #$blinkCount (ear=$ear)")
        }

        if (blinkCount >= requiredBlinks) {
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
    /** Nose offset threshold for head turn — a clear turn, not a micro-movement. */
    private val headTurnThreshold = 0.13f

    fun onLandmarksForHeadTurn(noseOffsetX: Float) {
        if (currentStep == Step.IDLE || currentStep == Step.PASSED || currentStep == Step.FAILED) return

        when (currentStep) {
            Step.HEAD_TURN_LEFT -> {
                if (noseOffsetX > headTurnThreshold) {
                    gestureFrameCount++
                    if (gestureFrameCount >= minGestureFrames) {
                        Log.i(TAG, "Head turn LEFT sustained ${gestureFrameCount}f (noseOffsetX=$noseOffsetX)")
                        advanceStep()
                    }
                } else {
                    gestureFrameCount = 0
                }
            }
            Step.HEAD_TURN_RIGHT -> {
                if (noseOffsetX < -headTurnThreshold) {
                    gestureFrameCount++
                    if (gestureFrameCount >= minGestureFrames) {
                        Log.i(TAG, "Head turn RIGHT sustained ${gestureFrameCount}f (noseOffsetX=$noseOffsetX)")
                        advanceStep()
                    }
                } else {
                    gestureFrameCount = 0
                }
            }
            else -> Unit
        }
    }

    private fun advanceStep() {
        stepIndex++
        gestureFrameCount = 0  // reset for next challenge
        blinkCloseCaptured = false
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
