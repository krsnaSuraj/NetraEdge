package com.netraedge

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log

/**
 * HapticHelper — premium 2026 haptic feedback for state transitions.
 *
 *  - tap(): short 10ms tap (button press, state change)
 *  - success(): 3-tap pattern (verification success)
 *  - error(): 2 long buzzes (spoof detected, denied)
 *  - challenge(): mid pulse (active challenge step)
 *
 *  Uses Android's VibrationEffect for API 26+ (rich haptics),
 *  falls back to legacy Vibrator API for older devices.
 *  All vibrate calls are wrapped in try/catch — haptics must NEVER crash the app.
 */
class HapticHelper(context: Context) {
    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vm = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
        vm?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    private fun safeVibrate(effect: VibrationEffect) {
        try {
            vibrator?.vibrate(effect)
        } catch (e: SecurityException) {
            Log.w("HapticHelper", "Vibration permission denied: ${e.message}")
        } catch (e: Exception) {
            Log.w("HapticHelper", "Vibration failed: ${e.message}")
        }
    }

    @Suppress("DEPRECATION")
    private fun safeVibrateLegacy(ms: Long) {
        try {
            vibrator?.vibrate(ms)
        } catch (e: SecurityException) {
            Log.w("HapticHelper", "Vibration permission denied: ${e.message}")
        } catch (e: Exception) {
            Log.w("HapticHelper", "Vibration failed: ${e.message}")
        }
    }

    @Suppress("DEPRECATION")
    private fun safeVibrateLegacy(pattern: LongArray, repeat: Int) {
        try {
            vibrator?.vibrate(pattern, repeat)
        } catch (e: SecurityException) {
            Log.w("HapticHelper", "Vibration permission denied: ${e.message}")
        } catch (e: Exception) {
            Log.w("HapticHelper", "Vibration failed: ${e.message}")
        }
    }

    fun tap() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            safeVibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_TICK))
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            safeVibrate(VibrationEffect.createOneShot(10, 60))
        } else {
            safeVibrateLegacy(10)
        }
    }

    fun challenge() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            safeVibrate(VibrationEffect.createOneShot(40, 120))
        } else {
            safeVibrateLegacy(40)
        }
    }

    fun success() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val timings = longArrayOf(0, 25, 60, 25, 60, 80)
            val amplitudes = intArrayOf(0, 180, 0, 180, 0, 255)
            safeVibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
        } else {
            safeVibrateLegacy(longArrayOf(0, 25, 60, 25, 60, 80), -1)
        }
    }

    fun error() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val timings = longArrayOf(0, 120, 80, 120)
            val amplitudes = intArrayOf(0, 255, 0, 255)
            safeVibrate(VibrationEffect.createWaveform(timings, amplitudes, -1))
        } else {
            safeVibrateLegacy(longArrayOf(0, 120, 80, 120), -1)
        }
    }
}
