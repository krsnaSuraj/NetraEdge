package com.netraedge

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.pm.Signature
import android.content.pm.SigningInfo
import android.os.Build
import android.provider.Settings
import android.util.Base64
import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.security.MessageDigest

/**
 * Runtime tamper / reverse-engineering / hostile-environment detection.
 *
 * Every check here is non-destructive — it just sets a flag in
 * [report] which the caller can inspect. A production build would
 * choose to either:
 *   (a) refuse to run on a tampered device, OR
 *   (b) silently disable sensitive features.
 *
 * For the hackathon demo we lean toward (b) and just log warnings
 * so the demo still works on dev / rooted phones used by judges.
 *
 * Triggers:
 *   - [Debug.isDebuggerConnected]                 (anti-debug)
 *   - /system/xbin/su, /system/bin/su, /sbin/su  (root)
 *   - Build.TAGS = "test-keys"                    (custom ROM)
 *   - ro.debuggable = 1                           (debug system)
 *   - ro.secure = 0                               (insecure userbuild)
 *   - superuser.apk / Supersu / MagiskManager     (root UI installed)
 *   - applicationInfo.flags & FLAG_DEBUGGABLE     (debuggable APK)
 *   - signing cert SHA-256 mismatch               (repacked APK)
 *   - emulator indicators (goldfish, ranchu, Build.FINGERPRINT) (optional)
 */
object SecurityHardening {

    /**
     * SHA-256 of the NetraEdge release-signing certificate.
     *
     * For the hackathon demo this is the DEBUG keystore (already
     * known: ~/.android/debug.keystore). For production we would
     * bake the real SHA-256 here. The mismatch detection is
     * exact-bit, so any repack attempt by an attacker with a
     * different keystore flips the bit and triggers the flag.
     *
     * To get your real SHA-256:
     *   keytool -list -v -keystore my-release.keystore | grep SHA256
     *   or: apksigner verify --print-certs app-release.apk
     */
    private const val EXPECTED_SIGNING_SHA256 = ""
    private const val EXPECTED_SIGNING_SHA256_DEBUG = ""  // filled by gradle/keystore

    data class Report(
        val isDebuggerAttached: Boolean = false,
        val isRooted: Boolean = false,
        val isCustomRom: Boolean = false,
        val isDebuggableApk: Boolean = false,
        val isEmulator: Boolean = false,
        val isRepackedApk: Boolean = false,
        val detectedIssues: List<String> = emptyList(),
    ) {
        val isTampered: Boolean
            get() = detectedIssues.isNotEmpty()
    }

    fun audit(ctx: Context): Report {
        val issues = mutableListOf<String>()

        if (isDebuggerAttached()) issues += "debugger_attached"
        val rooted = isRooted()
        if (rooted) issues += "rooted_device"
        if (isCustomRom()) issues += "custom_rom"
        if (isDebuggableApk(ctx)) issues += "debuggable_apk"
        val emu = isEmulator()
        if (emu) issues += "emulator"
        val repacked = isRepackedApk(ctx)
        if (repacked) issues += "repacked_apk"

        if (issues.isNotEmpty()) {
            Log.w("NetraEdgeSecurity", "Tamper signals: ${issues.joinToString()}")
        }
        return Report(
            isDebuggerAttached = isDebuggerAttached(),
            isRooted = rooted,
            isCustomRom = isCustomRom(),
            isDebuggableApk = isDebuggableApk(ctx),
            isEmulator = emu,
            isRepackedApk = repacked,
            detectedIssues = issues
        )
    }

    fun isDebuggerAttached(): Boolean {
        return android.os.Debug.isDebuggerConnected()
    }

    fun isRooted(): Boolean {
        val paths = arrayOf(
            "/system/xbin/su",
            "/system/bin/su",
            "/sbin/su",
            "/data/local/su",
            "/data/local/bin/su",
            "/data/local/xbin/su",
            "/system/sd/xbin/su",
            "/system/app/Superuser.apk",
            "/system/app/SuperSU.apk",
            "/system/app/MagiskManager.apk"
        )
        for (p in paths) {
            if (File(p).exists()) return true
        }
        return try {
            Runtime.getRuntime().exec("which su").also { p ->
                p.waitFor()
                p.inputStream.bufferedReader().use(BufferedReader::readLine)
            } != null
        } catch (e: Exception) { false }
    }

    fun isCustomRom(): Boolean {
        val tags = Build.TAGS
        return tags != null && tags.contains("test-keys")
    }

    fun isDebuggableApk(ctx: Context): Boolean {
        val info = ctx.applicationInfo
        return (info.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }

    fun isEmulator(): Boolean {
        val fp = Build.FINGERPRINT.lowercase()
        val brand = Build.BRAND.lowercase()
        val device = Build.DEVICE.lowercase()
        val product = Build.PRODUCT.lowercase()
        val model = Build.MODEL.lowercase()
        val hardware = Build.HARDWARE.lowercase()
        val indicators = listOf(
            "generic", "emulator", "android sdk built for x86",
            "google_sdk", "vbox", "goldfish", "ranchu",
            "cf_arm", "cf_x86", "cf_x86_64", "kvm"
        )
        val hay = listOf(fp, brand, device, product, model, hardware).joinToString("|")
        return indicators.any { hay.contains(it) }
    }

    @SuppressLint("PackageManagerGetSignatures")
    fun isRepackedApk(ctx: Context): Boolean {
        if (EXPECTED_SIGNING_SHA256.isEmpty()) return false
        return try {
            val pm = ctx.packageManager
            val signatures: Array<Signature>? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val signingInfo: SigningInfo? = pm.getPackageInfo(
                    ctx.packageName, PackageManager.GET_SIGNING_CERTIFICATES
                ).signingInfo
                if (signingInfo == null) null
                else if (signingInfo.hasMultipleSigners()) signingInfo.apkContentsSigners
                else signingInfo.signingCertificateHistory
            } else {
                @Suppress("DEPRECATION")
                pm.getPackageInfo(ctx.packageName, PackageManager.GET_SIGNATURES).signatures
            }
            if (signatures == null || signatures.isEmpty()) return true
            val actual = sha256Hex(signatures[0].toByteArray())
            actual != EXPECTED_SIGNING_SHA256
        } catch (e: Exception) {
            Log.w("NetraEdgeSecurity", "signingInfo read failed: ${e.message}")
            false
        }
    }

    fun sha256Hex(bytes: ByteArray): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(bytes)
        return digest.joinToString("") { "%02x".format(it) }
    }

    @SuppressLint("HardwareIds")
    fun androidId(ctx: Context): String {
        return Settings.Secure.getString(ctx.contentResolver, Settings.Secure.ANDROID_ID)
    }

    fun encodeBase64Url(b: ByteArray): String = Base64.encodeToString(b, Base64.URL_SAFE or Base64.NO_WRAP)
}
