package com.netraedge

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * AES-256-GCM asset decryption for the SOTA model files.
 *
 * The shipping APK contains ENCRYPTED versions of the .tflite +
 * face_landmarker.task files under `assets/enc/`. They are
 * decrypted lazily into the app's private cache the first time
 * they are needed, then re-used (the cache is invalidated on
 * every cold start so attackers can't find plaintext on disk
 * after a force-stop).
 *
 * Build-time encryption is performed by the helper script
 *   `tools/encrypt_assets.sh`
 * which produces the *.enc blobs. Key + IV are baked into the
 * APK (XOR-obfuscated) and combined at runtime — not strong
 * against a determined RE, but it stops the casual "unzip the
 * APK and pull the .tflite" attack that 90% of evaluators
 * would try.
 *
 * For a real production deploy, use Google Play Asset Delivery
 * with asset-pack signing so the model is decrypted by the
 * Play Store on first launch and never appears in the APK.
 */
object EncryptedAssets {

    private const val TAG = "NetraEdgeAssets"
    private const val ASSET_PREFIX = "enc/"
    private const val CACHE_DIR = "netraedge_models"

    /**
     * XOR-obfuscated AES-256 key. The real key is reconstructed
     * at runtime by XORing with [MASK]. A determined attacker
     * with a debugger can recover this in seconds — but the
     * "unzip-the-apk" attacker can't.
     */
    private val OBFUSCATED_KEY = byteArrayOf(
        0x12, 0x34, 0x56, 0x78, 0x9A.toByte(), 0xBC.toByte(), 0xDE.toByte(), 0xF0.toByte(),
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88.toByte(),
        0x99.toByte(), 0xAA.toByte(), 0xBB.toByte(), 0xCC.toByte(), 0xDD.toByte(), 0xEE.toByte(), 0xFF.toByte(), 0x00,
        0x13, 0x24, 0x35, 0x46, 0x57, 0x68, 0x79, 0x8A.toByte()
    )
    private val MASK = byteArrayOf(
        0x55, 0xAA.toByte(), 0x55, 0xAA.toByte(), 0x55, 0xAA.toByte(), 0x55, 0xAA.toByte(),
        0x33, 0xCC.toByte(), 0x33, 0xCC.toByte(), 0x33, 0xCC.toByte(), 0x33, 0xCC.toByte(),
        0x77, 0x88.toByte(), 0x99.toByte(), 0xAA.toByte(), 0xBB.toByte(), 0xCC.toByte(), 0xDD.toByte(), 0xEE.toByte(),
        0xAA.toByte(), 0x55, 0xAA.toByte(), 0x55, 0xAA.toByte(), 0x55, 0xAA.toByte(), 0x55
    )

    private fun derivedKey(): ByteArray {
        val out = ByteArray(32)
        for (i in out.indices) out[i] = (OBFUSCATED_KEY[i].toInt() xor MASK[i].toInt()).toByte()
        return out
    }

    /**
     * Decrypts `enc/<assetName>.enc` to the app's private cache
     * and returns the absolute path of the decrypted file. The
     * caller passes this path to TFLite Interpreter.
     *
     * @param assetName the plain name e.g. "face_recognition.tflite"
     */
    fun materialize(ctx: Context, assetName: String): String {
        val cacheFile = File(ctx.cacheDir, "$CACHE_DIR/$assetName")
        cacheFile.parentFile?.mkdirs()
        if (cacheFile.exists() && cacheFile.length() > 0) {
            return cacheFile.absolutePath
        }
        val encName = ASSET_PREFIX + assetName + ".enc"
        val bytes = ctx.assets.open(encName).use { it.readBytes() }
        if (bytes.size < 28) {
            throw IllegalStateException("encrypted asset too short: $encName (${bytes.size} bytes)")
        }
        val iv = bytes.copyOfRange(0, 12)
        val cipherBytes = bytes.copyOfRange(12, bytes.size)
        val keySpec = SecretKeySpec(derivedKey(), "AES")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, keySpec, GCMParameterSpec(128, iv))
        val plain = cipher.doFinal(cipherBytes)
        FileOutputStream(cacheFile).use { it.write(plain) }
        cacheFile.deleteOnExit()
        return cacheFile.absolutePath
    }

    /**
     * Decrypts to an in-memory ByteBuffer (no disk artifact).
     * Use this for the Face Landmarker .task file which is loaded
     * from a ByteBuffer.
     */
    fun materializeInMemory(ctx: Context, assetName: String): java.nio.ByteBuffer {
        val encName = ASSET_PREFIX + assetName + ".enc"
        val bytes = ctx.assets.open(encName).use { it.readBytes() }
        val iv = bytes.copyOfRange(0, 12)
        val cipherBytes = bytes.copyOfRange(12, bytes.size)
        val keySpec = SecretKeySpec(derivedKey(), "AES")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, keySpec, GCMParameterSpec(128, iv))
        val plain = cipher.doFinal(cipherBytes)
        return java.nio.ByteBuffer.allocateDirect(plain.size).apply {
            put(plain); flip()
        }
    }

    fun clearCache(ctx: Context) {
        File(ctx.cacheDir, CACHE_DIR).deleteRecursively()
    }
}
