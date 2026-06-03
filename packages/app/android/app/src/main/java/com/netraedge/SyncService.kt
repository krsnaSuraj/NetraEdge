package com.netraedge

import android.content.Context
import android.net.wifi.WifiManager
import android.util.Log
import java.net.HttpURLConnection
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.URL
import kotlin.random.Random
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * NetraEdge Sync — uploads the local enrolled embedding to the cloud
 * (mock server or AWS Lambda) with Differential Privacy noise applied.
 *
 *  1. Auto-detect reachable host:
 *     - Emulator: 10.0.2.2 (host loopback alias)
 *     - Real device: WiFi gateway IP (e.g., 192.168.1.1) or override via prefs
 *  2. Apply Laplace noise to the 128-dim embedding (Dwork & Roth 2014).
 *  3. POST to /enrollments as JSON.
 *  4. On success → mark local record as synced, eligible for purge.
 *
 *  Threading: all network calls run on Dispatchers.IO. Callbacks delivered
 *  on Dispatchers.Main.
 *
 *  URL resolution priority:
 *    1. User-set override in SharedPreferences (NetraEdge/sync_url)
 *    2. Detected gateway IP : 4000
 *    3. 10.0.2.2 : 4000 (emulator fallback)
 */
class SyncService(
    private val appContext: Context,
    private val endpoint: String = "",
    private val apiKey: String = DEFAULT_API_KEY
) {

    interface Callback {
        fun onStarted()
        fun onSuccess(serverId: String)
        fun onError(message: String)
    }

    /**
     * Apply Laplace noise to a 128-dim embedding.
     * Equivalent to the JS DPNoise in packages/core/src/sync/DPNoise.ts.
     *
     *   scale = sensitivity / epsilon
     *   sample = -scale * sign(u) * ln(1 - 2|u|),  u ~ U(-0.5, 0.5)
     */
    private fun applyDpNoise(embedding: FloatArray, epsilon: Float = 1.0f, sensitivity: Float = 2.0f): FloatArray {
        val scale = sensitivity / epsilon
        val out = FloatArray(embedding.size)
        for (i in embedding.indices) {
            val u = Random.nextDouble(-0.5, 0.5)
            val noise = if (u == 0.0) 0.0 else -scale * Math.signum(u) * Math.log(1.0 - 2.0 * Math.abs(u))
            out[i] = embedding[i] + noise.toFloat()
        }
        // Re-normalize to unit L2
        var norm = 0.0
        for (v in out) norm += v * v
        norm = Math.sqrt(norm).coerceAtLeast(1e-9)
        for (i in out.indices) out[i] = (out[i] / norm).toFloat()
        return out
    }

    /**
     * Resolve the sync endpoint:
     *  1. Pref override (user manually configured)
     *  2. WiFi gateway IP (works for real device on same network as the server)
     *  3. 10.0.2.2 (emulator only)
     */
    private fun resolveEndpoint(): String {
        if (endpoint.isNotEmpty()) return endpoint

        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.getString(KEY_SYNC_URL, null)?.let {
            if (it.isNotBlank()) return it.trimEnd('/')
        }

        // Try WiFi gateway IP — works for real device on same LAN
        detectGatewayIp()?.let { ip ->
            return "http://$ip:4000"
        }

        // Emulator fallback
        return "http://10.0.2.2:4000"
    }

    /**
     * Detect the device's WiFi gateway IP. This is the host PC's address on the
     * local network when the device is connected via WiFi.
     *
     * Returns null on emulator (Android emulator WiFi reports no DHCP gateway)
     * or if no network is connected.
     */
    private fun detectGatewayIp(): String? {
        try {
            // Method 1: WifiManager DHCP gateway
            val wm = appContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            if (wm != null && wm.isWifiEnabled) {
                val dhcp = wm.dhcpInfo
                if (dhcp != null && dhcp.gateway != 0) {
                    val ip = formatIp(dhcp.gateway)
                    if (ip != null && ip != "0.0.0.0") {
                        Log.i(TAG, "WiFi DHCP gateway: $ip")
                        return ip
                    }
                }
            }

            // Method 2: iterate network interfaces and find the host's reachable IP
            // (looks for 192.168.x.x or 10.x.x.x — the device's own subnet)
            val interfaces = NetworkInterface.getNetworkInterfaces()?.toList() ?: emptyList()
            for (intf in interfaces) {
                if (!intf.isUp || intf.isLoopback || intf.isVirtual) continue
                val addrs: List<java.net.InetAddress> = try {
                    val list = intf.inetAddresses
                    val result = ArrayList<java.net.InetAddress>()
                    while (list.hasMoreElements()) result.add(list.nextElement())
                    result
                } catch (e: Exception) { continue }
                for (addr in addrs) {
                    if (addr is Inet4Address && !addr.isLoopbackAddress) {
                        val host = addr.hostAddress ?: continue
                        if (host.startsWith("192.168.") || host.startsWith("10.")) {
                            val parts = host.split(".")
                            if (parts.size == 4) {
                                val gatewayGuess = "${parts[0]}.${parts[1]}.${parts[2]}.1"
                                Log.i(TAG, "Guessed gateway from subnet: $gatewayGuess (device IP: $host)")
                                return gatewayGuess
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Gateway detection failed: ${e.message}")
        }
        return null
    }

    private fun formatIp(raw: Int): String? {
        return try {
            "${raw and 0xFF}.${raw shr 8 and 0xFF}.${raw shr 16 and 0xFF}.${raw shr 24 and 0xFF}"
        } catch (e: Exception) {
            null
        }
    }

    fun setOverrideEndpoint(url: String) {
        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_SYNC_URL, url).apply()
    }

    fun getOverrideEndpoint(): String? {
        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_SYNC_URL, null)
    }

    fun clearOverrideEndpoint() {
        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().remove(KEY_SYNC_URL).apply()
    }

    fun upload(
        userId: String,
        embedding: FloatArray,
        callback: Callback,
        scope: CoroutineScope
    ) {
        callback.onStarted()
        scope.launch(Dispatchers.IO) {
            val targetUrl = resolveEndpoint()
            try {
                val noisy = applyDpNoise(embedding)
                val payload = JSONObject().apply {
                    put("userId", userId)
                    put("embedding", JSONArray(noisy.toList()))
                    put("metadata", JSONObject().apply {
                        put("dpApplied", true)
                        put("epsilon", 1.0)
                        put("source", "netraedge-android-v1")
                    })
                    put("syncedAt", System.currentTimeMillis())
                }
                val url = URL("$targetUrl/enrollments")
                Log.i(TAG, "POST $url (embedding dim=${noisy.size}, DP ε=1.0)")
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    doOutput = true
                    connectTimeout = 5000
                    readTimeout = 8000
                    setRequestProperty("Content-Type", "application/json")
                    if (apiKey.isNotEmpty()) setRequestProperty("x-api-key", apiKey)
                }
                conn.outputStream.use { it.write(payload.toString().toByteArray()) }
                val code = conn.responseCode
                if (code in 200..299) {
                    val resp = conn.inputStream.bufferedReader().use { it.readText() }
                    val serverId = try {
                        JSONObject(resp).optString("serverId", "srv_${System.currentTimeMillis()}")
                    } catch (e: Exception) {
                        "srv_${System.currentTimeMillis()}"
                    }
                    Log.i(TAG, "Sync OK: $code, serverId=$serverId")
                    withContext(Dispatchers.Main) { callback.onSuccess(serverId) }
                } else {
                    val msg = "HTTP $code from $targetUrl: ${conn.errorStream?.bufferedReader()?.use { it.readText() } ?: ""}"
                    Log.w(TAG, "Sync failed: $msg")
                    withContext(Dispatchers.Main) { callback.onError(msg) }
                }
            } catch (e: java.net.ConnectException) {
                val msg = "Cannot reach $targetUrl — make sure the sync server is running and the device is on the same WiFi network"
                Log.w(TAG, "Sync connect exception: ${e.message}")
                withContext(Dispatchers.Main) { callback.onError(msg) }
            } catch (e: java.net.SocketTimeoutException) {
                val msg = "Sync timed out connecting to $targetUrl"
                Log.w(TAG, "Sync timeout: ${e.message}")
                withContext(Dispatchers.Main) { callback.onError(msg) }
            } catch (e: java.net.UnknownHostException) {
                val msg = "Unknown host — check WiFi connection (tried $targetUrl)"
                Log.w(TAG, "Sync unknown host: ${e.message}")
                withContext(Dispatchers.Main) { callback.onError(msg) }
            } catch (e: Exception) {
                Log.e(TAG, "Sync exception", e)
                withContext(Dispatchers.Main) { callback.onError("Sync error: ${e.message ?: "Unknown"}") }
            }
        }
    }

    companion object {
        private const val TAG = "SyncService"
        const val DEFAULT_API_KEY = "demo-key"
        const val PREFS_NAME = "netraedge_sync_prefs"
        const val KEY_SYNC_URL = "sync_url"
    }
}
