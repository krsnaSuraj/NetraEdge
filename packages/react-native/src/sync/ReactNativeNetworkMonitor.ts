/**
 * React Native network monitor — tracks device connectivity.
 *
 * Uses @react-native-community/netinfo to detect network state changes.
 * Implements the NetworkMonitor interface from @netraedge/core.
 */

import type { NetworkMonitor } from '@netraedge/core';

export type ConnectivityCallback = (online: boolean) => void;

interface NetInfoModule {
  fetch(): Promise<{ isConnected: boolean | null; isInternetReachable: boolean | null }>;
  addEventListener(callback: (state: { isConnected: boolean | null; isInternetReachable: boolean | null }) => void): () => void;
}

export class ReactNativeNetworkMonitor implements NetworkMonitor {
  private readonly _listeners = new Set<ConnectivityCallback>();
  private _unsubscribe: (() => void) | null = null;
  private readonly _netInfo: NetInfoModule;
  private _disposed = false;

  constructor(netInfo: NetInfoModule) {
    this._netInfo = netInfo;
    this._unsubscribe = this._netInfo.addEventListener((state) => {
      if (this._disposed) return;
      const online = state.isConnected === true && state.isInternetReachable !== false;
      for (const cb of this._listeners) {
        try {
          cb(online);
        } catch {
          /* listener error should not break monitoring */
        }
      }
    });
  }

  async isOnline(): Promise<boolean> {
    if (this._disposed) return false;
    const state = await this._netInfo.fetch();
    return state.isConnected === true && state.isInternetReachable !== false;
  }

  onConnectivityChange(callback: ConnectivityCallback): () => void {
    this._listeners.add(callback);
    return () => {
      this._listeners.delete(callback);
    };
  }

  dispose(): void {
    this._disposed = true;
    this._unsubscribe?.();
    this._unsubscribe = null;
    this._listeners.clear();
  }
}
