/**
 * App context — provides pipeline and sync manager to the component tree.
 *
 * Wraps React Context to avoid prop drilling. Components access
 * the pipeline and sync manager via useAppContext().
 */

import { createContext, useContext } from 'react';
import type { FacePipeline, SyncManager } from '@netraedge/core';

export interface AppContextValue {
  readonly pipeline: FacePipeline;
  readonly syncManager: SyncManager | null;
}

export const AppContext = createContext<AppContextValue | null>(null);

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return ctx;
}

export const useAppContextValue = useAppContext;
