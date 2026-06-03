/**
 * i18n — lightweight runtime i18n for NetraEdge.
 *
 * - No external dependency (avoids i18next / react-intl bundle bloat)
 * - Sync access via `t(key)` / `t(key, params)`
 * - Locale persisted to AsyncStorage (optional)
 * - Falls back to English if a key is missing
 * - Supports parameter interpolation: `t('key', { name: 'X' })`
 *
 * Supported locales: en, hi.
 */

import en from './en.json';
import hi from './hi.json';

export type LocaleCode = 'en' | 'hi';

export const SUPPORTED_LOCALES: ReadonlyArray<{ code: LocaleCode; label: string }> = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिन्दी' },
];

type Dictionary = Readonly<Record<string, unknown>>;
type Leaf = string;

const dictionaries: Record<LocaleCode, Dictionary> = { en, hi };

let currentLocale: LocaleCode = 'en';
const subscribers = new Set<(locale: LocaleCode) => void>();

function getByPath(obj: Dictionary, path: string): unknown {
  const parts = path.split('.');
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = (cur as Dictionary)[p];
  }
  return cur;
}

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    const v = params[key];
    return v == null ? `{${key}}` : String(v);
  });
}

export function getLocale(): LocaleCode {
  return currentLocale;
}

export function setLocale(next: LocaleCode): void {
  if (next === currentLocale) return;
  if (!dictionaries[next]) return;
  currentLocale = next;
  for (const cb of subscribers) {
    try { cb(next); } catch { /* ignore */ }
  }
}

export function subscribeLocale(cb: (locale: LocaleCode) => void): () => void {
  subscribers.add(cb);
  return () => { subscribers.delete(cb); };
}

export function t(key: string, params?: Record<string, string | number>): string {
  const cur = getByPath(dictionaries[currentLocale], key);
  if (typeof cur === 'string') return interpolate(cur, params);
  // Fallback to English
  const en = getByPath(dictionaries.en, key);
  if (typeof en === 'string') return interpolate(en, params);
  return key;
}

/** For tests: reset to default. */
export function __resetI18nForTests(): void {
  currentLocale = 'en';
  subscribers.clear();
}

export type { Leaf };
