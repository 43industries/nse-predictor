/** Browser localStorage helpers for desk filters / favourites. */

export const PREF_KEYS = {
  minProbUp: "nse_portfolio_min_prob_up",
  minHitRate: "nse_portfolio_min_hit_rate",
  marketMinProb: "nse_market_min_prob",
  marketFavourites: "nse_market_favourites",
} as const;

export function loadNumber(key: string, fallback: number): number {
  if (typeof window === "undefined") return fallback;
  const n = Number(window.localStorage.getItem(key));
  return Number.isFinite(n) ? n : fallback;
}

export function saveNumber(key: string, value: number): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, String(value));
}

export function loadStringList(key: string): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.map((c) => String(c).toUpperCase()) : [];
  } catch {
    return [];
  }
}

export function saveStringList(key: string, values: string[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, JSON.stringify(values));
}

export function toggleInList(values: string[], code: string): string[] {
  const c = code.toUpperCase();
  return values.includes(c) ? values.filter((x) => x !== c) : [...values, c];
}
