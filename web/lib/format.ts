export function dig(obj: unknown, path: string[], fallback: unknown = null): unknown {
  let cur: any = obj;
  for (const p of path) {
    if (cur == null) return fallback;
    cur = cur[p];
  }
  return cur ?? fallback;
}

export function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

export function asList(v: unknown): Record<string, unknown>[] {
  return Array.isArray(v) ? (v as Record<string, unknown>[]) : [];
}

export function pct(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(digits);
}

export function fmtChecked(v: unknown): string {
  if (v == null || v === "") return "";
  return String(v).slice(0, 19);
}

export function callLabel(...parts: Array<string | null | undefined>): string {
  for (const p of parts) {
    if (p) return String(p).toUpperCase();
  }
  return "—";
}

export function rangeLabel(
  lo: number | null | undefined,
  hi: number | null | undefined,
  digits = 1
): string {
  if (lo == null || hi == null) return "—";
  return `${pct(lo, digits)} … ${pct(hi, digits)}`;
}

export function stockPath(code: string): string {
  return `/stock/${encodeURIComponent(String(code).toUpperCase())}`;
}

export function cellText(v: unknown, digits = 3): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number(v).toFixed(digits);
  return String(v);
}
