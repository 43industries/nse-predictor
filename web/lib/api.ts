import { num, pct } from "./format";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number
  ) {
    super(message);
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(
      `API unreachable at ${API}. Start: uvicorn api.main:app --reload --app-dir .`
    );
  }
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(text || `HTTP ${res.status}`, res.status);
  }
  return res.json() as Promise<T>;
}

function getJson<T>(path: string) {
  return requestJson<T>(path);
}

function postJson<T>(path: string, body: unknown) {
  return requestJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export type Forecast = {
  stock_code: string;
  as_of_date?: string;
  price?: number | null;
  direction_5d?: string;
  pred_label?: string;
  prob_up?: number;
  ret5d_point?: number | null;
  ret5d_low?: number | null;
  ret5d_high?: number | null;
  vol20d?: number | null;
  disclaimer?: string;
};

export type SetupRow = {
  stock_code: string;
  trade_date?: string | null;
  price?: number | null;
  pred_label?: string | null;
  prob_up?: number | null;
  ret5d_lo?: number | null;
  ret5d_hi?: number | null;
  vol20d?: number | null;
  hit_rate?: number | null;
  flag?: string | null;
  skip_reason?: string | null;
};

export type DeskBrief = {
  as_of?: string | null;
  n_setups: number;
  n_skips: number;
  threshold: number;
  setups: SetupRow[];
  skips: SetupRow[];
  disclaimer: string;
};

export type JournalEntry = {
  logged_at?: string;
  as_of_date?: string | null;
  stock_code: string;
  stance: string;
  prob_up?: number | null;
  pred_label?: string | null;
  price?: number | null;
  note?: string;
  source?: string;
};

export type BankRow = {
  stock_code: string;
  bank_name: string;
  period?: string;
  pat?: number | null;
  pat_yoy?: number | null;
  roe?: number | null;
  roa?: number | null;
  nim?: number | null;
  cir?: number | null;
  npl?: number | null;
  assets?: number | null;
  loans?: number | null;
  deposits?: number | null;
  nii?: number | null;
  nfi?: number | null;
  eps?: number | null;
  dividend?: number | null;
};

export type JournalLogBody = {
  stock_code: string;
  stance: "long" | "flat" | "skip";
  note?: string;
  as_of_date?: string | null;
  prob_up?: number | null;
  pred_label?: string | null;
  price?: number | null;
};

export const api = {
  base: API,
  deskBrief: () => getJson<DeskBrief>("/companion/desk-brief"),
  journal: () => getJson<{ entries: JournalEntry[]; disclaimer: string }>("/companion/journal"),
  logJournal: (body: JournalLogBody) => postJson<{ ok: boolean }>("/companion/journal", body),
  forecasts: () => getJson<Forecast[]>("/forecasts"),
  forecast: (code: string) => getJson<Forecast>(`/forecasts/${encodeURIComponent(code)}`),
  history: (code: string, limit = 252) =>
    getJson<{ stock_code: string; points: { date: string; price: number; volume: number }[] }>(
      `/history/${encodeURIComponent(code)}?limit=${limit}`
    ),
  accuracy: (code: string) =>
    getJson<{ stock_code: string; hit_rate?: number; n?: number; disclaimer?: string }>(
      `/accuracy/${encodeURIComponent(code)}`
    ),
  drivers: () => getJson<{ drivers: { feature?: string; mean_abs_shap?: number }[] }>("/drivers"),
  banks: () => getJson<{ banks: BankRow[]; disclaimer: string }>("/fundamentals/banks"),
  evaluation: () => getJson<Record<string, unknown>>("/evaluation"),
  collectionStatus: () => getJson<Record<string, unknown>>("/collection/status"),
  monitor: () => getJson<Record<string, unknown>>("/monitor"),
  monitorHistory: (tail = 14) =>
    getJson<{ rows: Record<string, unknown>[]; disclaimer?: string }>(
      `/monitor/history?tail=${tail}`
    ),
  disclaimer: () => getJson<{ disclaimer: string }>("/disclaimer"),
};

export function logoUrl(code: string): string {
  return `${API}/logos/${encodeURIComponent(String(code).toUpperCase())}`;
}

export { num, pct };
