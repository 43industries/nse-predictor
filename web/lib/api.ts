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
  hit_rate?: number | null;
  held_shares?: number | null;
  weight?: number | null;
  book_status?: "flat" | "held" | "overweight" | string | null;
  suggested_side?: "buy" | "sell" | null;
  suggested_shares?: number | null;
  suggest_note?: string | null;
  disclaimer?: string;
};

export type MoverRow = {
  stock_code: string;
  price?: number | null;
  ret_1d: number;
  volume?: number;
  trade_date?: string | null;
};

export type MarketMovers = {
  as_of?: string | null;
  gainers: MoverRow[];
  losers: MoverRow[];
  movers: MoverRow[];
  disclaimer?: string;
  note?: string;
};

export type MarketBoard = {
  as_of?: string | null;
  rows: Forecast[];
  n: number;
  threshold?: number;
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
  skip_label?: string | null;
  held_shares?: number | null;
  weight?: number | null;
  book_status?: "flat" | "held" | "overweight" | string | null;
  suggested_side?: "buy" | "sell" | null;
  suggested_shares?: number | null;
  suggest_note?: string | null;
};

export type DeskWatchlistItem = {
  stock_code: string;
  prob_up?: number | null;
  pred_label?: string | null;
  skip_reason?: string | null;
  skip_label?: string | null;
  in_setups?: boolean;
};

export type DeskBriefSummary = {
  as_of?: string | null;
  n_setups: number;
  n_skips: number;
  n_up: number;
  n_down: number;
  n_flat: number;
  entry_threshold: number;
  min_prob_up: number;
  min_hit_rate: number;
  n_at_chance: number;
  model_caution: boolean;
  headline: string;
};

export type DeskBrief = {
  as_of?: string | null;
  n_setups: number;
  n_skips: number;
  threshold: number;
  filters?: {
    min_prob_up?: number;
    min_hit_rate?: number;
    threshold?: number;
    entry_threshold?: number;
    top_n?: number;
  };
  setups: SetupRow[];
  skips: SetupRow[];
  summary?: DeskBriefSummary;
  watchlist?: DeskWatchlistItem[];
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

export type PortfolioPosition = {
  stock_code: string;
  shares: number;
  avg_cost: number;
  last_price?: number | null;
  market_value?: number | null;
  unrealized_pnl?: number | null;
  weight?: number | null;
  opened_at?: string | null;
};

export type PortfolioRisk = {
  policy?: {
    max_names?: number;
    max_weight_pct?: number;
    min_cash_pct?: number;
    lot_size?: number;
    cost_bps?: number;
  };
  utilization?: {
    n_names?: number;
    cash_pct?: number;
    max_position_weight?: number;
    nav_kes?: number;
    cash_kes?: number;
  };
  breaches?: string[];
};

export type PortfolioBook = {
  cash_kes: number;
  currency?: string;
  lot_size?: number;
  cost_bps?: number;
  mode?: string;
  updated_at?: string;
  positions: PortfolioPosition[];
  equity_kes?: number;
  nav_kes?: number;
  cost_basis_kes?: number;
  unrealized_pnl_kes?: number;
  n_names?: number;
  risk?: PortfolioRisk;
  disclaimer?: string;
};

export type PortfolioSuggestion = {
  stock_code: string;
  action: string;
  side?: "buy" | "sell" | null;
  shares: number;
  prob_up?: number;
  pred_label?: string;
  price?: number | null;
  hit_rate?: number | null;
  conviction?: number;
  est_notional_kes?: number;
  reason?: string;
  priority?: number;
};

export type PortfolioSuggestions = {
  as_of?: string | null;
  nav_kes?: number;
  cash_kes?: number;
  suggestions: PortfolioSuggestion[];
  n_suggestions: number;
  risk?: Record<string, unknown>;
  filters?: { min_prob_up?: number; min_hit_rate?: number; threshold?: number };
  disclaimer?: string;
  mode?: string;
};

export type NavPoint = {
  date: string;
  nav_kes: number;
  cash_kes: number;
  equity_kes: number;
  unrealized_pnl_kes: number;
};

export type PortfolioTicket = {
  id: string;
  status: string;
  stock_code: string;
  side: "buy" | "sell" | string;
  shares: number;
  limit_price?: number | null;
  reason?: string;
  suggestion_action?: string | null;
  created_at?: string;
  resolved_at?: string | null;
  fill_price?: number | null;
};

export type PortfolioFill = {
  ticket_id: string;
  stock_code: string;
  side: string;
  shares: number;
  price: number;
  fees_bps?: number;
  fees_kes?: number;
  notional_kes?: number;
  ts?: string;
  source?: string;
};

export type TicketCreateBody = {
  stock_code: string;
  side: "buy" | "sell";
  shares: number;
  reason?: string;
  limit_price?: number | null;
  suggestion_action?: string | null;
};

export const api = {
  base: API,
  deskBrief: (opts?: { top_n?: number; min_prob_up?: number; min_hit_rate?: number }) => {
    const q = new URLSearchParams();
    if (opts?.top_n != null) q.set("top_n", String(opts.top_n));
    if (opts?.min_prob_up != null) q.set("min_prob_up", String(opts.min_prob_up));
    if (opts?.min_hit_rate != null) q.set("min_hit_rate", String(opts.min_hit_rate));
    const qs = q.toString();
    return getJson<DeskBrief>(qs ? `/companion/desk-brief?${qs}` : "/companion/desk-brief");
  },
  journal: () => getJson<{ entries: JournalEntry[]; disclaimer: string }>("/companion/journal"),
  logJournal: (body: JournalLogBody) => postJson<{ ok: boolean }>("/companion/journal", body),
  forecasts: () => getJson<Forecast[]>("/forecasts"),
  marketBoard: () => getJson<MarketBoard>("/market/board"),
  marketMovers: (top_n = 5) => getJson<MarketMovers>(`/market/movers?top_n=${top_n}`),
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
  portfolio: () => getJson<PortfolioBook>("/portfolio"),
  portfolioSuggestions: (opts?: { min_prob_up?: number; min_hit_rate?: number }) => {
    const q = new URLSearchParams();
    if (opts?.min_prob_up != null) q.set("min_prob_up", String(opts.min_prob_up));
    if (opts?.min_hit_rate != null) q.set("min_hit_rate", String(opts.min_hit_rate));
    const qs = q.toString();
    return getJson<PortfolioSuggestions>(
      qs ? `/portfolio/suggestions?${qs}` : "/portfolio/suggestions"
    );
  },
  portfolioNavHistory: (days = 90) =>
    getJson<{ points: NavPoint[]; days: number; disclaimer?: string }>(
      `/portfolio/nav-history?days=${days}`
    ),
  portfolioTickets: (status?: string) =>
    getJson<{ tickets: PortfolioTicket[]; disclaimer?: string }>(
      status ? `/portfolio/tickets?status=${encodeURIComponent(status)}` : "/portfolio/tickets"
    ),
  portfolioFills: (limit = 50) =>
    getJson<{ fills: PortfolioFill[]; disclaimer?: string }>(`/portfolio/fills?limit=${limit}`),
  portfolioCreateTicket: (body: TicketCreateBody) =>
    postJson<{ ticket: PortfolioTicket; disclaimer?: string }>("/portfolio/tickets", body),
  portfolioCreateTicketsBatch: (tickets: TicketCreateBody[]) =>
    postJson<{
      created: PortfolioTicket[];
      errors: { index: number; stock_code?: string; error: string }[];
      n_created: number;
      n_errors: number;
    }>("/portfolio/tickets/batch", { tickets }),
  portfolioConfirmTicket: (id: string, price?: number | null) =>
    postJson<{ ticket: PortfolioTicket; fill: PortfolioFill; book: PortfolioBook }>(
      `/portfolio/tickets/${encodeURIComponent(id)}/confirm`,
      price != null ? { price } : {}
    ),
  portfolioRejectTicket: (id: string) =>
    postJson<{ ticket: PortfolioTicket }>(`/portfolio/tickets/${encodeURIComponent(id)}/reject`, {}),
  portfolioReset: (cash_kes?: number) =>
    postJson<PortfolioBook>("/portfolio/reset", cash_kes != null ? { cash_kes } : {}),
  portfolioSeedPosition: (body: { stock_code: string; shares: number; avg_cost: number }) =>
    postJson<PortfolioBook>("/portfolio/positions", body),
};

export function logoUrl(code: string): string {
  return `${API}/logos/${encodeURIComponent(String(code).toUpperCase())}`;
}

export { num, pct };
