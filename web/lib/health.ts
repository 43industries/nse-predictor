import { asList, asRecord, dig, num, pct } from "./format";

export type HealthView = {
  hasMonitor: boolean;
  monitorStatus: string;
  score: Record<string, unknown>;
  laggedHit: number | null;
  gatesM: Record<string, unknown>;
  alerts: Record<string, unknown>[];
  checkedAt: unknown;
  problem: Record<string, unknown>;
  deadZone: number;
  gates: Record<string, unknown>;
  ric: Record<string, unknown>;
  ricb: Record<string, unknown>;
  cal: Record<string, unknown>;
  wfSum: Record<string, unknown>;
  fragile: boolean;
  abstain: number | null;
  atChance: string[];
  sel: Record<string, unknown>;
  rq: Record<string, unknown>;
  proto: Record<string, unknown>;
  selBits: string[];
  baselineChart: { label: string; acc: number }[];
  wf: Record<string, unknown>[];
  wfBeats: number;
  cov: Record<string, unknown>;
  confusion: Record<string, unknown>;
  perStock: Record<string, unknown>[];
  stress: Record<string, unknown>;
  calBins: { mean_prob: number; frac_up: number; perfect: number }[];
  collection: {
    status: string;
    lastRun: string;
    files: string;
    source: string;
  };
};

export function buildHealthView(
  evalJson: Record<string, unknown> | null,
  monitor: Record<string, unknown> | null,
  collection: Record<string, unknown> | null
): HealthView {
  const mon = monitor || {};
  const score = asRecord(dig(mon, ["score"], {}));
  const outcomes = asRecord(dig(mon, ["outcomes"], {}));
  const gatesM = asRecord(dig(mon, ["gates"], {}));
  const diag = asRecord(dig(evalJson, ["diagnostics"], {}));
  const gates = asRecord(dig(evalJson, ["validation_gates"], {}));
  const ric = asRecord(dig(diag, ["rank_ic"], {}));
  const ricb = asRecord(dig(diag, ["rank_ic_bootstrap"], {}));
  const cal = asRecord(dig(diag, ["calibration"], {}));
  const wfSum = asRecord(dig(evalJson, ["walk_forward_summary"], {}));
  const sel = asRecord(dig(evalJson, ["selection"], {}));
  const rq = asRecord(dig(sel, ["return_quantiles"], {}));
  const proto = asRecord(dig(evalJson, ["protocol"], {}));
  const baselines = asRecord(dig(evalJson, ["baselines"], {}));
  const models = asRecord(dig(evalJson, ["models"], {}));
  const lgbm = asRecord(dig(models, ["lightgbm_direction"], {}));
  const problem = asRecord(dig(evalJson, ["problem"], {}));
  const wf = asList(dig(evalJson, ["walk_forward"], []));
  const perStock = asList(dig(diag, ["per_stock"], []));
  const rawAtChance = dig(diag, ["at_chance_stocks"], []) as unknown;
  const atChance = Array.isArray(rawAtChance)
    ? rawAtChance.map(String)
    : perStock.filter((r) => r.flag === "at_chance").map((r) => String(r.stock_code));

  let abstain = dig(diag, ["abstain_rate"], null) as number | null;
  if (abstain == null) {
    abstain = dig(lgbm, ["abstain_rate"], null) as number | null;
  }

  const validUtility = sel.valid_utility as number | undefined;
  const validDir = asRecord(dig(sel, ["valid_direction"], {}));
  const selBits: string[] = [];
  if (typeof validUtility === "number" && !Number.isNaN(validUtility)) {
    selBits.push(`valid utility ${pct(validUtility, 1)}`);
  }
  if (validDir.accuracy != null) {
    const lift = validDir.lift_vs_always_up;
    const liftTxt =
      lift != null
        ? `${Number(lift) >= 0 ? "+" : ""}${pct(lift as number, 1)}`
        : "—";
    selBits.push(`valid accuracy ${pct(validDir.accuracy as number, 1)} · lift vs always-up ${liftTxt}`);
  }
  if (rq.vol_scaled) {
    selBits.push(`vol-scaled bands (ref=${num(rq.vol_scale_ref as number, 4)})`);
  }

  const calRaw = asList(dig(diag, ["calibration_bins"], []));
  const calBins = calRaw
    .filter((r) => r.mean_prob != null && r.frac_up != null)
    .map((r) => ({
      mean_prob: Number(r.mean_prob),
      frac_up: Number(r.frac_up),
      perfect: Number(r.mean_prob),
    }))
    .sort((a, b) => a.mean_prob - b.mean_prob);

  const col = collection || {};

  return {
    hasMonitor: Boolean(mon.status || mon.checked_at || mon.score),
    monitorStatus: String(mon.status || "—").toUpperCase(),
    score,
    laggedHit: outcomes.skipped ? null : ((outcomes.hit_rate_nonflat as number | null) ?? null),
    gatesM,
    alerts: asList(dig(mon, ["alerts"], [])),
    checkedAt: mon.checked_at,
    problem,
    deadZone: Number(problem.dead_zone_xs ?? 0.01),
    gates,
    ric,
    ricb,
    cal,
    wfSum,
    fragile: Boolean(dig(evalJson, ["stress_fragile"], false)),
    abstain,
    atChance,
    sel,
    rq,
    proto,
    selBits,
    baselineChart: [
      {
        label: "Always up",
        acc: Number(dig(baselines, ["always_up", "accuracy"], 0) || 0) * 100,
      },
      {
        label: "MA 5/20",
        acc: Number(dig(baselines, ["ma_5_20", "accuracy"], 0) || 0) * 100,
      },
      { label: "LightGBM", acc: Number(lgbm.accuracy || 0) * 100 },
    ],
    wf,
    wfBeats: wf.filter((f) => f.beats_baseline).length,
    cov: asRecord(dig(models, ["return_interval_coverage"], {})),
    confusion: asRecord(dig(diag, ["confusion"], {})),
    perStock,
    stress: asRecord(dig(evalJson, ["stress_rank_ic"], {})),
    calBins,
    collection: {
      status: String(col.status || col.ok || "—"),
      lastRun: String(col.finished_at || col.written_at || col.checked_at || "").slice(0, 19) || "—",
      files: String(col.n_files ?? col.files ?? col.n_rows ?? "—"),
      source: String(col.source || col.path || "drop-folder"),
    },
  };
}
