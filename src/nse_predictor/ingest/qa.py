"""Price / macro / sector data quality checks for NSE panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.config import ROOT, settings
from nse_predictor.universe import investable_equities, load_universe
from nse_predictor.util.datetime import to_ns

KNOWN_REMAPS = [
    {"from": "BBK", "to": "ABSA", "note": "Barclays Kenya rebranded to Absa"},
    {"from": "CFC", "to": "CFCI", "note": "Historical ticker variants — verify before merge"},
    {"from": "SAF", "to": "SCOM", "note": "Safaricom sometimes coded SAF in older lists"},
]

JUMP_ABS_RET = 0.40


def _load_investable(prices: pd.DataFrame) -> list[str]:
    uni = load_universe(prices)
    inv = investable_equities(uni)
    if inv:
        return inv
    code = prices["stock_code"].astype(str)
    eq = prices.loc[~code.str.contains(r"-R$|^\^", regex=True)]
    return (
        eq.groupby("stock_code")["volume"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )


def check_duplicates(prices: pd.DataFrame) -> dict:
    n = int(prices.duplicated(subset=["trade_date", "stock_code"], keep=False).sum())
    return {"duplicate_rows": n, "ok": n == 0}


def check_calendar_gaps(
    prices: pd.DataFrame, investable: list[str], max_gap_days: int = 10
) -> dict:
    """Flag long gaps between consecutive observations for investable names."""
    sub = prices.loc[prices["stock_code"].isin(investable), ["stock_code", "trade_date"]].copy()
    if sub.empty:
        return {"n_gap_events": 0, "examples": [], "ok": True}

    sub["trade_date"] = to_ns(sub["trade_date"])
    sub = sub.dropna(subset=["trade_date"]).sort_values(["stock_code", "trade_date"])
    market_days = set(sub["trade_date"].dt.normalize().unique())

    sub["prev"] = sub.groupby("stock_code", sort=False)["trade_date"].shift(1)
    sub["calendar_days"] = (sub["trade_date"] - sub["prev"]).dt.days
    candidates = sub[sub["calendar_days"] > max_gap_days].dropna(subset=["prev"])

    gaps: list[dict] = []
    for row in candidates.itertuples(index=False):
        a, b = row.prev, row.trade_date
        delta = int(row.calendar_days)
        between = pd.bdate_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1))
        missing_mkt = sum(1 for d in between if d in market_days)
        if missing_mkt >= 3 or delta > 21:
            gaps.append(
                {
                    "stock_code": row.stock_code,
                    "from": str(pd.Timestamp(a).date()),
                    "to": str(pd.Timestamp(b).date()),
                    "calendar_days": delta,
                    "market_days_skipped": int(missing_mkt),
                }
            )
    return {"n_gap_events": len(gaps), "examples": gaps[:40], "ok": len(gaps) < 50}


def check_zero_volume(prices: pd.DataFrame, investable: list[str]) -> dict:
    sub = prices.loc[prices["stock_code"].isin(investable)].copy()
    if sub.empty:
        return {"per_stock": [], "flagged": [], "ok": True}
    if "zero_volume" not in sub.columns:
        sub["zero_volume"] = sub["volume"].fillna(0) <= 0
    if "days_since_trade" not in sub.columns:
        sub["days_since_trade"] = 0

    stats = (
        sub.groupby("stock_code", sort=False)
        .agg(
            zero_volume_frac=("zero_volume", "mean"),
            max_days_since_trade=("days_since_trade", "max"),
        )
        .reset_index()
    )
    rows = [
        {
            "stock_code": r.stock_code,
            "zero_volume_frac": float(r.zero_volume_frac),
            "max_days_since_trade": int(r.max_days_since_trade),
        }
        for r in stats.itertuples(index=False)
    ]
    hot = [r for r in rows if r["zero_volume_frac"] > 0.3 or r["max_days_since_trade"] > 20]
    return {"per_stock": rows, "flagged": hot, "ok": len(hot) == 0}


def check_jumps(prices: pd.DataFrame, investable: list[str], thr: float = JUMP_ABS_RET) -> dict:
    sub = prices.loc[prices["stock_code"].isin(investable)].copy()
    if "ret_1d" not in sub.columns:
        sub = sub.sort_values(["stock_code", "trade_date"])
        sub["ret_1d"] = sub.groupby("stock_code", sort=False)["price"].pct_change()

    jump_mask = sub["ret_1d"].abs() > thr
    n_jumps = int(jump_mask.sum())
    big = sub.loc[jump_mask, ["trade_date", "stock_code", "price", "ret_1d", "volume"]].copy()
    if len(big) > 50:
        big = big.assign(_abs=big["ret_1d"].abs()).nlargest(50, "_abs").drop(columns=["_abs"])

    examples = [
        {
            "trade_date": str(pd.to_datetime(r.trade_date).date()),
            "stock_code": r.stock_code,
            "ret_1d": float(r.ret_1d),
            "price": float(r.price),
            "volume": float(r.volume) if pd.notna(r.volume) else 0.0,
        }
        for r in big.itertuples(index=False)
    ]

    # Adj vs close: investable only; both positive (avoids blank/"-" noise)
    adj_flags = 0
    if {"adj_price", "day_close"}.issubset(prices.columns):
        inv = prices.loc[prices["stock_code"].isin(investable)]
        both = inv["adj_price"].notna() & inv["day_close"].notna()
        both &= (inv["adj_price"] > 0) & (inv["day_close"] > 0)
        if both.any():
            ratio = inv.loc[both, "adj_price"] / inv.loc[both, "day_close"]
            adj_flags = int(((ratio - 1).abs() > 0.05).sum())

    return {
        "n_jumps": n_jumps,
        "threshold": thr,
        "examples": examples,
        "adj_vs_close_disagreements": adj_flags,
        "ok": n_jumps < 100,
    }


def check_ticker_hygiene(prices: pd.DataFrame) -> dict:
    codes = prices["stock_code"].astype(str).unique().tolist()
    rights = sorted(c for c in codes if c.endswith("-R"))
    indices = sorted(c for c in codes if c.startswith("^") or "NASI" in c or "N20" in c)
    present = set(codes)
    remap_hits = [
        {
            **item,
            "from_present": item["from"] in present,
            "to_present": item["to"] in present,
        }
        for item in KNOWN_REMAPS
    ]
    return {
        "n_rights": len(rights),
        "rights_sample": rights[:20],
        "n_indices": len(indices),
        "indices": indices,
        "known_remaps": remap_hits,
        "ok": True,
    }


def validate_macro(macro: pd.DataFrame, source: str) -> dict:
    if macro is None or macro.empty:
        return {"ok": False, "source": source or "missing", "error": "empty macro"}
    m = macro.copy()
    m["observe_date"] = to_ns(m["observe_date"])
    m = m.dropna(subset=["observe_date"]).sort_values("observe_date")
    if m.empty:
        return {"ok": False, "source": source or "missing", "error": "empty after parse"}
    mono = bool(m["observe_date"].is_monotonic_increasing)
    future_stub = int((m["observe_date"] > pd.Timestamp.today() + pd.Timedelta(days=30)).sum())
    return {
        "ok": mono and future_stub == 0 and len(m) > 10,
        "source": source,
        "n_rows": int(len(m)),
        "min_date": str(m["observe_date"].min().date()),
        "max_date": str(m["observe_date"].max().date()),
        "monotonic": mono,
        "future_stub_rows": future_stub,
    }


def check_macro_asof_leak(prices: pd.DataFrame, macro: pd.DataFrame, sample_n: int = 500) -> dict:
    """Sample join: macro observe_date must not be after trade_date."""
    if macro is None or macro.empty or prices.empty:
        return {"ok": True, "checked": 0, "leaks": 0}
    p = prices.sample(min(sample_n, len(prices)), random_state=42)[["trade_date"]].copy()
    p["trade_date"] = to_ns(p["trade_date"])
    p = p.dropna(subset=["trade_date"]).sort_values("trade_date")
    m = macro[["observe_date"]].copy()
    m["macro_asof"] = to_ns(m["observe_date"])
    m = m.dropna(subset=["macro_asof"]).sort_values("macro_asof")[["macro_asof"]]
    joined = pd.merge_asof(p, m, left_on="trade_date", right_on="macro_asof", direction="backward")
    leaks = joined["macro_asof"].notna() & (joined["macro_asof"] > joined["trade_date"])
    n_leaks = int(leaks.sum())
    return {"ok": n_leaks == 0, "checked": int(len(joined)), "leaks": n_leaks}


def check_sector_coverage(sectors: pd.DataFrame, investable: list[str]) -> dict:
    if sectors is None or sectors.empty:
        return {"ok": False, "coverage": 0.0, "orphans": list(investable), "n_mapped": 0}
    mapped = set(sectors["stock_code"].astype(str).str.upper())
    orphans = [c for c in investable if c not in mapped]
    cov = 1.0 - len(orphans) / max(len(investable), 1)
    return {
        "ok": cov >= 0.9,
        "coverage": float(cov),
        "n_mapped": int(len(investable) - len(orphans)),
        "orphans": orphans,
    }


def run_price_qa(
    prices: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
    macro_source: str = "unknown",
) -> dict:
    processed = settings.data_processed_dir
    if prices is None:
        prices = pd.read_parquet(processed / "prices.parquet")
    if macro is None and (processed / "macro.parquet").exists():
        macro = pd.read_parquet(processed / "macro.parquet")
    if sectors is None and (processed / "sectors.parquet").exists():
        sectors = pd.read_parquet(processed / "sectors.parquet")

    investable = _load_investable(prices)
    empty = pd.DataFrame()
    report = {
        "investable": investable,
        "prices": {
            "n_rows": int(len(prices)),
            "min_date": str(pd.to_datetime(prices["trade_date"]).min().date()),
            "max_date": str(pd.to_datetime(prices["trade_date"]).max().date()),
            "n_tickers": int(prices["stock_code"].nunique()),
        },
        "duplicates": check_duplicates(prices),
        "calendar_gaps": check_calendar_gaps(prices, investable),
        "zero_volume": check_zero_volume(prices, investable),
        "jumps": check_jumps(prices, investable),
        "ticker_hygiene": check_ticker_hygiene(prices),
        "macro": validate_macro(macro if macro is not None else empty, macro_source),
        "macro_asof": check_macro_asof_leak(prices, macro if macro is not None else empty),
        "sectors": check_sector_coverage(sectors if sectors is not None else empty, investable),
    }
    if str(macro_source).startswith("synthetic"):
        report["macro"]["banner"] = "SYNTHETIC MACRO — not official CBK/KNBS"
    report["ok"] = all(report[k].get("ok", True) for k in ("duplicates", "macro_asof"))
    return report


def write_qa_reports(report: dict, reports_dir: Path | None = None) -> tuple[Path, Path]:
    reports_dir = Path(reports_dir or (ROOT / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "data_qa.json"
    md_path = reports_dir / "data_qa.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    banner = report.get("macro", {}).get("banner")
    lines = ["# Data QA report", ""]
    if banner:
        lines += [f"> **{banner}**", ""]
    lines += [
        f"Prices: {report['prices']['n_rows']} rows, "
        f"{report['prices']['min_date']} → {report['prices']['max_date']}, "
        f"{report['prices']['n_tickers']} tickers.",
        "",
        f"Investable focus ({len(report['investable'])}): "
        + ", ".join(report["investable"]),
        "",
        "## Duplicates",
        f"- duplicate_rows: **{report['duplicates']['duplicate_rows']}** "
        f"(ok={report['duplicates']['ok']})",
        "",
        "## Calendar gaps",
        f"- gap events: **{report['calendar_gaps']['n_gap_events']}**",
        "",
        "## Zero volume / illiquidity",
        f"- flagged names: **{len(report['zero_volume']['flagged'])}**",
    ]
    for r in report["zero_volume"]["flagged"][:15]:
        lines.append(
            f"  - {r['stock_code']}: zero_vol={r['zero_volume_frac']:.0%}, "
            f"max_dst={r['max_days_since_trade']}"
        )
    lines += [
        "",
        "## Large jumps",
        f"- |ret_1d| > {report['jumps']['threshold']:.0%}: "
        f"**{report['jumps']['n_jumps']}** (showing {len(report['jumps']['examples'])})",
        f"- adj vs close disagreements on investable (>5%): "
        f"**{report['jumps']['adj_vs_close_disagreements']}**",
        "",
        "## Ticker hygiene",
        f"- rights issues: {report['ticker_hygiene']['n_rights']}",
        f"- indices: {', '.join(report['ticker_hygiene']['indices'])}",
        "",
        "## Known remaps (manual review)",
    ]
    for item in report["ticker_hygiene"]["known_remaps"]:
        lines.append(
            f"- {item['from']} → {item['to']}: {item['note']} "
            f"(from={item['from_present']}, to={item['to_present']})"
        )
    lines += [
        "",
        "## Macro",
        f"- source: **{report['macro'].get('source')}** ok={report['macro'].get('ok')}",
        f"- asof leak check: leaks={report['macro_asof'].get('leaks')} "
        f"(checked={report['macro_asof'].get('checked')})",
        "",
        "## Sectors",
        f"- coverage of investable: **{report['sectors'].get('coverage', 0):.0%}**",
        f"- orphans: {', '.join(report['sectors'].get('orphans') or []) or 'none'}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
