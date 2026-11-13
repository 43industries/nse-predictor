"""Rank today's desk setups and skip list for the trading companion."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nse_predictor.companion.thresholds import at_chance_set, flag_map, model_threshold
from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history
from nse_predictor.portfolio.sizing import suggest_size

# Back-compat aliases for older imports
_at_chance_set = at_chance_set
_flag_map = flag_map
_model_threshold = model_threshold
_suggest_size = suggest_size


def build_desk_brief(
    forecasts: pd.DataFrame,
    evaluation: dict | None = None,
    error_hist: pd.DataFrame | None = None,
    model_meta: dict | None = None,
    top_n: int = 10,
    min_prob_up: float | None = None,
    min_hit_rate: float | None = None,
    book: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build today's setups (actionable) and skip list.

    Setups: bullish, not at-chance, pass min_prob_up / min_hit_rate for new names.
    Annotated with book status and suggested paper ticket size.
    """
    disclaimer = settings.disclaimer
    thr = model_threshold(evaluation, model_meta)
    min_prob = float(min_prob_up if min_prob_up is not None else settings.portfolio_min_prob_up)
    min_hit = float(min_hit_rate if min_hit_rate is not None else 0.52)
    entry_thr = max(thr, min_prob)

    empty = {
        "as_of": None,
        "n_setups": 0,
        "n_skips": 0,
        "threshold": thr,
        "filters": {
            "min_prob_up": min_prob,
            "min_hit_rate": min_hit,
            "threshold": thr,
            "entry_threshold": entry_thr,
            "top_n": int(top_n),
        },
        "setups": [],
        "skips": [],
        "disclaimer": disclaimer,
    }
    if forecasts is None or forecasts.empty:
        return empty

    fc = forecasts.copy()
    fc["stock_code"] = fc["stock_code"].astype(str).str.strip().str.upper()
    fc = fc[~fc["stock_code"].str.startswith("^")].copy()
    if fc.empty:
        return empty

    chance = at_chance_set(evaluation)
    flags = flag_map(evaluation)
    as_of = pd.to_datetime(fc["trade_date"]).max()
    as_of_s = str(as_of.date()) if pd.notna(as_of) else None

    if book is not None:
        book_mtm = book
    else:
        from nse_predictor.portfolio.store import get_book_mtm

        book_mtm = get_book_mtm()
    pos_map = {str(p["stock_code"]).upper(): p for p in (book_mtm.get("positions") or [])}
    nav = float(book_mtm.get("nav_kes") or 0.0)
    cash = float(book_mtm.get("cash_kes") or 0.0)
    n_names = int(book_mtm.get("n_names") or 0)
    max_w = float(settings.portfolio_max_weight_pct)

    hits: dict[str, float | None] = {}
    for code in fc["stock_code"].tolist():
        if error_hist is not None and not error_hist.empty:
            hits[code] = summarize_error_history(error_hist, code, lookback_n=60).get("hit_rate")
        else:
            hits[code] = None

    rows: list[dict] = []
    for _, r in fc.iterrows():
        code = str(r["stock_code"])
        prob = float(r["prob_up"]) if pd.notna(r.get("prob_up")) else float("nan")
        label = str(r.get("pred_label") or "").lower()
        flag = flags.get(code, "")
        hr = hits.get(code)
        lo = r.get("ret5d_xs_lo", r.get("ret5d_lo"))
        hi = r.get("ret5d_xs_hi", r.get("ret5d_hi"))
        held = pos_map.get(code)
        held_shares = int(held["shares"]) if held else 0
        weight = float(held["weight"] or 0.0) if held and held.get("weight") is not None else 0.0
        if held_shares <= 0:
            book_status = "flat"
        elif weight > max_w + 1e-9:
            book_status = "overweight"
        else:
            book_status = "held"

        price = float(r["price"]) if pd.notna(r.get("price")) else None
        side, shares, size_note = suggest_size(
            prob=prob if prob == prob else 0.5,
            thr=thr,
            price=price,
            held_shares=held_shares,
            weight=weight,
            nav=nav,
            cash=cash,
            n_names=n_names,
        )

        rows.append(
            {
                "stock_code": code,
                "trade_date": as_of_s,
                "price": price,
                "pred_label": label or None,
                "prob_up": prob if prob == prob else None,
                "ret5d_lo": float(lo) if lo == lo and lo is not None else None,
                "ret5d_hi": float(hi) if hi == hi and hi is not None else None,
                "vol20d": float(r["vol20d"]) if pd.notna(r.get("vol20d")) else None,
                "hit_rate": float(hr) if hr is not None and hr == hr else None,
                "flag": flag or None,
                "signal_strength": abs(prob - 0.5) if prob == prob else 0.0,
                "held_shares": held_shares,
                "weight": weight if held_shares else None,
                "book_status": book_status,
                "suggested_side": side,
                "suggested_shares": shares,
                "suggest_note": size_note,
            }
        )

    setups: list[dict] = []
    skips: list[dict] = []
    for row in rows:
        code = row["stock_code"]
        label = (row.get("pred_label") or "").lower()
        prob = row.get("prob_up")
        hr = row.get("hit_rate")
        held_shares = int(row.get("held_shares") or 0)
        flagged = code in chance or row.get("flag") in ("at_chance", "thin_n")

        if flagged:
            skips.append({**row, "skip_reason": row.get("flag") or "at_chance"})
            continue
        if label == "flat":
            skips.append({**row, "skip_reason": "flat"})
            continue
        if prob is None or prob < entry_thr:
            skips.append({**row, "skip_reason": "low_conviction"})
            continue
        if held_shares == 0 and hr is not None and float(hr) < min_hit:
            skips.append({**row, "skip_reason": "low_hit_rate"})
            continue
        if label != "up" and not (prob is not None and prob >= entry_thr):
            skips.append({**row, "skip_reason": "low_conviction"})
            continue
        setups.append(row)

    setups.sort(
        key=lambda x: (
            float(x.get("signal_strength") or 0.0),
            float(x.get("hit_rate") or 0.0),
        ),
        reverse=True,
    )
    setups = setups[: int(top_n)]
    skips.sort(key=lambda x: str(x.get("stock_code") or ""))

    return {
        "as_of": as_of_s,
        "n_setups": len(setups),
        "n_skips": len(skips),
        "threshold": thr,
        "filters": empty["filters"],
        "setups": setups,
        "skips": skips,
        "disclaimer": disclaimer,
    }
