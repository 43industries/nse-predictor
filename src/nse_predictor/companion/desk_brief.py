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

SKIP_LABELS = {
    "at_chance": "Model unreliable on this name",
    "thin_n": "Too little history to trust",
    "flat": "Sitting out",
    "low_conviction": "Below confidence bar",
    "low_hit_rate": "Recent accuracy too low",
}


def skip_label(reason: str | None) -> str:
    key = str(reason or "").strip().lower()
    return SKIP_LABELS.get(key, "Skipped for today")


def _label_counts(rows: list[dict]) -> tuple[int, int, int]:
    n_up = n_down = n_flat = 0
    for row in rows:
        lab = str(row.get("pred_label") or "").lower()
        if lab == "up":
            n_up += 1
        elif lab == "down":
            n_down += 1
        elif lab == "flat":
            n_flat += 1
    return n_up, n_down, n_flat


def _headline(n_setups: int, n_up: int, n_down: int, n_flat: int) -> str:
    if n_setups <= 0:
        return f"No buy setups. {n_flat} flat, {n_down} down."
    if n_setups == 1:
        return f"1 setup above the confidence bar ({n_up} up · {n_flat} flat · {n_down} down)."
    return f"{n_setups} setups above the confidence bar ({n_up} up · {n_flat} flat · {n_down} down)."


def _watchlist(rows: list[dict], setup_codes: set[str], n: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda r: float(r["prob_up"]) if r.get("prob_up") is not None else -1.0,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in ranked[:n]:
        reason = row.get("skip_reason")
        out.append(
            {
                "stock_code": row["stock_code"],
                "prob_up": row.get("prob_up"),
                "pred_label": row.get("pred_label"),
                "skip_reason": reason,
                "skip_label": skip_label(reason) if reason else None,
                "in_setups": row["stock_code"] in setup_codes,
            }
        )
    return out


def _summary(
    *,
    as_of: str | None,
    setups: list[dict],
    skips: list[dict],
    rows: list[dict],
    entry_thr: float,
    min_prob: float,
    min_hit: float,
    n_at_chance: int,
) -> dict[str, Any]:
    n_up, n_down, n_flat = _label_counts(rows)
    n_names = len(rows)
    caution = n_names > 0 and (n_at_chance / n_names) >= 0.40
    return {
        "as_of": as_of,
        "n_setups": len(setups),
        "n_skips": len(skips),
        "n_up": n_up,
        "n_down": n_down,
        "n_flat": n_flat,
        "entry_threshold": entry_thr,
        "min_prob_up": min_prob,
        "min_hit_rate": min_hit,
        "n_at_chance": n_at_chance,
        "model_caution": caution,
        "headline": _headline(len(setups), n_up, n_down, n_flat),
    }


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

    empty_summary = _summary(
        as_of=None,
        setups=[],
        skips=[],
        rows=[],
        entry_thr=entry_thr,
        min_prob=min_prob,
        min_hit=min_hit,
        n_at_chance=0,
    )
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
        "summary": empty_summary,
        "watchlist": [],
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
            reason = row.get("flag") or "at_chance"
            skips.append({**row, "skip_reason": reason, "skip_label": skip_label(reason)})
            continue
        if label == "flat":
            skips.append({**row, "skip_reason": "flat", "skip_label": skip_label("flat")})
            continue
        if prob is None or prob < entry_thr:
            skips.append(
                {**row, "skip_reason": "low_conviction", "skip_label": skip_label("low_conviction")}
            )
            continue
        if held_shares == 0 and hr is not None and float(hr) < min_hit:
            skips.append(
                {**row, "skip_reason": "low_hit_rate", "skip_label": skip_label("low_hit_rate")}
            )
            continue
        if label != "up" and not (prob is not None and prob >= entry_thr):
            skips.append(
                {**row, "skip_reason": "low_conviction", "skip_label": skip_label("low_conviction")}
            )
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

    # Attach skip_reason on full row list for watchlist ranking context
    skip_by_code = {s["stock_code"]: s for s in skips}
    annotated = []
    for row in rows:
        if row["stock_code"] in skip_by_code:
            annotated.append(skip_by_code[row["stock_code"]])
        else:
            annotated.append(row)

    n_at_chance = sum(
        1
        for s in skips
        if str(s.get("skip_reason") or "") in ("at_chance", "thin_n")
        or s.get("flag") in ("at_chance", "thin_n")
    )
    setup_codes = {s["stock_code"] for s in setups}
    summary = _summary(
        as_of=as_of_s,
        setups=setups,
        skips=skips,
        rows=rows,
        entry_thr=entry_thr,
        min_prob=min_prob,
        min_hit=min_hit,
        n_at_chance=n_at_chance,
    )

    return {
        "as_of": as_of_s,
        "n_setups": len(setups),
        "n_skips": len(skips),
        "threshold": thr,
        "filters": empty["filters"],
        "setups": setups,
        "skips": skips,
        "summary": summary,
        "watchlist": _watchlist(annotated, setup_codes, n=3),
        "disclaimer": disclaimer,
    }
