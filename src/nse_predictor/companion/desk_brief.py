"""Rank today's desk setups and skip list for the trading companion."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history


def _at_chance_set(evaluation: dict | None) -> set[str]:
    evaluation = evaluation or {}
    diag = evaluation.get("diagnostics") or {}
    names = diag.get("at_chance_stocks") or []
    if not names:
        names = [
            str(r.get("stock_code") or "").upper()
            for r in (diag.get("per_stock") or [])
            if r.get("flag") in ("at_chance", "thin_n")
        ]
    return {str(c).upper() for c in names if c}


def _flag_map(evaluation: dict | None) -> dict[str, str]:
    evaluation = evaluation or {}
    diag = evaluation.get("diagnostics") or {}
    out: dict[str, str] = {}
    for r in diag.get("per_stock") or []:
        code = str(r.get("stock_code") or "").upper()
        if code and r.get("flag"):
            out[code] = str(r["flag"])
    for c in diag.get("at_chance_stocks") or []:
        out.setdefault(str(c).upper(), "at_chance")
    return out


def _model_threshold(evaluation: dict | None, model_meta: dict | None) -> float:
    meta = model_meta or {}
    if meta.get("threshold") is not None:
        try:
            return float(meta["threshold"])
        except (TypeError, ValueError):
            pass
    sel = (evaluation or {}).get("selection") or {}
    try:
        return float(sel.get("threshold") or 0.5)
    except (TypeError, ValueError):
        return 0.5


def build_desk_brief(
    forecasts: pd.DataFrame,
    evaluation: dict | None = None,
    error_hist: pd.DataFrame | None = None,
    model_meta: dict | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build today's setups (actionable) and skip list.

    Setups: pred_label up (or prob_up >= threshold) and not at-chance/thin_n.
    Skips: at-chance / thin_n, or flat / below-threshold names.
    """
    disclaimer = settings.disclaimer
    empty = {
        "as_of": None,
        "n_setups": 0,
        "n_skips": 0,
        "threshold": _model_threshold(evaluation, model_meta),
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

    thr = _model_threshold(evaluation, model_meta)
    chance = _at_chance_set(evaluation)
    flags = _flag_map(evaluation)
    as_of = pd.to_datetime(fc["trade_date"]).max()
    as_of_s = str(as_of.date()) if pd.notna(as_of) else None

    hits: dict[str, float | None] = {}
    codes = fc["stock_code"].tolist()
    for code in codes:
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
        rows.append(
            {
                "stock_code": code,
                "trade_date": as_of_s,
                "price": float(r["price"]) if pd.notna(r.get("price")) else None,
                "pred_label": label or None,
                "prob_up": prob if prob == prob else None,
                "ret5d_lo": float(lo) if lo == lo and lo is not None else None,
                "ret5d_hi": float(hi) if hi == hi and hi is not None else None,
                "vol20d": float(r["vol20d"]) if pd.notna(r.get("vol20d")) else None,
                "hit_rate": float(hr) if hr is not None and hr == hr else None,
                "flag": flag or None,
                "signal_strength": abs(prob - 0.5) if prob == prob else 0.0,
            }
        )

    setups: list[dict] = []
    skips: list[dict] = []
    for row in rows:
        code = row["stock_code"]
        label = (row.get("pred_label") or "").lower()
        prob = row.get("prob_up")
        flagged = code in chance or row.get("flag") in ("at_chance", "thin_n")
        is_up = label == "up" or (prob is not None and prob >= thr)
        if flagged:
            skips.append({**row, "skip_reason": row.get("flag") or "at_chance"})
        elif not is_up or label == "flat":
            reason = "flat" if label == "flat" else "low_conviction"
            skips.append({**row, "skip_reason": reason})
        else:
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
        "setups": setups,
        "skips": skips,
        "disclaimer": disclaimer,
    }
