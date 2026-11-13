"""Full forecast board enriched with hit rate, book status, and ticket hints."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nse_predictor.companion.thresholds import model_threshold
from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history
from nse_predictor.portfolio.sizing import suggest_size
from nse_predictor.portfolio.store import get_book_mtm


def build_market_board(
    forecasts: pd.DataFrame,
    *,
    evaluation: dict | None = None,
    error_hist: pd.DataFrame | None = None,
    model_meta: dict | None = None,
) -> dict[str, Any]:
    disclaimer = settings.disclaimer
    empty = {"as_of": None, "rows": [], "n": 0, "disclaimer": disclaimer}
    if forecasts is None or forecasts.empty:
        return empty

    fc = forecasts.copy()
    fc["stock_code"] = fc["stock_code"].astype(str).str.strip().str.upper()
    fc = fc[~fc["stock_code"].str.startswith("^")].copy()
    if fc.empty:
        return empty

    thr = model_threshold(evaluation, model_meta)
    book = get_book_mtm()
    pos_map = {str(p["stock_code"]).upper(): p for p in (book.get("positions") or [])}
    nav = float(book.get("nav_kes") or 0.0)
    cash = float(book.get("cash_kes") or 0.0)
    n_names = int(book.get("n_names") or 0)
    max_w = float(settings.portfolio_max_weight_pct)

    as_of = pd.to_datetime(fc["trade_date"]).max()
    as_of_s = str(as_of.date()) if pd.notna(as_of) else None

    hits: dict[str, float | None] = {}
    for code in fc["stock_code"].tolist():
        if error_hist is not None and not error_hist.empty:
            hits[code] = summarize_error_history(error_hist, code, lookback_n=60).get("hit_rate")
        else:
            hits[code] = None

    rows: list[dict[str, Any]] = []
    for _, r in fc.iterrows():
        code = str(r["stock_code"])
        prob = float(r["prob_up"]) if pd.notna(r.get("prob_up")) else None
        label = str(r.get("pred_label") or r.get("direction_5d") or "").lower() or None
        price = float(r["price"]) if pd.notna(r.get("price")) else None
        lo = r.get("ret5d_lo", r.get("ret5d_xs_lo"))
        hi = r.get("ret5d_hi", r.get("ret5d_xs_hi"))
        held = pos_map.get(code)
        held_shares = int(held["shares"]) if held else 0
        weight = float(held["weight"] or 0.0) if held and held.get("weight") is not None else 0.0
        if held_shares <= 0:
            book_status = "flat"
        elif weight > max_w + 1e-9:
            book_status = "overweight"
        else:
            book_status = "held"

        side, shares, note = suggest_size(
            prob=prob if prob is not None else 0.5,
            thr=thr,
            price=price,
            held_shares=held_shares,
            weight=weight,
            nav=nav,
            cash=cash,
            n_names=n_names,
        )
        hr = hits.get(code)
        rows.append(
            {
                "stock_code": code,
                "as_of_date": as_of_s,
                "price": price,
                "direction_5d": label,
                "pred_label": label,
                "prob_up": prob,
                "ret5d_point": float(r["ret5d"]) if pd.notna(r.get("ret5d")) else None,
                "ret5d_low": float(lo) if lo is not None and lo == lo else None,
                "ret5d_high": float(hi) if hi is not None and hi == hi else None,
                "vol20d": float(r["vol20d"]) if pd.notna(r.get("vol20d")) else None,
                "hit_rate": float(hr) if hr is not None and hr == hr else None,
                "held_shares": held_shares,
                "weight": weight if held_shares else None,
                "book_status": book_status,
                "suggested_side": side,
                "suggested_shares": shares,
                "suggest_note": note,
                "disclaimer": disclaimer,
            }
        )

    rows.sort(key=lambda x: float(x.get("prob_up") or 0.0), reverse=True)
    return {
        "as_of": as_of_s,
        "rows": rows,
        "n": len(rows),
        "threshold": thr,
        "disclaimer": disclaimer,
    }
