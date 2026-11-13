"""Score-aware paper portfolio suggestions (human must confirm tickets)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nse_predictor.companion.thresholds import at_chance_set, model_threshold
from nse_predictor.config import settings
from nse_predictor.eval.error_history import summarize_error_history
from nse_predictor.portfolio.sizing import conviction_scale, round_lots
from nse_predictor.portfolio.store import get_book_mtm


def build_suggestions(
    forecasts: pd.DataFrame,
    *,
    evaluation: dict | None = None,
    model_meta: dict | None = None,
    error_hist: pd.DataFrame | None = None,
    min_prob_up: float | None = None,
    min_hit_rate: float | None = None,
    processed=None,
) -> dict[str, Any]:
    """Propose buy/add/hold/trim/exit/stay_flat actions vs the paper book.

    Never places orders — caller creates tickets and a human confirms fills.
    """
    book = get_book_mtm(processed)
    lot = int(book.get("lot_size") or settings.portfolio_lot_size)
    max_names = int(settings.portfolio_max_names)
    max_w = float(settings.portfolio_max_weight_pct)
    min_cash = float(settings.portfolio_min_cash_pct)
    min_prob = float(min_prob_up if min_prob_up is not None else settings.portfolio_min_prob_up)
    min_hit = float(min_hit_rate if min_hit_rate is not None else 0.52)
    exit_prob = float(settings.portfolio_exit_prob_up)
    thr = model_threshold(evaluation, model_meta)
    chance = at_chance_set(evaluation)

    empty = {
        "as_of": None,
        "nav_kes": book.get("nav_kes"),
        "cash_kes": book.get("cash_kes"),
        "suggestions": [],
        "n_suggestions": 0,
        "risk": {
            "max_names": max_names,
            "max_weight_pct": max_w,
            "min_cash_pct": min_cash,
            "min_prob_up": min_prob,
            "min_hit_rate": min_hit,
            "exit_prob_up": exit_prob,
            "threshold": thr,
            "lot_size": lot,
        },
        "filters": {
            "min_prob_up": min_prob,
            "min_hit_rate": min_hit,
            "threshold": thr,
        },
        "disclaimer": settings.disclaimer,
        "mode": "paper",
    }
    if forecasts is None or forecasts.empty:
        return empty

    fc = forecasts.copy()
    fc["stock_code"] = fc["stock_code"].astype(str).str.strip().str.upper()
    fc = fc[~fc["stock_code"].str.startswith("^")].copy()
    if fc.empty:
        return empty

    as_of = pd.to_datetime(fc["trade_date"]).max()
    as_of_s = str(as_of.date()) if pd.notna(as_of) else None

    pos_map = {str(p["stock_code"]).upper(): p for p in book.get("positions") or []}
    nav = float(book.get("nav_kes") or 0.0)
    cash = float(book.get("cash_kes") or 0.0)
    n_names = int(book.get("n_names") or 0)
    deployable = max(0.0, cash - min_cash * nav) if nav > 0 else cash

    hit_rates: dict[str, float | None] = {}
    for code in fc["stock_code"].tolist():
        if error_hist is not None and not error_hist.empty:
            hit_rates[code] = summarize_error_history(error_hist, code, lookback_n=60).get(
                "hit_rate"
            )
        else:
            hit_rates[code] = None

    suggestions: list[dict[str, Any]] = []

    for _, r in fc.iterrows():
        code = str(r["stock_code"])
        prob = float(r["prob_up"]) if pd.notna(r.get("prob_up")) else 0.5
        label = str(r.get("pred_label") or "").lower()
        price = float(r["price"]) if pd.notna(r.get("price")) else None
        hit = hit_rates.get(code)
        held = pos_map.get(code)
        held_shares = int(held["shares"]) if held else 0
        weight = float(held["weight"] or 0.0) if held and held.get("weight") is not None else 0.0

        if code in chance:
            suggestions.append(
                {
                    "stock_code": code,
                    "action": "stay_flat" if held_shares == 0 else "hold",
                    "side": None,
                    "shares": 0,
                    "prob_up": prob,
                    "pred_label": label,
                    "price": price,
                    "hit_rate": hit,
                    "reason": "at_chance_or_thin — suppress new risk",
                    "priority": 0,
                }
            )
            continue

        if price is None or price <= 0:
            suggestions.append(
                {
                    "stock_code": code,
                    "action": "hold" if held_shares else "stay_flat",
                    "side": None,
                    "shares": 0,
                    "prob_up": prob,
                    "pred_label": label,
                    "price": price,
                    "hit_rate": hit,
                    "reason": "missing_price",
                    "priority": 0,
                }
            )
            continue

        bullish = (label == "up") or (prob >= max(thr, min_prob))
        bearish = (label == "down") or (prob <= exit_prob)

        if held_shares > 0 and bearish:
            sell = round_lots(held_shares, lot) or held_shares
            sell = min(sell, held_shares)
            if held_shares < lot:
                sell = held_shares
            action = "exit" if sell >= held_shares else "trim"
            suggestions.append(
                {
                    "stock_code": code,
                    "action": action,
                    "side": "sell",
                    "shares": int(sell),
                    "prob_up": prob,
                    "pred_label": label,
                    "price": price,
                    "hit_rate": hit,
                    "reason": f"weak_signal prob_up={prob:.2f} <= {exit_prob:.2f}",
                    "priority": int(100 * (exit_prob - prob + 0.5)),
                }
            )
            continue

        if held_shares > 0 and not bullish:
            suggestions.append(
                {
                    "stock_code": code,
                    "action": "hold",
                    "side": None,
                    "shares": 0,
                    "prob_up": prob,
                    "pred_label": label,
                    "price": price,
                    "hit_rate": hit,
                    "reason": "held_neutral",
                    "priority": 1,
                }
            )
            continue

        if bullish:
            if held_shares == 0 and hit is not None and float(hit) < min_hit:
                suggestions.append(
                    {
                        "stock_code": code,
                        "action": "stay_flat",
                        "side": None,
                        "shares": 0,
                        "prob_up": prob,
                        "pred_label": label,
                        "price": price,
                        "hit_rate": hit,
                        "reason": f"hit_rate {hit:.0%} < min_hit_rate {min_hit:.0%}",
                        "priority": 0,
                    }
                )
                continue
            if held_shares == 0 and prob < min_prob:
                suggestions.append(
                    {
                        "stock_code": code,
                        "action": "stay_flat",
                        "side": None,
                        "shares": 0,
                        "prob_up": prob,
                        "pred_label": label,
                        "price": price,
                        "hit_rate": hit,
                        "reason": f"prob_up {prob:.2f} < min_prob_up {min_prob:.2f}",
                        "priority": 0,
                    }
                )
                continue

            conv = conviction_scale(prob, thr)
            target_w = max_w * conv
            target_value = target_w * nav if nav > 0 else target_w * (cash + 1)
            current_value = weight * nav if nav > 0 else held_shares * price
            room_value = max(0.0, target_value - current_value)
            room_value = min(room_value, deployable)
            if held_shares == 0 and n_names >= max_names:
                suggestions.append(
                    {
                        "stock_code": code,
                        "action": "stay_flat",
                        "side": None,
                        "shares": 0,
                        "prob_up": prob,
                        "pred_label": label,
                        "price": price,
                        "hit_rate": hit,
                        "reason": f"max_names={max_names}",
                        "priority": 0,
                    }
                )
                continue
            buy_shares = round_lots(room_value / price, lot)
            if buy_shares <= 0:
                suggestions.append(
                    {
                        "stock_code": code,
                        "action": "hold" if held_shares else "stay_flat",
                        "side": None,
                        "shares": 0,
                        "prob_up": prob,
                        "pred_label": label,
                        "price": price,
                        "hit_rate": hit,
                        "reason": "no_lot_capacity_or_cash",
                        "priority": 0,
                    }
                )
                continue
            cost = buy_shares * price * (1 + float(book.get("cost_bps") or 0) / 10000.0)
            if cost > deployable + 1e-6:
                buy_shares = round_lots(deployable / (price * 1.01), lot)
                cost = buy_shares * price * (1 + float(book.get("cost_bps") or 0) / 10000.0)
            if buy_shares <= 0:
                suggestions.append(
                    {
                        "stock_code": code,
                        "action": "hold" if held_shares else "stay_flat",
                        "side": None,
                        "shares": 0,
                        "prob_up": prob,
                        "pred_label": label,
                        "price": price,
                        "hit_rate": hit,
                        "reason": "insufficient_cash_buffer",
                        "priority": 0,
                    }
                )
                continue
            deployable = max(0.0, deployable - cost)
            if held_shares == 0:
                n_names += 1
            action = "add" if held_shares else "open"
            suggestions.append(
                {
                    "stock_code": code,
                    "action": action,
                    "side": "buy",
                    "shares": int(buy_shares),
                    "prob_up": prob,
                    "pred_label": label,
                    "price": price,
                    "hit_rate": hit,
                    "conviction": round(conv, 3),
                    "est_notional_kes": round(buy_shares * price, 2),
                    "reason": (
                        f"bullish prob_up={prob:.2f} target_w<={target_w:.0%} "
                        f"conviction={conv:.2f}"
                    ),
                    "priority": int(100 * prob),
                }
            )
            continue

        suggestions.append(
            {
                "stock_code": code,
                "action": "stay_flat",
                "side": None,
                "shares": 0,
                "prob_up": prob,
                "pred_label": label,
                "price": price,
                "hit_rate": hit,
                "reason": "below_entry_threshold",
                "priority": 0,
            }
        )

    actionable = {"open", "add", "trim", "exit"}
    suggestions.sort(
        key=lambda s: (
            0 if s["action"] in actionable else 1,
            -int(s.get("priority") or 0),
            s["stock_code"],
        )
    )

    return {
        "as_of": as_of_s,
        "nav_kes": book.get("nav_kes"),
        "cash_kes": book.get("cash_kes"),
        "suggestions": suggestions,
        "n_suggestions": sum(1 for s in suggestions if s["action"] in actionable),
        "risk": empty["risk"],
        "filters": empty["filters"],
        "disclaimer": settings.disclaimer,
        "mode": "paper",
    }
