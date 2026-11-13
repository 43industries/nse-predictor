"""Shared lot rounding, conviction scale, and paper ticket size hints."""

from __future__ import annotations

from nse_predictor.config import settings


def round_lots(shares: float, lot: int) -> int:
    if lot <= 0:
        return max(0, int(shares))
    return max(0, int(shares // lot) * lot)


def conviction_scale(prob_up: float, thr: float) -> float:
    """Soft scale on target weight: clip((prob - thr) / 0.15, 0.5, 1.25)."""
    raw = (float(prob_up) - float(thr)) / 0.15
    return float(max(0.5, min(1.25, raw)))


def suggest_size(
    *,
    prob: float,
    thr: float,
    price: float | None,
    held_shares: int,
    weight: float,
    nav: float,
    cash: float,
    n_names: int,
) -> tuple[str | None, int, str | None]:
    """Return (suggested_side, suggested_shares, note) for paper tickets."""
    lot = int(settings.portfolio_lot_size)
    max_w = float(settings.portfolio_max_weight_pct)
    max_names = int(settings.portfolio_max_names)
    min_cash = float(settings.portfolio_min_cash_pct)
    exit_prob = float(settings.portfolio_exit_prob_up)

    if price is None or price <= 0 or nav <= 0:
        return None, 0, "missing_price"

    if held_shares > 0 and prob <= exit_prob:
        sell = (
            held_shares
            if held_shares < lot
            else min(held_shares, round_lots(held_shares, lot) or held_shares)
        )
        return "sell", int(sell), "trim_or_exit"

    if held_shares > 0:
        conv = conviction_scale(prob, thr)
        target_w = max_w * conv
        if weight >= target_w - 1e-6:
            return None, 0, "hold_at_target"
        room = max(0.0, target_w * nav - weight * nav)
        deployable = max(0.0, cash - min_cash * nav)
        room = min(room, deployable)
        shares = round_lots(room / price, lot)
        if shares <= 0:
            return None, 0, "hold_no_room"
        return "buy", int(shares), "add"

    if n_names >= max_names:
        return None, 0, "max_names"
    conv = conviction_scale(prob, thr)
    target_w = max_w * conv
    deployable = max(0.0, cash - min_cash * nav)
    room = min(target_w * nav, deployable)
    shares = round_lots(room / price, lot)
    if shares <= 0:
        return None, 0, "no_cash_or_lot"
    return "buy", int(shares), "open"
