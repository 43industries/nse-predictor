"""Shared model threshold and at-chance helpers for companion + portfolio brain."""

from __future__ import annotations


def at_chance_set(evaluation: dict | None) -> set[str]:
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


def flag_map(evaluation: dict | None) -> dict[str, str]:
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


def model_threshold(evaluation: dict | None, model_meta: dict | None) -> float:
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
