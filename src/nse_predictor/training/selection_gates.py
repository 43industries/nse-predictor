"""Ship gates for feature-preset and direction-head selection (valid only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def label_counts(pred_label: pd.Series | np.ndarray) -> dict[str, int]:
    s = pd.Series(pred_label).astype(str).str.lower()
    return {
        "n_up": int((s == "up").sum()),
        "n_down": int((s == "down").sum()),
        "n_flat": int((s == "flat").sum()),
        "n_scored": int(((s == "up") | (s == "down")).sum()),
    }


def is_degenerate_labels(pred_label: pd.Series | np.ndarray) -> bool:
    c = label_counts(pred_label)
    return c["n_up"] == 0 or c["n_down"] == 0


def gate_direction_candidate(
    *,
    valid_metrics: dict[str, Any] | None,
    pred_label: pd.Series | np.ndarray | None,
    valid_utility: float | None = None,
    require_utility: bool = False,
) -> tuple[bool, str | None]:
    """Return (passes, rejected_reason)."""
    metrics = valid_metrics or {}
    if pred_label is not None and is_degenerate_labels(pred_label):
        c = label_counts(pred_label)
        return False, f"degenerate_labels up={c['n_up']} down={c['n_down']}"

    if metrics.get("beats_always_up") is False:
        return False, "does_not_beat_always_up"

    if require_utility and valid_utility is not None:
        try:
            u = float(valid_utility)
        except (TypeError, ValueError):
            u = float("nan")
        if u == u and u <= 0:
            return False, f"valid_utility={u:.4f}<=0"

    return True, None


def rank_ic_value(ric: Any) -> float:
    if isinstance(ric, dict):
        v = ric.get("mean_ic")
    else:
        v = ric
    try:
        out = float(v)
    except (TypeError, ValueError):
        return -1.0
    return out if out == out else -1.0
