"""Evaluation metrics vs baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_up": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_up": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n": int(len(y_true)),
        "base_rate_up": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if y_prob is not None:
        out["mean_prob_up"] = float(np.mean(y_prob))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(len(y_true)),
    }


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(inside)),
        "mean_width": float(np.mean(np.asarray(upper) - np.asarray(lower))),
        "n": int(len(y_true)),
    }


def compare_to_always_up(metrics: dict) -> dict:
    always_up_acc = metrics.get("base_rate_up", 0.5)
    return {
        **metrics,
        "always_up_accuracy": always_up_acc,
        "beats_always_up": metrics["accuracy"] > always_up_acc,
        "lift_vs_always_up": metrics["accuracy"] - always_up_acc,
    }


def majority_baseline_pred(y_true: np.ndarray) -> np.ndarray:
    """Predict the majority class for every row."""
    y_true = np.asarray(y_true).astype(int)
    if len(y_true) == 0:
        return y_true
    maj = 1 if float(y_true.mean()) >= 0.5 else 0
    return np.full(len(y_true), maj, dtype=int)


def abstain_rate(pred_label: np.ndarray | pd.Series) -> float:
    """Fraction of predictions labeled flat."""
    s = pd.Series(pred_label).astype(str).str.lower()
    if len(s) == 0:
        return float("nan")
    return float((s == "flat").mean())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE over equal-width probability bins (same grid as calibration_table)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def rank_ic(
    panel: pd.DataFrame,
    pred_col: str,
    target_col: str = "y_ret_5d_xs",
    date_col: str = "trade_date",
    min_names: int = 5,
) -> dict:
    """Mean daily Spearman rank correlation between prediction and target."""
    if panel.empty or pred_col not in panel.columns or target_col not in panel.columns:
        return {"mean_ic": float("nan"), "n_days": 0, "median_ic": float("nan")}
    d = panel[[date_col, pred_col, target_col]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col])
    ics: list[float] = []
    for _, g in d.groupby(date_col):
        if len(g) < min_names:
            continue
        if g[pred_col].nunique() < 2 or g[target_col].nunique() < 2:
            continue
        ic = g[pred_col].corr(g[target_col], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    if not ics:
        return {"mean_ic": float("nan"), "n_days": 0, "median_ic": float("nan")}
    arr = np.asarray(ics, dtype=float)
    return {
        "mean_ic": float(np.mean(arr)),
        "median_ic": float(np.median(arr)),
        "n_days": int(len(arr)),
        "ic_std": float(np.std(arr)),
    }
