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
