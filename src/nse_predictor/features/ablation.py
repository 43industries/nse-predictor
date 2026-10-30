"""Validation-only feature-set ablation (no test peeking)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.eval.metrics import rank_ic
from nse_predictor.features.registry import ABLATION_PRESETS, MACRO_OFFICIAL, model_feature_cols
from nse_predictor.models.tabular import DirectionModel
from nse_predictor.training.fit import direction_metrics_bundle


def select_feature_preset(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    macro_source: str,
    fund_included: bool,
) -> tuple[str, list[str], dict]:
    """Pick preset by valid accuracy, then Rank IC."""
    best_name = "full_gated"
    best_cols = model_feature_cols(macro_source=macro_source, fund_included=fund_included)
    best_score = (-1.0, -1.0)
    rows: list[dict] = []

    if valid.empty or int(valid["y_dir_5d"].notna().sum()) < 200:
        return best_name, best_cols, {"skipped": True, "reason": "valid too small", "presets": []}

    for name, groups in ABLATION_PRESETS.items():
        cols = [
            c
            for c in model_feature_cols(
                macro_source=macro_source, fund_included=fund_included, groups=groups
            )
            if c in train.columns
        ]
        if len(cols) < 5:
            continue
        probe = DirectionModel(feature_cols=cols).fit(train, balance_classes=True)
        metrics, pred = direction_metrics_bundle(valid, probe)
        ric: dict = {"mean_ic": float("nan")}
        if "y_ret_5d_xs" in valid.columns:
            merged = valid.merge(
                pred[["trade_date", "stock_code", "prob_up"]],
                on=["trade_date", "stock_code"],
                how="inner",
            )
            ric = rank_ic(merged, "prob_up", "y_ret_5d_xs")
        acc = float(metrics.get("accuracy") or 0.0)
        ric_m = ric.get("mean_ic")
        ric_m = float(ric_m) if ric_m is not None and not (isinstance(ric_m, float) and np.isnan(ric_m)) else -1.0
        rows.append(
            {
                "preset": name,
                "n_features": len(cols),
                "features": cols,
                "valid_accuracy": acc,
                "valid_rank_ic": ric,
            }
        )
        score = (acc, ric_m)
        if score > best_score:
            best_score = score
            best_name = name
            best_cols = cols

    return best_name, best_cols, {"skipped": False, "presets": rows, "selected": best_name}


def write_ablation_report(
    path: Path,
    selected: str,
    feature_cols: list[str],
    macro_source: str,
    fund_included: bool,
    ablation: dict,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected": selected,
        "feature_cols": feature_cols,
        "macro_source": macro_source,
        "fundamentals_included": fund_included,
        "macro_excluded": [c for c in MACRO_OFFICIAL if c not in feature_cols],
        **ablation,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
