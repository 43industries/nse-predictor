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
from nse_predictor.training.selection_gates import gate_direction_candidate, rank_ic_value


def select_feature_preset(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    macro_source: str,
    fund_included: bool,
) -> tuple[str, list[str], dict]:
    """Pick preset by valid Rank IC, then accuracy, subject to ship gates."""
    best_name = "core_relative"
    best_cols = model_feature_cols(
        macro_source=macro_source, fund_included=fund_included, groups=ABLATION_PRESETS.get("core_relative")
    )
    if not best_cols:
        best_cols = model_feature_cols(macro_source=macro_source, fund_included=fund_included)
    rows: list[dict] = []

    if valid.empty or int(valid["y_dir_5d"].notna().sum()) < 200:
        return best_name, best_cols, {
            "skipped": True,
            "reason": "valid too small",
            "selection_metric": "valid_rank_ic",
            "selection_gates": ["non_degenerate_labels", "beats_always_up"],
            "presets": [],
        }

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
        # Light calibrate + default threshold for gate checks
        try:
            probe.calibrate(valid)
        except Exception:  # noqa: BLE001
            pass
        probe.set_threshold(0.55)
        probe.set_decision_mode("threshold")
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
        ric_m = rank_ic_value(ric)
        passes, rejected = gate_direction_candidate(
            valid_metrics=metrics,
            pred_label=pred["pred_label"],
            require_utility=False,
        )
        rows.append(
            {
                "preset": name,
                "n_features": len(cols),
                "features": cols,
                "valid_accuracy": acc,
                "valid_rank_ic": ric,
                "beats_always_up": bool(metrics.get("beats_always_up")),
                "label_counts": {
                    "n_up": int((pred["pred_label"].astype(str).str.lower() == "up").sum()),
                    "n_down": int((pred["pred_label"].astype(str).str.lower() == "down").sum()),
                    "n_flat": int((pred["pred_label"].astype(str).str.lower() == "flat").sum()),
                },
                "passes_gates": passes,
                "rejected_reason": rejected,
            }
        )

    # Prefer gated winners by Rank IC then accuracy; else best Rank IC overall with reason.
    gated = [r for r in rows if r.get("passes_gates")]
    pool = gated if gated else rows
    pool_sorted = sorted(
        pool,
        key=lambda r: (rank_ic_value(r.get("valid_rank_ic")), float(r.get("valid_accuracy") or 0.0)),
        reverse=True,
    )
    if pool_sorted:
        winner = pool_sorted[0]
        best_name = str(winner["preset"])
        best_cols = list(winner["features"])
        if not gated:
            for r in rows:
                if r["preset"] == best_name and not r.get("rejected_reason"):
                    r["rejected_reason"] = "fallback_no_gated_winner"
                    r["passes_gates"] = False

    return best_name, best_cols, {
        "skipped": False,
        "selection_metric": "valid_rank_ic",
        "selection_gates": ["non_degenerate_labels", "beats_always_up"],
        "n_gated_winners": len(gated),
        "presets": rows,
        "selected": best_name,
        "selected_passes_gates": best_name in {r["preset"] for r in gated},
    }


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
