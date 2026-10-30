"""Validation pack helpers used by train_evaluate (stress, diagnostics, report)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.eval.diagnostics import (
    binary_from_labels,
    calibration_pack,
    confusion_counts,
)
from nse_predictor.eval.metrics import abstain_rate, coverage, rank_ic, regression_metrics
from nse_predictor.eval.protocol import LABEL_HORIZON_DAYS, apply_embargo
from nse_predictor.eval.walk_forward import stress_slice
from nse_predictor.models.tabular import DirectionModel
from nse_predictor.training.fit import direction_metrics_bundle

VALIDATION_REPORT_KEYS = (
    "protocol",
    "calibration",
    "walk_forward_summary",
    "stress_rank_ic",
    "validation_gates",
    "research_backtest",
)
STRESS_PERIODS = ("covid_2020", "fx_2023")


def is_stress_fragile(stress_ric: dict) -> bool:
    for v in stress_ric.values():
        if not isinstance(v, dict) or v.get("skipped"):
            continue
        mic = v.get("mean_ic")
        if mic is not None and mic == mic and mic < 0:
            return True
    return False


def stress_oos_retrain(
    panel: pd.DataFrame,
    feature_cols: list[str],
    dir_sel: dict,
    periods: tuple[str, ...] = STRESS_PERIODS,
    horizon_days: int = LABEL_HORIZON_DAYS,
) -> tuple[dict, dict]:
    """Retrain on pre-period data; return (stress_rank_ic, stress_accuracy)."""
    threshold = float(dir_sel.get("threshold") or 0.5)
    mode = str(dir_sel.get("decision_mode") or "threshold")
    params = dir_sel.get("best_params") or {}
    ric_out: dict = {}
    acc_out: dict = {}

    for period in periods:
        sl = stress_slice(panel, period).dropna(subset=["y_dir_5d"])
        if len(sl) < 50:
            skip = {"n": len(sl), "skipped": True}
            ric_out[period] = skip
            acc_out[period] = dict(skip)
            continue

        slice_start = pd.to_datetime(sl["trade_date"]).min()
        pre = panel[pd.to_datetime(panel["trade_date"]) < slice_start]
        pre = apply_embargo(pre, slice_start - pd.Timedelta(days=1), horizon_days=horizon_days)
        if len(pre) < 500:
            skip = {"n": len(sl), "skipped": True, "reason": "thin_pre"}
            ric_out[period] = skip
            acc_out[period] = dict(skip)
            continue

        m = DirectionModel(feature_cols=feature_cols, params=params).fit(
            pre, early_stopping_rounds=None
        )
        m.set_threshold(threshold)
        m.set_decision_mode(mode)
        metrics, pred = direction_metrics_bundle(sl, m)
        acc_out[period] = metrics

        if "y_ret_5d_xs" not in sl.columns:
            ric_out[period] = {"skipped": True, "reason": "no_y_ret_5d_xs"}
            continue
        merged = sl.merge(
            pred[["trade_date", "stock_code", "prob_up"]],
            on=["trade_date", "stock_code"],
            how="inner",
        )
        ric = rank_ic(merged, "prob_up", "y_ret_5d_xs")
        ric["abstain_rate"] = metrics.get("abstain_rate")
        ric_out[period] = ric

    return ric_out, acc_out


def direction_diagnostics(eval_df: pd.DataFrame, dir_pred: pd.DataFrame) -> dict:
    y = eval_df["y_dir_5d"].astype(int).values
    yp_f = binary_from_labels(dir_pred["pred_label"])
    scored = ~np.isnan(yp_f)
    out: dict = {"abstain_rate": abstain_rate(dir_pred["pred_label"])}
    if scored.sum() < 20:
        out["confusion"] = confusion_counts(
            y, (dir_pred["pred_label"] == "up").astype(int).values
        )
        return out

    yt = y[scored]
    pr = dir_pred.loc[scored, "prob_up"].values
    out["confusion"] = confusion_counts(yt, yp_f[scored].astype(int))
    pack = calibration_pack(yt, pr, pred_label=dir_pred["pred_label"])
    out["calibration_bins"] = pack.pop("calibration_bins")
    out["calibration"] = pack
    return out


def eval_return_model(eval_df: pd.DataFrame, ret_model) -> dict:
    tgt = ret_model.target_col
    ev = eval_df.dropna(subset=[tgt])
    if ev.empty:
        return {}
    rp = ret_model.predict(ev)
    ric = ev.merge(
        rp[["trade_date", "stock_code", "pred_value"]],
        on=["trade_date", "stock_code"],
        how="inner",
    )
    return {
        "lightgbm_return_5d_xs": regression_metrics(ev[tgt].values, rp["pred_value"].values),
        "return_interval_coverage": coverage(
            ev[tgt].values, rp["lower"].values, rp["upper"].values
        ),
        "rank_ic_return_model": rank_ic(ric, "pred_value", tgt),
    }


def write_validation_report(results: dict, path: Path) -> dict:
    payload = {
        "protocol": results.get("protocol") or {},
        "calibration": results.get("diagnostics", {}).get("calibration") or {},
        "walk_forward_summary": results.get("walk_forward_summary") or {},
        "stress_rank_ic": results.get("stress_rank_ic") or {},
        "stress_fragile": bool(results.get("stress_fragile")),
        "validation_gates": results.get("validation_gates") or {},
        "research_backtest": results.get("research_backtest") or {},
        "rank_ic": results.get("diagnostics", {}).get("rank_ic") or {},
        "abstain_rate": results.get("diagnostics", {}).get("abstain_rate"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    missing = [k for k in VALIDATION_REPORT_KEYS if k not in payload]
    if missing:
        raise RuntimeError(f"validation_report missing keys: {missing}")
    return payload


def map_stock_freq(
    frames: list[pd.DataFrame],
    freq_map: dict,
) -> list[pd.DataFrame]:
    """Attach stock_freq column to each frame using a shared frequency map."""
    out = []
    for df in frames:
        d = df.copy()
        d["stock_freq"] = d["stock_code"].map(freq_map).fillna(0.0)
        out.append(d)
    return out
