"""Reusable fit / select helpers for NSE tabular models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nse_predictor.eval.diagnostics import binary_from_labels, tune_threshold
from nse_predictor.eval.metrics import abstain_rate, coverage, rank_ic
from nse_predictor.eval.protocol import LABEL_HORIZON_DAYS, apply_embargo, embargo_train_tail
from nse_predictor.eval.walk_forward import evaluate_direction, walk_forward_splits
from nse_predictor.models.tabular import (
    ABSTAIN_BAND,
    DirectionModel,
    DirectionRankModel,
    ReturnIntervalModel,
    HAS_LGBM,
)
from nse_predictor.training.tune import tune_direction, tune_regressor

MIN_VALID = 200
STOCK_FREQ_FLAG_FRAC = 0.4
STOCK_FREQ_IC_LIFT = 0.01


def split_meta(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return [None, None, 0]
    return [
        str(df["trade_date"].min().date()),
        str(df["trade_date"].max().date()),
        int(len(df)),
    ]


def _finite(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _concat_tv(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    if len(valid) == 0:
        return train
    out = pd.concat([train, valid], ignore_index=True)
    return out if len(out) else train


def valid_rank_ic(valid: pd.DataFrame, pred: pd.DataFrame) -> float:
    if "y_ret_5d_xs" not in valid.columns:
        return float("nan")
    merged = valid.merge(
        pred[["trade_date", "stock_code", "prob_up"]],
        on=["trade_date", "stock_code"],
        how="inner",
    )
    return _finite((rank_ic(merged, "prob_up", "y_ret_5d_xs") or {}).get("mean_ic"))


def direction_metrics_bundle(df: pd.DataFrame, model) -> tuple[dict, pd.DataFrame]:
    pred = model.predict(df)
    y = df["y_dir_5d"].astype(int).values
    yp_f = binary_from_labels(pred["pred_label"])
    mask = ~np.isnan(yp_f)
    n_flat = int((pred["pred_label"].astype(str).str.lower() == "flat").sum())
    ar = abstain_rate(pred["pred_label"])
    if mask.sum() < 20:
        yhat = (pred["pred_label"] == "up").astype(int).values
        metrics = evaluate_direction(y, yhat, pred["prob_up"].values)
        metrics["n_flat"] = n_flat
        metrics["abstain_rate"] = ar
        return metrics, pred
    metrics = evaluate_direction(y[mask], yp_f[mask].astype(int), pred.loc[mask, "prob_up"].values)
    metrics["n_flat"] = n_flat
    metrics["n_scored"] = int(mask.sum())
    metrics["abstain_rate"] = ar
    return metrics, pred


def _configure_decision(model, valid: pd.DataFrame) -> dict:
    """Calibrate + threshold on valid; fall back to ranked mode if needed."""
    model.calibrate(valid)
    valid_pred = model.predict(valid)
    tuned = tune_threshold(
        valid["y_dir_5d"].astype(int).values,
        valid_pred["prob_up"].values,
        min_threshold=0.35,
        max_threshold=0.65,
        metric="f1",
        reject_always_up=True,
    )
    ric = valid_rank_ic(valid, valid_pred)
    if tuned.get("found") and tuned.get("beats_always_up"):
        mode, thr = "threshold", float(tuned["threshold"])
    elif ric == ric and ric > 0:
        mode, thr = "ranked", 0.5
    else:
        mode, thr = "threshold", float(tuned["threshold"]) if tuned.get("found") else 0.5

    model.set_threshold(thr)
    model.set_decision_mode(mode)
    valid_metrics, _ = direction_metrics_bundle(valid, model)
    return {
        "threshold": model.threshold_,
        "calibration": "isotonic" if model.calibration_status_ == "isotonic" else "none",
        "calibration_status": model.calibration_status_,
        "decision_mode": mode,
        "valid_direction": valid_metrics,
        "valid_rank_ic": ric,
        "tune": tuned,
    }


def _default_decision() -> dict:
    return {
        "threshold": 0.5,
        "calibration": "none",
        "calibration_status": "none",
        "decision_mode": "threshold",
        "valid_direction": {},
    }


def fit_direction_selected(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    min_valid: int = MIN_VALID,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
    tune: bool = True,
) -> tuple[DirectionModel, dict]:
    """Train binary direction with optional tuning; decision layer on valid."""
    best_params = params or {}
    tune_info: dict = {}
    if tune and len(valid) >= min_valid:
        best_params, tune_info = tune_direction(train, valid, feature_cols)

    fit_base = train if len(train) else valid
    probe = DirectionModel(feature_cols=feature_cols, params=best_params)
    probe.fit(fit_base, eval_df=valid if len(valid) >= 100 else None)

    selection: dict = {
        "direction_head": "binary",
        "best_params": best_params,
        "tune_info": {
            "best_rank_ic": tune_info.get("best_rank_ic"),
            "n_trials": len(tune_info.get("trials") or []),
        },
        "scale_pos_weight": probe.scale_pos_weight_,
        "best_iteration": probe.best_iteration_,
        "features": list(probe.features),
        "n_features": len(probe.features),
        "abstain_band": ABSTAIN_BAND,
    }
    if len(valid) >= min_valid:
        selection.update(_configure_decision(probe, valid))
    else:
        probe.set_threshold(0.5)
        probe.set_decision_mode("threshold")
        selection.update(_default_decision())

    model = DirectionModel(feature_cols=feature_cols, params=best_params)
    model.fit(_concat_tv(train, valid) if len(valid) else fit_base, early_stopping_rounds=None)
    model.copy_decision_from(probe)
    selection["scale_pos_weight"] = model.scale_pos_weight_
    return model, selection


def fit_direction_ranker_selected(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[DirectionRankModel, dict]:
    if not HAS_LGBM:
        raise RuntimeError("LightGBM required for ranker")
    probe = DirectionRankModel(feature_cols=feature_cols)
    probe.fit(train)
    selection = {"direction_head": "ranker", "abstain_band": ABSTAIN_BAND}
    selection.update(_configure_decision(probe, valid))

    model = DirectionRankModel(feature_cols=feature_cols)
    model.fit(_concat_tv(train, valid))
    model.copy_decision_from(probe)
    selection["features"] = list(model.features)
    selection["n_features"] = len(model.features)
    return model, selection


def select_direction_head(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[object, dict, dict]:
    """Ablate binary vs ranker on probe-stage valid Rank IC."""
    bin_model, bin_sel = fit_direction_selected(train, valid, feature_cols=feature_cols, tune=True)
    bin_ic = _finite(bin_sel.get("valid_rank_ic"))
    ablation: dict = {
        "binary": {
            "valid_rank_ic": bin_ic,
            "best_params": bin_sel.get("best_params"),
            "tune_info": bin_sel.get("tune_info"),
            "decision_mode": bin_sel.get("decision_mode"),
            "calibration_status": bin_sel.get("calibration_status"),
        },
    }
    winner_model, winner_sel, winner_name, best_ic = bin_model, bin_sel, "binary", (
        bin_ic if bin_ic == bin_ic else -1.0
    )

    try:
        rk_model, rk_sel = fit_direction_ranker_selected(train, valid, feature_cols=feature_cols)
        rk_ic = _finite(rk_sel.get("valid_rank_ic"))
        ablation["ranker"] = {
            "valid_rank_ic": rk_ic,
            "decision_mode": rk_sel.get("decision_mode"),
            "calibration_status": rk_sel.get("calibration_status"),
        }
        if rk_ic == rk_ic and rk_ic > best_ic:
            winner_model, winner_sel, winner_name, best_ic = rk_model, rk_sel, "ranker", rk_ic
    except Exception as exc:  # noqa: BLE001
        ablation["ranker"] = {"error": str(exc)}

    ablation["selected"] = winner_name
    ablation["selected_rank_ic"] = best_ic
    winner_sel["direction_head"] = winner_name
    if not winner_sel.get("best_params"):
        winner_sel["best_params"] = bin_sel.get("best_params") or {}
    if winner_name == "ranker":
        winner_sel["best_params_ranker"] = getattr(winner_model, "params", {}) or {}
        winner_sel["tune_info"] = bin_sel.get("tune_info") or {}
        winner_sel["best_iteration"] = bin_sel.get("best_iteration")
    return winner_model, winner_sel, ablation


def fit_return_selected(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    q_low: float = 0.10,
    q_high: float = 0.90,
    min_valid: int = MIN_VALID,
    feature_cols: list[str] | None = None,
    target_col: str = "y_ret_5d_xs",
    tune: bool = True,
) -> tuple[ReturnIntervalModel, dict]:
    params: dict = {}
    tune_info: dict = {}
    if tune and len(valid) >= min_valid:
        params, tune_info = tune_regressor(
            ReturnIntervalModel, train, valid, feature_cols, target_col
        )

    fit_base = train if len(train) else valid
    probe = ReturnIntervalModel(feature_cols=feature_cols, params=params, target_col=target_col)
    probe.fit(fit_base, eval_df=valid if len(valid) >= 100 else None)
    selection: dict = {
        "return_target": probe.target_col,
        "best_params_return": params,
        "tune_return": tune_info,
    }
    valid_ret = valid.dropna(subset=[probe.target_col]) if len(valid) else valid
    if len(valid_ret) >= min_valid:
        probe.set_residual_quantiles(valid_ret, q_low=q_low, q_high=q_high)
        pred = probe.predict(valid_ret)
        selection["valid_return_coverage"] = coverage(
            valid_ret[probe.target_col].values, pred["lower"].values, pred["upper"].values
        )

    model = ReturnIntervalModel(feature_cols=feature_cols, params=params, target_col=target_col)
    model.fit(_concat_tv(train, valid) if len(valid) else fit_base, early_stopping_rounds=None)
    model.copy_interval_from(probe)
    selection["return_quantiles"] = {
        "q_low": model.q_low_,
        "q_high": model.q_high_,
        "p_low": model.quantile_low_p_,
        "p_high": model.quantile_high_p_,
        "target_coverage": model.quantile_high_p_ - model.quantile_low_p_,
    }
    return model, selection


def _new_direction_head(
    direction_head: str,
    feature_cols: list[str] | None,
    params: dict | None,
):
    if direction_head == "ranker" and HAS_LGBM:
        try:
            return DirectionRankModel(feature_cols=feature_cols)
        except Exception:  # noqa: BLE001
            pass
    return DirectionModel(feature_cols=feature_cols, params=params or {})


def _fit_direction(
    head: str,
    rows: pd.DataFrame,
    feature_cols: list[str] | None,
    params: dict | None,
    eval_df: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = 50,
):
    m = _new_direction_head(head, feature_cols, params)
    try:
        if isinstance(m, DirectionModel):
            m.fit(rows, eval_df=eval_df, early_stopping_rounds=early_stopping_rounds)
        else:
            m.fit(rows)
        return m
    except Exception:  # noqa: BLE001
        m = DirectionModel(feature_cols=feature_cols, params=params or {})
        m.fit(rows, eval_df=eval_df, early_stopping_rounds=early_stopping_rounds)
        return m


def walk_forward_summary(folds: list[dict]) -> dict:
    ics = [
        float(mic)
        for f in folds
        if (mic := (f.get("rank_ic") or {}).get("mean_ic")) is not None and mic == mic
    ]
    return {
        "wf_mean_rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "wf_folds_ic_positive": int(sum(1 for x in ics if x > 0)),
        "wf_folds_beat_always_up": int(sum(1 for f in folds if f.get("beats_baseline"))),
        "n_folds": int(len(folds)),
    }


def run_walk_forward_direction(
    panel: pd.DataFrame,
    fallback_threshold: float = 0.5,
    start_year: int = 2018,
    end_year: int = 2025,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
    decision_mode: str = "threshold",
    direction_head: str = "binary",
    horizon_days: int = LABEL_HORIZON_DAYS,
) -> list[dict]:
    folds: list[dict] = []
    want_ranker = direction_head == "ranker" and HAS_LGBM
    head0 = "ranker" if want_ranker else "binary"

    for fold_name, tr, te in walk_forward_splits(panel, start_year=start_year, end_year=end_year):
        te = te.dropna(subset=["y_dir_5d"])
        if te.empty or len(tr) < 500:
            continue
        tr = embargo_train_tail(tr.sort_values("trade_date"), horizon_days=horizon_days)
        if len(tr) < 500:
            continue
        cut = int(len(tr) * 0.8)
        tr_fit, tr_val = tr.iloc[:cut], tr.iloc[cut:]
        if len(tr_val):
            val_start = pd.to_datetime(tr_val["trade_date"]).min()
            tr_fit = apply_embargo(
                tr_fit, val_start - pd.Timedelta(days=1), horizon_days=horizon_days
            )
        fit_rows = tr_fit if len(tr_fit) >= 500 else tr
        m = _fit_direction(
            head0,
            fit_rows,
            feature_cols,
            params,
            eval_df=tr_val if len(tr_val) >= 100 else None,
        )

        if len(tr_val) >= MIN_VALID:
            _configure_decision(m, tr_val)
            probe = m
            head = "ranker" if isinstance(probe, DirectionRankModel) else "binary"
            m = _fit_direction(head, tr, feature_cols, params, early_stopping_rounds=None)
            m.copy_decision_from(probe)
        else:
            m.set_threshold(fallback_threshold)
            m.set_decision_mode(decision_mode)

        metrics, pred = direction_metrics_bundle(te, m)
        ric_full = (
            rank_ic(
                te.merge(
                    pred[["trade_date", "stock_code", "prob_up"]],
                    on=["trade_date", "stock_code"],
                    how="inner",
                ),
                "prob_up",
                "y_ret_5d_xs",
            )
            if "y_ret_5d_xs" in te.columns
            else {"mean_ic": float("nan"), "n_days": 0}
        )
        base = evaluate_direction(te["y_dir_5d"].astype(int).values, np.ones(len(te), dtype=int))
        folds.append(
            {
                "fold": fold_name,
                "threshold": m.threshold_,
                "decision_mode": getattr(m, "decision_mode_", decision_mode),
                "model": metrics,
                "rank_ic": ric_full,
                "abstain_rate": metrics.get("abstain_rate"),
                "baseline_always_up": base,
                "beats_baseline": metrics.get("accuracy", 0) > base.get("accuracy", 0),
            }
        )
    return folds


def latest_labeled_rows(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    idx = panel.groupby("stock_code")["trade_date"].idxmax()
    return panel.loc[idx].reset_index(drop=True)


def maybe_add_stock_freq_feature(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    flagged_frac: float,
    baseline_rank_ic: float | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, bool, dict]:
    """Trial stock frequency encoding when many names are flagged; keep if Rank IC +0.01."""
    info: dict = {"flagged_frac": flagged_frac, "kept": False}
    if flagged_frac < STOCK_FREQ_FLAG_FRAC:
        return feature_cols, train, valid, False, info

    sizes = train.groupby("stock_code").size()
    freq_map = (sizes / sizes.max()).to_dict()
    tr, va = train.copy(), valid.copy()
    tr["stock_freq"] = tr["stock_code"].map(freq_map).fillna(0.0)
    va["stock_freq"] = va["stock_code"].map(freq_map).fillna(0.0)
    cols = list(feature_cols) + ["stock_freq"]

    try:
        probe = DirectionModel(feature_cols=cols, params={"n_estimators": 200})
        probe.fit(tr, early_stopping_rounds=None)
        trial_ic = valid_rank_ic(va, probe.predict(va))
        base = _finite(baseline_rank_ic)
        info.update({"trial_rank_ic": trial_ic, "baseline_rank_ic": base, "freq_map": freq_map})
        keep = trial_ic == trial_ic and (base != base or trial_ic >= base + STOCK_FREQ_IC_LIFT)
        info["kept"] = bool(keep)
        if keep:
            return cols, tr, va, True, info
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return feature_cols, train, valid, False, info
