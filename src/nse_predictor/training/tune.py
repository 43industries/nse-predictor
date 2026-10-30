"""LightGBM hyperparameter search on validation Rank IC / MAE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nse_predictor.eval.metrics import rank_ic, regression_metrics
from nse_predictor.models.tabular import DirectionModel, ReturnIntervalModel, VolatilityModel

DIRECTION_GRID = [
    {"num_leaves": 15, "min_child_samples": 60, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.05},
    {"num_leaves": 63, "min_child_samples": 20, "learning_rate": 0.03},
    {"num_leaves": 31, "min_child_samples": 80, "learning_rate": 0.03},
    {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.08},
    {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.05, "subsample": 0.7},
    {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.05, "colsample_bytree": 0.7},
    {"num_leaves": 47, "min_child_samples": 30, "learning_rate": 0.04},
]

REGRESSOR_GRID = [
    {"num_leaves": 15, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.05},
    {"num_leaves": 31, "learning_rate": 0.03},
    {"num_leaves": 63, "learning_rate": 0.03},
]


def _chrono_tail(df: pd.DataFrame, frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values("trade_date")
    cut = max(int(len(d) * (1.0 - frac)), 1)
    return d.iloc[:cut], d.iloc[cut:]


def _mean_rank_ic(valid: pd.DataFrame, pred: pd.DataFrame) -> tuple[dict, float]:
    if "y_ret_5d_xs" not in valid.columns:
        return {"mean_ic": float("nan")}, -1.0
    merged = valid.merge(
        pred[["trade_date", "stock_code", "prob_up"]],
        on=["trade_date", "stock_code"],
        how="inner",
    )
    ric = rank_ic(merged, "prob_up", "y_ret_5d_xs")
    ic = float(ric.get("mean_ic") or float("nan"))
    return ric, (-1.0 if np.isnan(ic) else ic)


def tune_direction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str] | None,
    max_trials: int = 24,
) -> tuple[dict, dict]:
    """Return (best_params, info) maximizing valid Rank IC of prob_up vs y_ret_5d_xs."""
    best_params: dict = {}
    best_ic = -1.0
    rows: list[dict] = []
    fit_tr, fit_es = _chrono_tail(train, 0.2)
    if len(fit_es) < 100:
        fit_tr, fit_es = train, valid

    for i, cfg in enumerate(DIRECTION_GRID[:max_trials]):
        params = {**cfg, "n_estimators": 500}
        try:
            m = DirectionModel(feature_cols=feature_cols, params=params)
            m.fit(fit_tr, eval_df=fit_es, early_stopping_rounds=40)
            ric, ic = _mean_rank_ic(valid, m.predict(valid))
            rows.append(
                {
                    "trial": i,
                    "params": params,
                    "valid_rank_ic": ric,
                    "best_iteration": m.best_iteration_,
                }
            )
            if ic > best_ic:
                best_ic, best_params = ic, params
        except Exception as exc:  # noqa: BLE001
            rows.append({"trial": i, "params": cfg, "error": str(exc)})

    if not best_params:
        best_params = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31}
    return best_params, {"best_rank_ic": best_ic, "trials": rows}


def tune_regressor(
    model_cls: type,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str] | None,
    target_hint: str,
    max_trials: int = 8,
) -> tuple[dict, dict]:
    best_params: dict = {}
    best_mae = float("inf")
    rows: list[dict] = []

    for i, cfg in enumerate(REGRESSOR_GRID[:max_trials]):
        params = {**cfg, "n_estimators": 400}
        try:
            if model_cls is ReturnIntervalModel:
                m = ReturnIntervalModel(
                    feature_cols=feature_cols, params=params, target_col=target_hint
                )
                m.fit(train, eval_df=valid, early_stopping_rounds=40)
                ev = valid.dropna(subset=[m.target_col])
                pred = m.predict(ev)
                y = ev[m.target_col].values
            else:
                m = VolatilityModel(feature_cols=feature_cols, params=params)
                m.fit(train)
                ev = valid.dropna(subset=["y_vol_20d"])
                pred = m.predict(ev)
                y = ev["y_vol_20d"].values
            met = regression_metrics(y, pred["pred_value"].values)
            rows.append({"trial": i, "params": params, "metrics": met})
            if met["mae"] < best_mae:
                best_mae, best_params = met["mae"], params
        except Exception as exc:  # noqa: BLE001
            rows.append({"trial": i, "error": str(exc)})

    if not best_params:
        best_params = {"n_estimators": 300, "learning_rate": 0.05}
    return best_params, {"best_mae": best_mae, "trials": rows}
