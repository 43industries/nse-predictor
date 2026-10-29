#!/usr/bin/env python3
"""Build features, train baselines + tabular models, walk-forward evaluate, save artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.baselines.rules import always_up, moving_average_direction
from nse_predictor.config import settings
from nse_predictor.eval.error_history import build_error_history
from nse_predictor.eval.metrics import coverage, regression_metrics
from nse_predictor.eval.walk_forward import evaluate_direction, stress_slice, time_split, walk_forward_splits
from nse_predictor.features.build import build_feature_matrix
from nse_predictor.models.classical import fit_arima_forecast, fit_garch_vol
from nse_predictor.models.tabular import DirectionModel, ReturnIntervalModel, VolatilityModel
from nse_predictor.universe import pick_universe


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = settings.data_processed_dir
    prices = pd.read_parquet(p / "prices.parquet")
    macro = pd.read_parquet(p / "macro.parquet")
    sectors = pd.read_parquet(p / "sectors.parquet")
    return prices, macro, sectors


def main() -> int:
    prices, macro, sectors = load_processed()
    universe = pick_universe(prices)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)

    feats = build_feature_matrix(prices, macro, sectors, tickers=universe)
    # Drop rows lacking core features / targets for training
    # Keep equity rows with a direction label and at least short-horizon features
    train_ready = feats.dropna(subset=["y_dir_5d", "ret_1d", "ma5_ratio"])
    train_ready = train_ready[~train_ready["stock_code"].astype(str).str.contains(r"-R$|^\^", regex=True)]
    train, valid, test = time_split(train_ready, settings.train_end, settings.valid_end)

    results: dict = {
        "universe": universe,
        "n_rows": len(train_ready),
        "splits": {
            "train": [str(train["trade_date"].min().date()), str(train["trade_date"].max().date()), len(train)],
            "valid": [str(valid["trade_date"].min().date()) if len(valid) else None,
                      str(valid["trade_date"].max().date()) if len(valid) else None, len(valid)],
            "test": [str(test["trade_date"].min().date()) if len(test) else None,
                     str(test["trade_date"].max().date()) if len(test) else None, len(test)],
        },
        "baselines": {},
        "models": {},
        "walk_forward": [],
        "stress": {},
        "classical": {},
    }

    # --- Baselines on test ---
    eval_df = test if len(test) else valid
    if eval_df.empty:
        eval_df = train_ready.tail(500)

    au = always_up(eval_df)
    y = eval_df["y_dir_5d"].astype(int).values
    results["baselines"]["always_up"] = evaluate_direction(y, np.ones_like(y), au["prob_up"].values)

    ma = moving_average_direction(eval_df)
    ma_pred = (ma["pred_label"] == "up").astype(int).values
    results["baselines"]["ma_5_20"] = evaluate_direction(y, ma_pred, ma["prob_up"].values)

    # Price baseline for returns
    zero_ret = np.zeros(len(eval_df))
    results["baselines"]["tomorrow_eq_today_ret5d"] = regression_metrics(
        eval_df["y_ret_5d"].values, zero_ret
    )

    # --- Tabular models ---
    fit_df = pd.concat([train, valid]) if len(valid) else train
    dir_model = DirectionModel().fit(fit_df)
    dir_pred = dir_model.predict(eval_df)
    yhat = (dir_pred["pred_label"] == "up").astype(int).values
    results["models"]["lightgbm_direction"] = evaluate_direction(y, yhat, dir_pred["prob_up"].values)
    dir_model.save(settings.models_dir / "direction_model.joblib")

    # Per-stock error history on the evaluation panel (test preferred)
    err_hist = build_error_history(eval_df, dir_pred, horizon_days=5)
    err_path = settings.data_processed_dir / "error_history.parquet"
    err_hist.to_parquet(err_path, index=False)
    results["error_history_path"] = str(err_path)
    results["error_history_rows"] = int(len(err_hist))

    shap_df = dir_model.shap_summary(eval_df)
    if shap_df is not None:
        shap_path = ROOT / "reports" / "shap_direction.csv"
        shap_df.to_csv(shap_path, index=False)
        results["models"]["shap_top5"] = shap_df.head(5).to_dict(orient="records")

    ret_model = ReturnIntervalModel().fit(fit_df)
    ret_pred = ret_model.predict(eval_df)
    results["models"]["lightgbm_return_5d"] = regression_metrics(
        eval_df["y_ret_5d"].values, ret_pred["pred_value"].values
    )
    results["models"]["return_interval_coverage"] = coverage(
        eval_df["y_ret_5d"].values, ret_pred["lower"].values, ret_pred["upper"].values
    )
    ret_model.save(settings.models_dir / "return_model.joblib")

    vol_model = VolatilityModel().fit(fit_df.dropna(subset=["y_vol_20d"]))
    vol_eval = eval_df.dropna(subset=["y_vol_20d"])
    if len(vol_eval):
        vol_pred = vol_model.predict(vol_eval)
        results["models"]["lightgbm_vol_20d"] = regression_metrics(
            vol_eval["y_vol_20d"].values, vol_pred["pred_value"].values
        )
    vol_model.save(settings.models_dir / "volatility_model.joblib")

    # --- Walk-forward direction ---
    for fold_name, tr, te in walk_forward_splits(train_ready, start_year=2018, end_year=2025):
        te = te.dropna(subset=["y_dir_5d"])
        if te.empty or len(tr) < 500:
            continue
        m = DirectionModel().fit(tr)
        p = m.predict(te)
        yt = te["y_dir_5d"].astype(int).values
        yp = (p["pred_label"] == "up").astype(int).values
        metrics = evaluate_direction(yt, yp, p["prob_up"].values)
        # baseline always-up on same fold
        base = evaluate_direction(yt, np.ones_like(yt))
        results["walk_forward"].append(
            {
                "fold": fold_name,
                "model": metrics,
                "baseline_always_up": base,
                "beats_baseline": metrics["accuracy"] > base["accuracy"],
            }
        )

    # --- Stress periods ---
    for period in ["covid_2020", "fx_2023"]:
        sl = stress_slice(train_ready, period).dropna(subset=["y_dir_5d"])
        if len(sl) < 50:
            results["stress"][period] = {"n": len(sl), "skipped": True}
            continue
        # use model trained excluding the stress year roughly
        pre = train_ready[train_ready["trade_date"] < sl["trade_date"].min()]
        if len(pre) < 500:
            results["stress"][period] = {"n": len(sl), "skipped": True}
            continue
        m = DirectionModel().fit(pre)
        p = m.predict(sl)
        yt = sl["y_dir_5d"].astype(int).values
        yp = (p["pred_label"] == "up").astype(int).values
        results["stress"][period] = evaluate_direction(yt, yp, p["prob_up"].values)

    # --- Classical per top liquid stock ---
    sample_code = universe[0]
    hist = prices[prices["stock_code"] == sample_code].sort_values("trade_date")
    results["classical"]["arima_example"] = {
        "stock": sample_code,
        **fit_arima_forecast(hist["price"], steps=5),
    }
    results["classical"]["garch_example"] = {
        "stock": sample_code,
        **fit_garch_vol(hist["price"].pct_change(), horizon=20),
    }

    # Latest forecasts for API/dashboard
    latest_dates = train_ready.groupby("stock_code")["trade_date"].max()
    latest_rows = []
    for code, dt in latest_dates.items():
        if code in {"NASI", "NSE20"}:
            continue
        row = train_ready[(train_ready["stock_code"] == code) & (train_ready["trade_date"] == dt)]
        if len(row):
            latest_rows.append(row)
    if latest_rows:
        latest = pd.concat(latest_rows)
        dpred = dir_model.predict(latest)
        rpred = ret_model.predict(latest)
        vpred = vol_model.predict(latest)
        forecast = latest[["trade_date", "stock_code", "price"]].merge(
            dpred, on=["trade_date", "stock_code"]
        )
        forecast = forecast.merge(
            rpred[["trade_date", "stock_code", "pred_value", "lower", "upper"]].rename(
                columns={"pred_value": "ret5d", "lower": "ret5d_lo", "upper": "ret5d_hi"}
            ),
            on=["trade_date", "stock_code"],
        )
        forecast = forecast.merge(
            vpred[["trade_date", "stock_code", "pred_value"]].rename(columns={"pred_value": "vol20d"}),
            on=["trade_date", "stock_code"],
        )
        forecast_path = settings.data_processed_dir / "latest_forecasts.parquet"
        forecast.to_parquet(forecast_path, index=False)
        results["latest_forecasts_path"] = str(forecast_path)

    out_json = ROOT / "reports" / "evaluation_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ["baselines", "models"]}, indent=2, default=str))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
