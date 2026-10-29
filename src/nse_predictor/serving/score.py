"""Score latest trading day with saved models (no retrain)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.features.build import build_feature_matrix
from nse_predictor.models.tabular import DirectionModel, ReturnIntervalModel, VolatilityModel
from nse_predictor.universe import load_universe


def score_latest_forecasts(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    processed = Path(processed_dir or settings.data_processed_dir)
    models = Path(models_dir or settings.models_dir)

    prices = pd.read_parquet(processed / "prices.parquet")
    macro = pd.read_parquet(processed / "macro.parquet")
    sectors = pd.read_parquet(processed / "sectors.parquet")
    universe = load_universe(prices)

    dir_path = models / "direction_model.joblib"
    ret_path = models / "return_model.joblib"
    vol_path = models / "volatility_model.joblib"
    for p in (dir_path, ret_path, vol_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run scripts/train_evaluate.py once before live scoring."
            )

    feats = build_feature_matrix(prices, macro, sectors, tickers=universe)
    feats = feats[~feats["stock_code"].astype(str).str.contains(r"-R$|^\^", regex=True)]
    # Need feature columns present even if targets are NaN for the latest day
    ready = feats.dropna(subset=["ret_1d", "ma5_ratio"])
    if ready.empty:
        raise RuntimeError("No scorable feature rows")

    latest_dates = ready.groupby("stock_code")["trade_date"].max()
    latest_rows = []
    for code, dt in latest_dates.items():
        row = ready[(ready["stock_code"] == code) & (ready["trade_date"] == dt)]
        if len(row):
            latest_rows.append(row)
    latest = pd.concat(latest_rows, ignore_index=True)

    dir_model = DirectionModel.load(dir_path)
    ret_model = ReturnIntervalModel.load(ret_path)
    vol_model = VolatilityModel.load(vol_path)

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
        vpred[["trade_date", "stock_code", "pred_value"]].rename(
            columns={"pred_value": "vol20d"}
        ),
        on=["trade_date", "stock_code"],
    )

    out_path = processed / "latest_forecasts.parquet"
    forecast.to_parquet(out_path, index=False)
    return {
        "n_stocks": int(forecast["stock_code"].nunique()),
        "as_of": str(pd.to_datetime(forecast["trade_date"]).max().date()),
        "path": str(out_path),
        "universe_size": len(universe),
    }
