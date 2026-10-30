"""Score latest trading day with saved models (no retrain)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from nse_predictor.config import settings
from nse_predictor.features.cache import load_or_build_features
from nse_predictor.ingest.fundamentals import load_fundamentals
from nse_predictor.models.tabular import ReturnIntervalModel, VolatilityModel
from nse_predictor.training.meta import load_model_meta
from nse_predictor.universe import investable_equities, load_universe


def _require_models(models: Path) -> tuple[Path, Path, Path]:
    paths = (
        models / "direction_model.joblib",
        models / "return_model.joblib",
        models / "volatility_model.joblib",
    )
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run scripts/train_evaluate.py once before live scoring."
            )
    return paths


def _latest_per_stock(ready: pd.DataFrame) -> pd.DataFrame:
    idx = ready.groupby("stock_code")["trade_date"].idxmax()
    return ready.loc[idx].reset_index(drop=True)


def _attach_stock_freq(ready: pd.DataFrame, meta: dict, feature_cols: list[str] | None) -> pd.DataFrame:
    if not meta.get("stock_freq_feature") or "stock_freq" not in (feature_cols or []):
        return ready
    freq_map = meta.get("stock_freq_map") or {}
    if not freq_map:
        if "stock_freq" not in ready.columns:
            raise RuntimeError("model_meta requires stock_freq but stock_freq_map is empty")
        return ready
    out = ready.copy()
    out["stock_freq"] = out["stock_code"].map(freq_map).fillna(0.0)
    return out


def score_latest_forecasts(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    processed = Path(processed_dir or settings.data_processed_dir)
    models = Path(models_dir or settings.models_dir)
    meta = load_model_meta(models)
    dir_path, ret_path, vol_path = _require_models(models)

    prices = pd.read_parquet(processed / "prices.parquet")
    macro = pd.read_parquet(processed / "macro.parquet")
    sectors = pd.read_parquet(processed / "sectors.parquet")
    universe = load_universe(prices)
    investable = investable_equities(universe)

    feats, _ = load_or_build_features(
        prices,
        macro,
        sectors,
        universe,
        fundamentals=load_fundamentals(),
        processed_dir=processed,
    )
    feature_cols = meta.get("feature_cols")
    if feature_cols:
        missing = [c for c in feature_cols if c not in feats.columns]
        if missing:
            raise RuntimeError(
                f"Feature panel missing columns required by model_meta: {missing[:10]}"
            )

    ready = feats[feats["stock_code"].isin(investable)].dropna(subset=["ret_1d", "ma5_ratio"])
    if ready.empty:
        raise RuntimeError("No scorable feature rows")
    ready = _attach_stock_freq(ready, meta, feature_cols)
    latest = _latest_per_stock(ready)

    dpred = joblib.load(dir_path).predict(latest)
    rpred = ReturnIntervalModel.load(ret_path).predict(latest)
    vpred = VolatilityModel.load(vol_path).predict(latest)

    forecast = latest[["trade_date", "stock_code", "price"]].merge(
        dpred, on=["trade_date", "stock_code"]
    )
    ret = rpred[["trade_date", "stock_code", "pred_value", "lower", "upper"]].rename(
        columns={
            "pred_value": "ret5d_xs",
            "lower": "ret5d_xs_lo",
            "upper": "ret5d_xs_hi",
        }
    )
    ret["ret5d"] = ret["ret5d_xs"]
    ret["ret5d_lo"] = ret["ret5d_xs_lo"]
    ret["ret5d_hi"] = ret["ret5d_xs_hi"]
    forecast = forecast.merge(ret, on=["trade_date", "stock_code"])
    forecast = forecast.merge(
        vpred[["trade_date", "stock_code", "pred_value"]].rename(columns={"pred_value": "vol20d"}),
        on=["trade_date", "stock_code"],
    )

    out_path = processed / "latest_forecasts.parquet"
    forecast.to_parquet(out_path, index=False)
    try:
        from nse_predictor.monitoring.outcomes import append_forecast_log

        append_forecast_log(forecast, processed)
    except Exception:  # noqa: BLE001
        pass
    return {
        "n_stocks": int(forecast["stock_code"].nunique()),
        "n_flat": int((forecast["pred_label"].astype(str).str.lower() == "flat").sum()),
        "as_of": str(pd.to_datetime(forecast["trade_date"]).max().date()),
        "path": str(out_path),
        "universe_size": len(universe),
        "investable_size": len(investable),
        "target": "5d_excess_vs_NASI_dead_zone_1pct",
        "direction_head": meta.get("direction_head"),
        "decision_mode": meta.get("decision_mode"),
        "return_target": meta.get("target_return", "y_ret_5d_xs"),
    }
