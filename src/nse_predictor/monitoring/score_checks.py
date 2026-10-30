"""Live forecast freshness and score hygiene."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.config import settings
from nse_predictor.features.cache import CACHE_PARQUET
from nse_predictor.monitoring.status import make_alert
from nse_predictor.training.meta import load_model_meta

FRESH_WARN_DAYS = 3
FRESH_FAIL_DAYS = 7
ABSTAIN_WARN = 0.70
COLLAPSE_STD = 0.02
COLLAPSE_UNIQUE = 5
COLLAPSE_UP_FRAC = 0.99


def _finite(x) -> bool:
    return x is not None and x == x


def _freshness_alerts(as_of_date: date | None, age_days: int | None) -> list[dict]:
    if age_days is None or as_of_date is None:
        return []
    if age_days > FRESH_FAIL_DAYS:
        return [
            make_alert(
                "forecast_stale",
                "fail",
                f"forecast as_of {as_of_date} is {age_days}d old (> {FRESH_FAIL_DAYS})",
            )
        ]
    if age_days > FRESH_WARN_DAYS and date.today().weekday() < 5:
        return [
            make_alert(
                "forecast_stale",
                "warn",
                f"forecast as_of {as_of_date} is {age_days}d old (> {FRESH_WARN_DAYS})",
            )
        ]
    return []


def _prob_stats(fc: pd.DataFrame) -> dict:
    probs = fc["prob_up"].astype(float) if "prob_up" in fc.columns else pd.Series(dtype=float)
    labels = (
        fc["pred_label"].astype(str).str.lower()
        if "pred_label" in fc.columns
        else pd.Series(dtype=str)
    )
    nonflat = labels != "flat"
    return {
        "abstain_rate": float((labels == "flat").mean()) if len(labels) else float("nan"),
        "up_frac_nonflat": float((labels[nonflat] == "up").mean()) if nonflat.any() else float("nan"),
        "n_nonflat": int(nonflat.sum()),
        "std_prob_up": float(probs.std()) if len(probs) else float("nan"),
        "n_unique_probs": int(pd.Series(np.round(probs, 4)).nunique()) if len(probs) else 0,
        "mean_prob_up": float(probs.mean()) if len(probs) else float("nan"),
    }


def _is_collapsed(stats: dict) -> bool:
    std_p = stats["std_prob_up"]
    up_frac = stats["up_frac_nonflat"]
    return (
        (_finite(std_p) and std_p < COLLAPSE_STD)
        or stats["n_unique_probs"] < COLLAPSE_UNIQUE
        or (_finite(up_frac) and up_frac >= COLLAPSE_UP_FRAC and stats["n_nonflat"] >= 5)
    )


def _missing_feature_cols(processed: Path, feature_cols: list[str]) -> tuple[list[str], str | None]:
    """Check model_meta cols against features_v2 cache columns (no rebuild)."""
    if not feature_cols:
        return [], None
    cache = processed / CACHE_PARQUET
    if not cache.exists():
        return [], f"{CACHE_PARQUET} missing"
    try:
        import pyarrow.parquet as pq

        cols = set(pq.read_schema(cache).names)
        return [c for c in feature_cols if c not in cols], None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def check_score_hygiene(
    processed: Path | None = None, models: Path | None = None
) -> tuple[dict, list[dict]]:
    processed = Path(processed or settings.data_processed_dir)
    models = Path(models or settings.models_dir)
    alerts: list[dict] = []
    fc_path = processed / "latest_forecasts.parquet"
    if not fc_path.exists():
        alerts.append(make_alert("forecasts_missing", "fail", "latest_forecasts.parquet missing"))
        return {"present": False}, alerts

    fc = pd.read_parquet(fc_path)
    as_of = pd.to_datetime(fc["trade_date"]).max() if len(fc) else pd.NaT
    as_of_date = as_of.date() if pd.notna(as_of) else None
    age_days = (date.today() - as_of_date).days if as_of_date else None
    alerts.extend(_freshness_alerts(as_of_date, age_days))

    stats = _prob_stats(fc)
    collapsed = _is_collapsed(stats)
    if collapsed:
        alerts.append(
            make_alert(
                "score_collapse",
                "fail",
                f"score collapse: std={stats['std_prob_up']:.4f} "
                f"unique={stats['n_unique_probs']} up_frac_nonflat={stats['up_frac_nonflat']}",
            )
        )
    if _finite(stats["abstain_rate"]) and stats["abstain_rate"] > ABSTAIN_WARN:
        alerts.append(
            make_alert(
                "abstain_high",
                "warn",
                f"abstain_rate={stats['abstain_rate']:.0%} > {ABSTAIN_WARN:.0%}",
            )
        )

    meta = load_model_meta(models)
    missing_cols, feat_err = _missing_feature_cols(processed, list(meta.get("feature_cols") or []))
    if feat_err:
        alerts.append(make_alert("feature_check_error", "warn", f"could not verify feature cols: {feat_err}"))
    elif missing_cols:
        alerts.append(
            make_alert(
                "feature_mismatch",
                "fail",
                f"panel missing model_meta cols: {missing_cols[:8]}",
            )
        )

    block = {
        "present": True,
        "as_of": str(as_of_date) if as_of_date else None,
        "age_days": age_days,
        "n_stocks": int(fc["stock_code"].nunique()) if "stock_code" in fc.columns else len(fc),
        "abstain_rate": stats["abstain_rate"],
        "mean_prob_up": stats["mean_prob_up"],
        "std_prob_up": stats["std_prob_up"],
        "n_unique_probs": stats["n_unique_probs"],
        "up_frac_nonflat": stats["up_frac_nonflat"],
        "collapsed": collapsed,
        "missing_feature_cols": missing_cols[:20],
        "checked_at_local": datetime.now().isoformat(timespec="seconds"),
    }
    return block, alerts
