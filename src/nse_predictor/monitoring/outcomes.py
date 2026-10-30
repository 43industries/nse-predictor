"""Forecast log + lagged outcome metrics (5d horizon)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.eval.metrics import abstain_rate, rank_ic
from nse_predictor.features.cache import CACHE_PARQUET
from nse_predictor.monitoring.status import forecast_log_path, make_alert

LABEL_HORIZON_DAYS = 5
HIT_LOOKBACK = 60
IC_DAYS = 20
MIN_RESOLVED = 20
_LOG_COLS = ["trade_date", "stock_code", "prob_up", "pred_label"]


def append_forecast_log(
    forecasts: pd.DataFrame,
    processed: Path | None = None,
    as_of_run: str | None = None,
) -> Path:
    """Append latest forecast snapshot rows to forecast_log.parquet."""
    path = forecast_log_path(processed)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in _LOG_COLS if c not in forecasts.columns]
    if missing:
        raise ValueError(f"forecasts missing columns: {missing}")

    chunk = forecasts[_LOG_COLS].copy()
    chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
    chunk["as_of_run"] = as_of_run or datetime.now(timezone.utc).isoformat()

    if path.exists():
        prev = pd.read_parquet(path)
        prev["trade_date"] = pd.to_datetime(prev["trade_date"])
        both = pd.concat([prev, chunk], ignore_index=True)
        both = both.drop_duplicates(
            subset=["trade_date", "stock_code", "as_of_run"], keep="last"
        )
    else:
        both = chunk
    both.to_parquet(path, index=False)
    return path


def _load_label_panel(processed: Path) -> pd.DataFrame:
    cache = processed / CACHE_PARQUET
    if not cache.exists():
        return pd.DataFrame()
    df = pd.read_parquet(cache)
    if not {"trade_date", "stock_code"}.issubset(df.columns):
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def check_outcomes(processed: Path | None = None) -> tuple[dict, list[dict]]:
    processed = Path(processed or settings.data_processed_dir)
    alerts: list[dict] = []
    log_path = forecast_log_path(processed)
    if not log_path.exists():
        return {"skipped": True, "reason": "no_forecast_log"}, alerts

    log = pd.read_parquet(log_path)
    log["trade_date"] = pd.to_datetime(log["trade_date"])
    panel = _load_label_panel(processed)
    if panel.empty or "y_dir_5d" not in panel.columns:
        return {"skipped": True, "reason": "no_label_panel"}, alerts

    label_cols = ["trade_date", "stock_code", "y_dir_5d"]
    if "y_ret_5d_xs" in panel.columns:
        label_cols.append("y_ret_5d_xs")
    labels = panel[label_cols].dropna(subset=["y_dir_5d"]).copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])

    cutoff = labels["trade_date"].max() - pd.Timedelta(days=LABEL_HORIZON_DAYS)
    eligible = (
        log[log["trade_date"] <= cutoff]
        .sort_values("as_of_run")
        .drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
    )
    merged = eligible.merge(labels, on=["trade_date", "stock_code"], how="inner")
    n_resolved = int(len(merged))
    if n_resolved < MIN_RESOLVED:
        return {
            "skipped": True,
            "reason": "thin_history",
            "n_resolved": n_resolved,
            "min_resolved": MIN_RESOLVED,
        }, alerts

    merged["pred_label"] = merged["pred_label"].astype(str).str.lower()
    scored = merged.loc[merged["pred_label"] != "flat"].copy()
    scored["pred_up"] = (scored["pred_label"] == "up").astype(int)
    scored["y"] = scored["y_dir_5d"].astype(int)
    scored["correct"] = scored["pred_up"] == scored["y"]
    tail = scored.sort_values("trade_date").tail(HIT_LOOKBACK)
    hit_rate = float(tail["correct"].mean()) if len(tail) else float("nan")
    base = float(tail["y"].mean()) if len(tail) else 0.5
    chance = max(base, 1.0 - base)

    ric = {"mean_ic": float("nan"), "n_days": 0}
    if "y_ret_5d_xs" in merged.columns and "prob_up" in merged.columns:
        recent = sorted(merged["trade_date"].unique())[-IC_DAYS:]
        ric = rank_ic(merged[merged["trade_date"].isin(recent)], "prob_up", "y_ret_5d_xs")

    if hit_rate == hit_rate and hit_rate <= chance + 0.01:
        alerts.append(
            make_alert(
                "hit_rate_at_chance",
                "warn",
                f"hit_rate={hit_rate:.1%} ≤ chance+1pp ({chance:.1%}) over {len(tail)} rows",
            )
        )
    mic = ric.get("mean_ic")
    if mic is not None and mic == mic and mic <= 0 and ric.get("n_days", 0) >= 5:
        alerts.append(
            make_alert(
                "rank_ic_nonpositive",
                "warn",
                f"recent Rank IC mean={mic:.3f} ≤ 0 over {ric.get('n_days')} days",
            )
        )

    return {
        "skipped": False,
        "n_resolved": n_resolved,
        "n_scored_nonflat": int(len(scored)),
        "hit_rate_nonflat": hit_rate,
        "hit_lookback": int(len(tail)),
        "chance_rate": chance,
        "abstain_rate_resolved": abstain_rate(merged["pred_label"]),
        "rank_ic": ric,
    }, alerts
