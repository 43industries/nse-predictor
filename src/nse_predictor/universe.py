"""Investable universe selection shared by train and live scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from nse_predictor.config import ROOT, settings

UNIVERSE_PATH = ROOT / "reports" / "universe_investable.json"
EXCLUDE_RE = r"-R$|^\^|NASI|N20I|NSE20|NSE 20|NSE-20"
_EXCLUDE_COMPILED = re.compile(EXCLUDE_RE, re.IGNORECASE)


def _equity_frame(prices: pd.DataFrame) -> pd.DataFrame:
    code = prices["stock_code"].astype(str)
    exclude = code.str.contains(EXCLUDE_RE, case=False, na=False, regex=True)
    return prices.loc[~exclude].copy()


def pick_universe(prices: pd.DataFrame, n: int = 15) -> list[str]:
    """Liquidity + hygiene screen; append index codes for feature context only."""
    eq = _equity_frame(prices)
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])
    stats = eq.groupby("stock_code").agg(
        avg_volume=("volume", "mean"),
        median_price=("price", "median"),
        n_days=("trade_date", "count"),
        last=("trade_date", "max"),
    )
    recent = (
        eq.sort_values("trade_date")
        .groupby("stock_code", group_keys=False)
        .tail(252)
        .groupby("stock_code")["price"]
        .median()
        .rename("recent_median_price")
    )
    stats = stats.join(recent, how="left")
    # Prefer recent price level so penny/illiquid names (e.g. MSC) drop out
    stats = stats[
        (stats["n_days"] >= 1500)
        & (stats["last"] >= pd.Timestamp("2023-01-01"))
        & (stats["recent_median_price"].fillna(stats["median_price"]) >= 1.0)
    ]
    if stats.empty:
        stats = (
            eq.groupby("stock_code")
            .agg(avg_volume=("volume", "mean"), median_price=("price", "median"))
        )
        stats = stats[stats["median_price"] >= 1.0]
    # Prefer names above median ADV among survivors, then take top-n
    if len(stats) > n:
        med_adv = float(stats["avg_volume"].median())
        above = stats[stats["avg_volume"] >= med_adv]
        if len(above) >= max(8, n // 2):
            stats = above
    liq = stats.sort_values("avg_volume", ascending=False).head(n).index.tolist()
    if len(liq) < 8:
        fallback = (
            _equity_frame(prices)
            .groupby("stock_code")["volume"]
            .mean()
            .sort_values(ascending=False)
            .head(n)
            .index.tolist()
        )
        liq = fallback

    present = set(prices["stock_code"].astype(str))
    for idx in ["^NASI", "^N20I", "NASI", "NSE20"]:
        if idx in present and idx not in liq:
            liq.append(idx)
    return liq


def investable_equities(universe: list[str]) -> list[str]:
    """Equities only (no index tickers) for training labels and scoring."""
    out: list[str] = []
    for c in universe:
        s = str(c).strip()
        if not s or s.startswith("^") or _EXCLUDE_COMPILED.search(s):
            continue
        out.append(s)
    return out


def save_universe(universe: list[str], path: Path | None = None) -> Path:
    path = path or UNIVERSE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe": list(universe),
        "investable": investable_equities(universe),
        "gates": {
            "n_days_min": 1500,
            "last_date_min": "2023-01-01",
            "median_price_min": 1.0,
            "recent_median_price_min": 1.0,
            "recent_window_days": 252,
            "adv_above_median_preferred": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_universe(prices: pd.DataFrame | None = None) -> list[str]:
    """Prefer persisted investable universe file; then evaluation JSON; else screen."""
    if UNIVERSE_PATH.exists():
        try:
            data = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
            uni = data.get("universe") or []
            if uni:
                return list(uni)
        except Exception:  # noqa: BLE001
            pass
    ev_path = ROOT / "reports" / "evaluation_results.json"
    if ev_path.exists():
        try:
            data = json.loads(ev_path.read_text(encoding="utf-8"))
            uni = data.get("universe") or []
            if uni:
                return list(uni)
        except Exception:  # noqa: BLE001
            pass
    if prices is None:
        path = settings.data_processed_dir / "prices.parquet"
        prices = pd.read_parquet(path)
    return pick_universe(prices)
