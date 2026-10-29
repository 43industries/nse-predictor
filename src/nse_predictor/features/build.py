"""Leakage-safe feature engineering for NSE daily panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def build_feature_matrix(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build features known as of each trade_date (no future leakage).

    Targets are computed from future returns but shifted so they align with as-of date:
    - y_dir_5d: 1 if price[t+5] > price[t], else 0
    - y_ret_5d, y_ret_20d: forward returns
    - y_vol_20d: forward 20-day realized vol of daily returns
    """
    df = prices.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if tickers:
        codes = {t.upper() for t in tickers}
        # Also keep indices for market context
        df = df[df["stock_code"].isin(codes) | df["stock_code"].str.contains("NSE|NASI", case=False, na=False)]

    df = df.sort_values(["stock_code", "trade_date"])

    # Market index return (NASI preferred; Wanjawa often uses ^NASI / ^N20I)
    idx_codes = {"NASI", "NSE20", "NSE 20", "NSE-20", "^NASI", "^N20I", "N20I"}
    idx = df[df["stock_code"].isin(idx_codes)].copy()
    if idx.empty:
        market = (
            df.groupby("trade_date")["price"]
            .mean()
            .pct_change()
            .rename("nasi_ret")
            .reset_index()
        )
    else:
        nasi = idx[idx["stock_code"].isin(["^NASI", "NASI"])]
        if nasi.empty:
            nasi = idx.drop_duplicates("trade_date", keep="first")
        else:
            nasi = nasi.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        market = nasi[["trade_date", "price"]].copy()
        market["nasi_ret"] = market["price"].pct_change()
        market = market[["trade_date", "nasi_ret"]]

    g = df.groupby("stock_code", group_keys=False)

    df["ret_1d"] = g["price"].pct_change()
    df["ma_5"] = g["price"].transform(lambda s: s.rolling(5).mean())
    df["ma_20"] = g["price"].transform(lambda s: s.rolling(20).mean())
    df["ma_50"] = g["price"].transform(lambda s: s.rolling(50).mean())
    df["ma5_ratio"] = df["price"] / df["ma_5"] - 1
    df["ma20_ratio"] = df["price"] / df["ma_20"] - 1
    df["ma50_ratio"] = df["price"] / df["ma_50"] - 1
    df["vol_20"] = g["ret_1d"].transform(lambda s: s.rolling(20).std())
    df["rsi_14"] = g["price"].transform(_rsi)
    macd_parts = []
    for code, part in df.groupby("stock_code"):
        m, sig = _macd(part["price"])
        tmp = part[["trade_date", "stock_code"]].copy()
        tmp["macd"] = m.values
        tmp["macd_signal"] = sig.values
        macd_parts.append(tmp)
    macd_df = pd.concat(macd_parts, ignore_index=True)
    df = df.merge(macd_df, on=["trade_date", "stock_code"], how="left")

    # 52-week distance using past window only
    df["high_252"] = g["price"].transform(lambda s: s.rolling(252, min_periods=20).max())
    df["low_252"] = g["price"].transform(lambda s: s.rolling(252, min_periods=20).min())
    df["dist_52w_high"] = df["price"] / df["high_252"] - 1
    df["dist_52w_low"] = df["price"] / df["low_252"] - 1

    vol_prev = g["volume"].shift(1)
    df["volume_chg"] = ((df["volume"] - vol_prev) / vol_prev.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    if "days_since_trade" not in df.columns:
        df["days_since_trade"] = 0

    df = df.merge(market, on="trade_date", how="left")

    if sectors is not None and not sectors.empty:
        sec = sectors[["stock_code", "market_sector"]].drop_duplicates("stock_code")
        df = df.merge(sec, on="stock_code", how="left")
        sector_ret = (
            df.groupby(["trade_date", "market_sector"])["ret_1d"]
            .mean()
            .rename("sector_ret")
            .reset_index()
        )
        df = df.merge(sector_ret, on=["trade_date", "market_sector"], how="left")
    else:
        df["sector_ret"] = np.nan

    if macro is not None and not macro.empty:
        m = macro.copy()
        m["observe_date"] = pd.to_datetime(m["observe_date"])
        m = m.sort_values("observe_date")
        m["cbk_chg"] = m["cbk_rate"].diff()
        m["usd_kes_chg"] = m["usd_kes"].pct_change()
        m["inflation_trend"] = m["inflation_yoy"].diff()
        df = pd.merge_asof(
            df.sort_values("trade_date"),
            m.rename(columns={"observe_date": "trade_date"}),
            on="trade_date",
            direction="backward",
        )

    df["dow"] = df["trade_date"].dt.dayofweek
    df["month"] = df["trade_date"].dt.month
    # Results season heuristic: Feb–Mar and Aug–Sep
    df["results_season"] = df["month"].isin([2, 3, 8, 9]).astype(int)

    # Re-bind groupby after merges
    g = df.groupby("stock_code", group_keys=False)

    # Targets (future info — never used as features)
    df["y_ret_5d"] = g["price"].transform(lambda s: s.shift(-5) / s - 1)
    df["y_ret_20d"] = g["price"].transform(lambda s: s.shift(-20) / s - 1)
    df["y_dir_5d"] = (df["y_ret_5d"] > 0).astype(float)
    df.loc[df["y_ret_5d"].isna(), "y_dir_5d"] = np.nan
    # Forward 20-day realized vol of daily returns starting tomorrow
    df["y_vol_20d"] = g["ret_1d"].transform(
        lambda s: s[::-1].rolling(20, min_periods=10).std()[::-1].shift(-1)
    )

    feature_cols = [
        "ret_1d",
        "ma5_ratio",
        "ma20_ratio",
        "ma50_ratio",
        "vol_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "dist_52w_high",
        "dist_52w_low",
        "volume_chg",
        "days_since_trade",
        "nasi_ret",
        "sector_ret",
        "cbk_rate",
        "cbk_chg",
        "tbill_91d",
        "usd_kes",
        "usd_kes_chg",
        "inflation_yoy",
        "inflation_trend",
        "lending_rate",
        "dow",
        "month",
        "results_season",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    meta = ["trade_date", "stock_code", "price", "volume", "y_dir_5d", "y_ret_5d", "y_ret_20d", "y_vol_20d"]
    out = df[meta + feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
