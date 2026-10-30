"""Leakage-safe feature engineering for NSE daily panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nse_predictor.features.registry import ALL_COMPUTED

DEAD_ZONE_XS: float = 0.01
NASI_CODES = ("^NASI", "NASI")
INDEX_EXCLUDE_RE = r"-R$|^\^|NASI|N20I|NSE20"
INDEX_CODES = {"NASI", "NSE20", "NSE 20", "NSE-20", "^NASI", "^N20I", "N20I"}
JUMP_ABS_RET = 0.40
META_COLS = [
    "trade_date",
    "stock_code",
    "price",
    "volume",
    "y_dir_5d",
    "y_ret_5d",
    "y_ret_5d_xs",
    "y_ret_20d",
    "y_vol_20d",
]


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
    return macd, macd.ewm(span=9, adjust=False).mean()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _nasi_price_series(df: pd.DataFrame) -> pd.DataFrame:
    nasi_src = df[df["stock_code"].isin(NASI_CODES)].sort_values("trade_date")
    if not nasi_src.empty:
        return (
            nasi_src.drop_duplicates("trade_date", keep="last")[["trade_date", "price"]]
            .rename(columns={"price": "nasi_price"})
            .sort_values("trade_date")
        )
    eq = ~df["stock_code"].astype(str).str.contains(
        INDEX_EXCLUDE_RE, case=False, na=False, regex=True
    )
    return (
        df.loc[eq]
        .groupby("trade_date", as_index=False)["price"]
        .mean()
        .rename(columns={"price": "nasi_price"})
        .sort_values("trade_date")
    )


def _market_panels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nasi_ret daily, nasi trailing feature panel)."""
    idx = df[df["stock_code"].isin(INDEX_CODES)]
    if idx.empty:
        ew = df.groupby("trade_date", as_index=False)["price"].mean()
        market = ew.assign(nasi_ret=ew["price"].pct_change())[["trade_date", "nasi_ret"]]
        nasi_panel = ew.rename(columns={"price": "nasi_price"})
    else:
        nasi = idx[idx["stock_code"].isin(NASI_CODES)]
        if nasi.empty:
            nasi = idx.drop_duplicates("trade_date", keep="first")
        else:
            nasi = nasi.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        nasi_panel = nasi[["trade_date", "price"]].rename(columns={"price": "nasi_price"})
        market = nasi_panel.assign(nasi_ret=nasi_panel["nasi_price"].pct_change())[
            ["trade_date", "nasi_ret"]
        ]

    nasi_panel = nasi_panel.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    px = nasi_panel["nasi_price"]
    nasi_panel["nasi_ret_trail_5d"] = px / px.shift(5) - 1
    nasi_panel["nasi_rsi_14"] = _rsi(px)
    nasi_panel["nasi_ma20_ratio"] = px / px.rolling(20).mean() - 1
    return market, nasi_panel


def _add_price_technicals(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stock_code", group_keys=False)
    df = df.copy()
    df["ret_1d"] = g["price"].pct_change()
    for span, col in ((5, "ma5_ratio"), (20, "ma20_ratio"), (50, "ma50_ratio")):
        ma = g["price"].transform(lambda s, n=span: s.rolling(n).mean())
        df[col] = df["price"] / ma - 1
    df["vol_20"] = g["ret_1d"].transform(lambda s: s.rolling(20).std())
    df["rsi_14"] = g["price"].transform(_rsi)

    macd_frames = []
    for _, part in df.groupby("stock_code", sort=False):
        m, sig = _macd(part["price"])
        macd_frames.append(
            part[["trade_date", "stock_code"]].assign(macd=m.values, macd_signal=sig.values)
        )
    df = df.merge(pd.concat(macd_frames, ignore_index=True), on=["trade_date", "stock_code"], how="left")

    g = df.groupby("stock_code", group_keys=False)
    high = g["price"].transform(lambda s: s.rolling(252, min_periods=20).max())
    low = g["price"].transform(lambda s: s.rolling(252, min_periods=20).min())
    df["dist_52w_high"] = df["price"] / high - 1
    df["dist_52w_low"] = df["price"] / low - 1
    return df


def _add_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stock_code", group_keys=False)
    df = df.copy()
    if "days_since_trade" not in df.columns:
        df["days_since_trade"] = 0
    prev_vol = g["volume"].shift(1)
    df["volume_chg"] = _safe_div(df["volume"] - prev_vol, prev_vol)
    df["adv_20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["rel_volume"] = _safe_div(df["volume"], df["adv_20"])
    day_amihud = _safe_div(df["ret_1d"].abs(), df["volume"])
    df["amihud_20"] = (
        df.assign(_a=day_amihud)
        .groupby("stock_code", group_keys=False)["_a"]
        .transform(lambda s: s.rolling(20, min_periods=5).mean())
    )
    df["zero_vol_streak"] = df["days_since_trade"]
    df["jump_flag_1d"] = (df["ret_1d"].abs() > JUMP_ABS_RET).astype(float)
    return df


def _add_sector_ret(df: pd.DataFrame, sectors: pd.DataFrame | None) -> pd.DataFrame:
    if sectors is None or sectors.empty:
        df = df.copy()
        df["sector_ret"] = np.nan
        return df
    sec = sectors[["stock_code", "market_sector"]].drop_duplicates("stock_code")
    df = df.merge(sec, on="stock_code", how="left")
    sector_ret = (
        df.groupby(["trade_date", "market_sector"])["ret_1d"].mean().rename("sector_ret").reset_index()
    )
    return df.merge(sector_ret, on=["trade_date", "market_sector"], how="left")


def _add_relative(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stock_code", group_keys=False)
    df = df.copy()
    trail_5 = g["price"].transform(lambda s: s / s.shift(5) - 1)
    df["xs_ret_1d"] = df["ret_1d"] - df["nasi_ret"]
    df["xs_ret_5d"] = trail_5 - df["nasi_ret_trail_5d"]
    df["sector_xs"] = df["sector_ret"] - df["nasi_ret"]
    df["stock_vs_sector_1d"] = df["ret_1d"] - df["sector_ret"]
    df["rsi_vs_nasi"] = df["rsi_14"] - df["nasi_rsi_14"]
    df["ma20_xs"] = df["ma20_ratio"] - df["nasi_ma20_ratio"]
    return df


def _add_macro(df: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    if macro is None or macro.empty:
        return df
    m = macro.copy()
    m["observe_date"] = pd.to_datetime(m["observe_date"])
    m = m.sort_values("observe_date")
    m["cbk_chg"] = m["cbk_rate"].diff()
    m["usd_kes_chg"] = m["usd_kes"].pct_change()
    m["inflation_trend"] = m["inflation_yoy"].diff()
    return pd.merge_asof(
        df.sort_values("trade_date"),
        m.rename(columns={"observe_date": "trade_date"}),
        on="trade_date",
        direction="backward",
    )


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["trade_date"].dt.dayofweek
    df["month"] = df["trade_date"].dt.month
    df["results_season"] = df["month"].isin([2, 3, 8, 9]).astype(int)
    return df


def _attach_excess_targets(df: pd.DataFrame, dead_zone_xs: float) -> pd.DataFrame:
    nasi_px = _nasi_price_series(df).copy()
    nasi_px["nasi_ret_5d"] = nasi_px["nasi_price"].shift(-5) / nasi_px["nasi_price"] - 1
    out = df.merge(nasi_px[["trade_date", "nasi_ret_5d"]], on="trade_date", how="left")
    out["y_ret_5d_xs"] = out["y_ret_5d"] - out["nasi_ret_5d"]
    xs = out["y_ret_5d_xs"]
    out["y_dir_5d"] = np.where(
        xs > dead_zone_xs,
        1.0,
        np.where(xs < -dead_zone_xs, 0.0, np.nan),
    )
    out.loc[xs.isna() | out["y_ret_5d"].isna() | out["nasi_ret_5d"].isna(), "y_dir_5d"] = np.nan
    return out


def _add_targets(df: pd.DataFrame, dead_zone_xs: float) -> pd.DataFrame:
    g = df.groupby("stock_code", group_keys=False)
    df = df.copy()
    df["y_ret_5d"] = g["price"].transform(lambda s: s.shift(-5) / s - 1)
    df["y_ret_20d"] = g["price"].transform(lambda s: s.shift(-20) / s - 1)
    df = _attach_excess_targets(df, dead_zone_xs=dead_zone_xs)
    g = df.groupby("stock_code", group_keys=False)
    df["y_vol_20d"] = g["ret_1d"].transform(
        lambda s: s[::-1].rolling(20, min_periods=10).std()[::-1].shift(-1)
    )
    return df


def _maybe_join_fundamentals(
    df: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    min_fund_coverage: float,
) -> tuple[pd.DataFrame, bool, str]:
    if fundamentals is None or fundamentals.empty:
        return df, False, "not_provided"

    from nse_predictor.ingest.fundamentals import (
        fundamentals_coverage,
        join_fundamentals_asof,
    )

    fund_codes = set(fundamentals["stock_code"].astype(str).str.upper())
    panel_codes = set(df["stock_code"].astype(str).str.upper())
    overlap = len(fund_codes & panel_codes) / max(len(panel_codes), 1)
    if overlap < min_fund_coverage:
        status = f"skipped low ticker overlap={overlap:.1%} < {min_fund_coverage:.0%}"
        print(f"fundamentals: {status}")
        return df, False, status

    df = join_fundamentals_asof(df, fundamentals)
    cov = fundamentals_coverage(df)
    if cov >= min_fund_coverage:
        return df, True, f"included coverage={cov:.1%}"
    status = f"skipped low coverage={cov:.1%} < {min_fund_coverage:.0%}"
    print(f"fundamentals: {status}")
    return df, False, status


def build_feature_matrix(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sectors: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    dead_zone_xs: float = DEAD_ZONE_XS,
    fundamentals: pd.DataFrame | None = None,
    min_fund_coverage: float = 0.30,
    macro_source: str = "synthetic",
) -> pd.DataFrame:
    """Build features known as of each trade_date (no future leakage)."""
    df = prices.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if tickers:
        codes = {t.upper() for t in tickers}
        df = df[
            df["stock_code"].isin(codes)
            | df["stock_code"].str.contains("NSE|NASI", case=False, na=False)
        ]
    df = df.sort_values(["stock_code", "trade_date"])

    market, nasi_panel = _market_panels(df)
    df = _add_price_technicals(df)
    df = _add_liquidity(df)
    df = df.merge(market, on="trade_date", how="left")
    df = df.merge(
        nasi_panel[["trade_date", "nasi_ret_trail_5d", "nasi_rsi_14", "nasi_ma20_ratio"]],
        on="trade_date",
        how="left",
    )
    df = _add_sector_ret(df, sectors)
    df = _add_relative(df)
    df = _add_macro(df, macro)
    df = _add_calendar(df)
    df = _add_targets(df, dead_zone_xs)
    df, fund_included, fund_status = _maybe_join_fundamentals(df, fundamentals, min_fund_coverage)

    for c in ALL_COMPUTED:
        if c not in df.columns:
            df[c] = np.nan

    out = df[META_COLS + list(ALL_COMPUTED)].copy().replace([np.inf, -np.inf], np.nan)
    out.attrs["fundamentals_status"] = fund_status
    out.attrs["fundamentals_included"] = fund_included
    out.attrs["macro_source"] = macro_source
    return out
