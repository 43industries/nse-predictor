"""Single source of truth for model feature groups (charter-aligned)."""

from __future__ import annotations

from typing import Iterable

REGISTRY_VERSION = "v2.0"

CORE_PRICE: list[str] = [
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
    "nasi_ret",
    "sector_ret",
]
RELATIVE: list[str] = [
    "xs_ret_1d",
    "xs_ret_5d",
    "sector_xs",
    "stock_vs_sector_1d",
    "rsi_vs_nasi",
    "ma20_xs",
]
LIQUIDITY: list[str] = [
    "volume_chg",
    "days_since_trade",
    "adv_20",
    "rel_volume",
    "amihud_20",
    "zero_vol_streak",
    "jump_flag_1d",
]
CALENDAR: list[str] = ["dow", "month", "results_season"]
MACRO_OFFICIAL: list[str] = [
    "cbk_rate",
    "cbk_chg",
    "tbill_91d",
    "usd_kes",
    "usd_kes_chg",
    "inflation_yoy",
    "inflation_trend",
    "lending_rate",
]
FUNDAMENTALS: list[str] = [
    "eps",
    "eps_yoy",
    "div_yield_proxy",
    "book_value",
    "book_to_price",
]

GROUPS: dict[str, list[str]] = {
    "CORE_PRICE": CORE_PRICE,
    "RELATIVE": RELATIVE,
    "LIQUIDITY": LIQUIDITY,
    "CALENDAR": CALENDAR,
    "MACRO_OFFICIAL": MACRO_OFFICIAL,
    "FUNDAMENTALS": FUNDAMENTALS,
}

ALL_COMPUTED: list[str] = [c for cols in GROUPS.values() for c in cols]

_FEATURE_DEFS: dict[str, str] = {
    "ret_1d": "1d price return",
    "ma5_ratio": "price/MA5 - 1",
    "ma20_ratio": "price/MA20 - 1",
    "ma50_ratio": "price/MA50 - 1",
    "vol_20": "20d std of ret_1d",
    "rsi_14": "14d RSI",
    "macd": "MACD line",
    "macd_signal": "MACD signal",
    "dist_52w_high": "distance to 252d high",
    "dist_52w_low": "distance to 252d low",
    "nasi_ret": "1d NASI return (past)",
    "sector_ret": "same-day sector mean ret_1d",
    "xs_ret_1d": "ret_1d - nasi_ret",
    "xs_ret_5d": "trailing 5d stock - NASI return",
    "sector_xs": "sector_ret - nasi_ret",
    "stock_vs_sector_1d": "ret_1d - sector_ret",
    "rsi_vs_nasi": "stock RSI14 - NASI RSI14",
    "ma20_xs": "stock ma20_ratio - NASI ma20_ratio",
    "volume_chg": "volume vs prior day",
    "days_since_trade": "zero-volume streak",
    "adv_20": "20d mean volume",
    "rel_volume": "volume / adv_20",
    "amihud_20": "20d mean |ret|/volume",
    "zero_vol_streak": "alias of days_since_trade",
    "jump_flag_1d": "1 if |ret_1d| > 40%",
    "dow": "day of week",
    "month": "calendar month",
    "results_season": "Feb–Mar / Aug–Sep flag",
    "cbk_rate": "CBR level (official only)",
    "cbk_chg": "CBR change",
    "tbill_91d": "91d T-bill",
    "usd_kes": "USD/KES level",
    "usd_kes_chg": "USD/KES pct change",
    "inflation_yoy": "CPI YoY",
    "inflation_trend": "inflation diff",
    "lending_rate": "lending rate",
    "eps": "as-of EPS",
    "eps_yoy": "EPS YoY change",
    "div_yield_proxy": "dividend / price",
    "book_value": "book value / share",
    "book_to_price": "book_value / price",
}

FEATURE_DOCS: dict[str, dict[str, str]] = {
    feat: {"group": group, "def": _FEATURE_DEFS.get(feat, "")}
    for group, cols in GROUPS.items()
    for feat in cols
}

ABLATION_PRESETS: dict[str, tuple[str, ...]] = {
    "core_only": ("core_only",),
    "core_relative": ("core", "relative"),
    "full_gated": ("full_gated",),
}


def is_official_macro(macro_source: str | None) -> bool:
    return (macro_source or "").strip().lower() == "official"


def _unique(cols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def model_feature_cols(
    macro_source: str | None = "synthetic",
    fund_included: bool = False,
    groups: Iterable[str] | None = None,
) -> list[str]:
    """Feature columns eligible for LightGBM given gates."""
    gset = {g.lower() for g in groups} if groups is not None else {"full"}

    if "full" in gset or "full_gated" in gset:
        cols = CORE_PRICE + RELATIVE + LIQUIDITY + CALENDAR
        if is_official_macro(macro_source):
            cols = cols + MACRO_OFFICIAL
        if fund_included:
            cols = cols + FUNDAMENTALS
        return list(cols)

    cols: list[str] = []
    wants_core = bool(gset & {"core", "core_only", "core_price"})
    if wants_core or "relative" in gset:
        cols += CORE_PRICE + CALENDAR
    if "relative" in gset:
        cols += RELATIVE
    if "liquidity" in gset:
        cols += LIQUIDITY
    if "macro" in gset and is_official_macro(macro_source):
        cols += MACRO_OFFICIAL
    if "fundamentals" in gset and fund_included:
        cols += FUNDAMENTALS
    return _unique(cols)


def available_features(
    df,
    macro_source: str | None = None,
    fund_included: bool | None = None,
    feature_cols: list[str] | None = None,
) -> list[str]:
    """Intersect requested model cols with columns present on ``df``."""
    if feature_cols is None:
        if fund_included is None:
            fund_included = any(
                c in df.columns and df[c].notna().any() for c in FUNDAMENTALS
            )
        if macro_source is None:
            macro_source = df.attrs.get("macro_source", "synthetic")
        feature_cols = model_feature_cols(
            macro_source=macro_source, fund_included=bool(fund_included)
        )
    return [c for c in feature_cols if c in df.columns]
