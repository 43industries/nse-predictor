"""Leakage-safe news sentiment features from data/external/news.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nse_predictor.config import settings

_SENT_MAP = {
    "positive": 1.0,
    "pos": 1.0,
    "bullish": 1.0,
    "negative": -1.0,
    "neg": -1.0,
    "bearish": -1.0,
    "neutral": 0.0,
    "neu": 0.0,
    "": 0.0,
}

_POS = ("gain", "rise", "profit", "growth", "beat", "surge", "strong", "dividend", "upgrade")
_NEG = ("loss", "fall", "drop", "weak", "miss", "cut", "fraud", "probe", "downgrade", "loss-making")


def _lexicon_score(headline: str) -> float:
    h = str(headline or "").lower()
    if not h.strip():
        return 0.0
    pos = sum(1 for w in _POS if w in h)
    neg = sum(1 for w in _NEG if w in h)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(pos + neg, 1)


def load_news_events(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or (settings.data_external_dir / "news.csv"))
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["stock_code", "event_date", "sentiment_score"])
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "stock_code" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["stock_code", "event_date", "sentiment_score"])
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["event_date"] = pd.to_datetime(df["date"], errors="coerce")
    if "sentiment" in df.columns:
        raw = df["sentiment"].fillna("").astype(str).str.strip().str.lower()
        score = raw.map(_SENT_MAP)
        missing = score.isna()
        if "headline" in df.columns:
            score = score.where(~missing, df["headline"].map(_lexicon_score))
        else:
            score = score.fillna(0.0)
    elif "headline" in df.columns:
        score = df["headline"].map(_lexicon_score)
    else:
        score = 0.0
    df["sentiment_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    out = (
        df.dropna(subset=["stock_code", "event_date"])
        .groupby(["stock_code", "event_date"], as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), news_count=("sentiment_score", "size"))
    )
    return out.sort_values(["stock_code", "event_date"]).reset_index(drop=True)


def join_news_features(panel: pd.DataFrame, news: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach news_sentiment_1d / 5d and news_count_5d (no future news)."""
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    for c in ("news_sentiment_1d", "news_sentiment_5d", "news_count_5d"):
        out[c] = np.nan
    events = news if news is not None else load_news_events()
    if events is None or events.empty:
        out["news_sentiment_1d"] = 0.0
        out["news_sentiment_5d"] = 0.0
        out["news_count_5d"] = 0.0
        return out

    frames: list[pd.DataFrame] = []
    by_code = {c: g for c, g in events.groupby("stock_code", sort=False)}
    for code, part in out.groupby("stock_code", sort=False):
        p = part.sort_values("trade_date")
        ev = by_code.get(str(code).upper())
        if ev is None or ev.empty:
            frames.append(
                p.assign(
                    news_sentiment_1d=0.0,
                    news_sentiment_5d=0.0,
                    news_count_5d=0.0,
                )
            )
            continue
        daily = (
            ev.rename(columns={"event_date": "trade_date"})
            .sort_values("trade_date")
            .drop_duplicates("trade_date", keep="last")
        )
        merged = pd.merge_asof(
            p,
            daily[["trade_date", "sentiment_score", "news_count"]],
            on="trade_date",
            direction="backward",
            tolerance=pd.Timedelta(days=0),
        )
        # exact-day sentiment; 0 if no headline that day
        merged["news_sentiment_1d"] = merged["sentiment_score"].fillna(0.0)
        # rebuild 5d rolling from a calendar join of event days onto panel dates
        cal = p[["trade_date"]].drop_duplicates().sort_values("trade_date")
        cal = cal.merge(
            daily[["trade_date", "sentiment_score", "news_count"]],
            on="trade_date",
            how="left",
        )
        cal["sentiment_score"] = cal["sentiment_score"].fillna(0.0)
        cal["news_count"] = cal["news_count"].fillna(0.0)
        cal["news_sentiment_5d"] = cal["sentiment_score"].rolling(5, min_periods=1).mean()
        cal["news_count_5d"] = cal["news_count"].rolling(5, min_periods=1).sum()
        merged = merged.drop(columns=["sentiment_score", "news_count"], errors="ignore")
        merged = merged.merge(
            cal[["trade_date", "news_sentiment_5d", "news_count_5d"]],
            on="trade_date",
            how="left",
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)
