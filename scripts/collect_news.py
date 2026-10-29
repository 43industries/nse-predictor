#!/usr/bin/env python3
"""Drop-folder news collector (no website scrape).

Place CSVs in data/external/news/incoming/ with columns:
  stock_code, date, headline, source, url [, sentiment]

Appends into data/external/news.csv (deduped by stock_code+date+headline).
Processed files move to data/external/news/incoming/archive/.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEWS_DIR = ROOT / "data" / "external" / "news"
INCOMING = NEWS_DIR / "incoming"
ARCHIVE = INCOMING / "archive"
MASTER = ROOT / "data" / "external" / "news.csv"
STATUS = ROOT / "data" / "processed" / "news_collection_status.json"

REQUIRED = {"stock_code", "date", "headline"}
OPTIONAL = {"source", "url", "sentiment"}
OUT_COLS = ["stock_code", "date", "headline", "source", "url", "sentiment"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "ticker": "stock_code",
        "code": "stock_code",
        "title": "headline",
        "published": "date",
        "pub_date": "date",
        "link": "url",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = ""
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    raw_dates = df["date"]
    parsed = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    still_bad = parsed.isna()
    if still_bad.any():
        parsed.loc[still_bad] = pd.to_datetime(
            raw_dates.loc[still_bad], errors="coerce", dayfirst=True
        )
    df["date"] = parsed
    df = df.dropna(subset=["date", "stock_code", "headline"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"] != ""]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["sentiment"] = (
        df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )
    return df[OUT_COLS]


def collect_news() -> dict:
    INCOMING.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(INCOMING.glob("*.csv"))
    errors: list[str] = []
    processed: list[str] = []
    frames: list[pd.DataFrame] = []

    if not files:
        status = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "rows_added": 0,
            "files_processed": [],
            "errors": [],
            "note": "No CSV files in data/external/news/incoming/",
            "master": str(MASTER.relative_to(ROOT)).replace("\\", "/"),
        }
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    for path in files:
        try:
            raw = pd.read_csv(path)
            frames.append(_normalize(raw))
            processed.append(path.name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")

    rows_added = 0
    if frames:
        batch = pd.concat(frames, ignore_index=True)
        batch["_key"] = (
            batch["stock_code"] + "|" + batch["date"] + "|" + batch["headline"].str.lower()
        )
        if MASTER.exists():
            existing = pd.read_csv(MASTER)
            for c in OUT_COLS:
                if c not in existing.columns:
                    existing[c] = ""
            existing = existing[OUT_COLS]
            existing["_key"] = (
                existing["stock_code"].astype(str).str.upper()
                + "|"
                + existing["date"].astype(str)
                + "|"
                + existing["headline"].astype(str).str.lower()
            )
            fresh = batch[~batch["_key"].isin(set(existing["_key"]))].drop(columns=["_key"])
            combined = pd.concat([existing.drop(columns=["_key"]), fresh], ignore_index=True)
            rows_added = len(fresh)
        else:
            combined = batch.drop(columns=["_key"])
            rows_added = len(combined)
        combined = combined.sort_values(["date", "stock_code"], ascending=[False, True])
        combined.to_csv(MASTER, index=False)

    for name in processed:
        src = INCOMING / name
        if src.exists():
            dest = ARCHIVE / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{name}"
            shutil.move(str(src), str(dest))

    status = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "rows_added": int(rows_added),
        "files_processed": processed,
        "errors": errors,
        "note": None if processed else "No valid CSVs processed",
        "master": str(MASTER.relative_to(ROOT)).replace("\\", "/"),
        "total_rows": int(len(pd.read_csv(MASTER))) if MASTER.exists() else 0,
    }
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> int:
    status = collect_news()
    print(json.dumps(status, indent=2))
    return 1 if status.get("errors") and not status.get("files_processed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
