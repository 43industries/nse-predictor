"""Local paper-trading intention journal (human intent only)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings

JOURNAL_COLS = [
    "logged_at",
    "as_of_date",
    "stock_code",
    "stance",
    "prob_up",
    "pred_label",
    "price",
    "note",
    "source",
]

VALID_STANCES = frozenset({"long", "flat", "skip"})


def journal_path(processed: Path | None = None) -> Path:
    return Path(processed or settings.data_processed_dir) / "paper_journal.parquet"


def load_journal(processed: Path | None = None) -> pd.DataFrame:
    path = journal_path(processed)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=JOURNAL_COLS)
    df = pd.read_parquet(path)
    for c in JOURNAL_COLS:
        if c not in df.columns:
            df[c] = None
    return df[JOURNAL_COLS].copy()


def append_entry(
    *,
    stock_code: str,
    stance: str,
    as_of_date: str | None = None,
    prob_up: float | None = None,
    pred_label: str | None = None,
    price: float | None = None,
    note: str = "",
    source: str = "desk_brief",
    processed: Path | None = None,
) -> pd.DataFrame:
    """Append one journal row; dedupe on as_of/stock/stance/logged_at keep last."""
    stance_n = str(stance).strip().lower()
    if stance_n not in VALID_STANCES:
        raise ValueError(f"stance must be one of {sorted(VALID_STANCES)}")

    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(as_of_date) if as_of_date else None,
        "stock_code": str(stock_code).strip().upper(),
        "stance": stance_n,
        "prob_up": float(prob_up) if prob_up is not None and prob_up == prob_up else None,
        "pred_label": str(pred_label).lower() if pred_label else None,
        "price": float(price) if price is not None and price == price else None,
        "note": str(note or ""),
        "source": str(source or "desk_brief"),
    }
    prev = load_journal(processed)
    both = pd.concat([prev, pd.DataFrame([row], columns=JOURNAL_COLS)], ignore_index=True)
    both = both.drop_duplicates(
        subset=["as_of_date", "stock_code", "stance", "logged_at"],
        keep="last",
    )
    path = journal_path(processed)
    path.parent.mkdir(parents=True, exist_ok=True)
    both.to_parquet(path, index=False)
    return both
