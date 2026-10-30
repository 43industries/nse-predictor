#!/usr/bin/env python3
"""Build data/processed/features_v2.parquet + features_v2.meta.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nse_predictor.config import settings
from nse_predictor.features.cache import load_or_build_features
from nse_predictor.ingest.fundamentals import load_fundamentals
from nse_predictor.universe import pick_universe, save_universe


def main() -> int:
    p = settings.data_processed_dir
    import pandas as pd

    prices = pd.read_parquet(p / "prices.parquet")
    macro = pd.read_parquet(p / "macro.parquet")
    sectors = pd.read_parquet(p / "sectors.parquet")
    universe = pick_universe(prices)
    save_universe(universe)
    fund = load_fundamentals()

    feats, info = load_or_build_features(
        prices, macro, sectors, universe, fundamentals=fund, force_rebuild=True
    )
    print(
        json.dumps(
            {
                **info,
                "n_rows": int(len(feats)),
                "fundamentals_status": feats.attrs.get("fundamentals_status"),
                "columns_sample": [c for c in ("xs_ret_1d", "rel_volume", "amihud_20") if c in feats.columns],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
