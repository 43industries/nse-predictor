"""Versioned point-in-time feature cache (features_v2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from nse_predictor.config import settings
from nse_predictor.features.build import DEAD_ZONE_XS, build_feature_matrix
from nse_predictor.features.registry import REGISTRY_VERSION
from nse_predictor.ingest.fundamentals import resolve_fundamentals_path

CACHE_NAME = "features_v2"
CACHE_PARQUET = f"{CACHE_NAME}.parquet"
CACHE_META = f"{CACHE_NAME}.meta.json"


def _mtime(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return path.stat().st_mtime


def load_macro_source(processed_dir: Path) -> str:
    meta_path = processed_dir / "ingest_meta.json"
    if not meta_path.exists():
        return "synthetic"
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("macro_source", "synthetic")
    except Exception:  # noqa: BLE001
        return "synthetic"


def compute_inputs_hash(
    prices_path: Path,
    macro_path: Path,
    sectors_path: Path,
    fundamentals_path: Path | None,
    dead_zone_xs: float,
    universe: list[str],
    macro_source: str,
) -> str:
    payload = {
        "prices_mtime": _mtime(prices_path),
        "macro_mtime": _mtime(macro_path),
        "sectors_mtime": _mtime(sectors_path),
        "fundamentals_mtime": _mtime(fundamentals_path),
        "fundamentals_path": str(fundamentals_path) if fundamentals_path else "",
        "dead_zone_xs": dead_zone_xs,
        "universe": sorted(universe),
        "macro_source": macro_source,
        "registry_version": REGISTRY_VERSION,
        "version": CACHE_NAME,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_paths(processed_dir: Path | None = None) -> tuple[Path, Path]:
    processed = Path(processed_dir or settings.data_processed_dir)
    return processed / CACHE_PARQUET, processed / CACHE_META


def meta_matches(meta_path: Path, expected_hash: str) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return meta.get("inputs_hash") == expected_hash


def load_or_build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    sectors: pd.DataFrame,
    universe: list[str],
    fundamentals: pd.DataFrame | None = None,
    dead_zone_xs: float = DEAD_ZONE_XS,
    processed_dir: Path | None = None,
    force_rebuild: bool = False,
    macro_source: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return feature matrix; load cache if meta hash matches else rebuild + refresh."""
    processed = Path(processed_dir or settings.data_processed_dir)
    parquet_path, meta_path = cache_paths(processed)
    prices_path = processed / "prices.parquet"
    macro_path = processed / "macro.parquet"
    sectors_path = processed / "sectors.parquet"
    fund_path = resolve_fundamentals_path()
    src = macro_source or load_macro_source(processed)

    expected = compute_inputs_hash(
        prices_path,
        macro_path,
        sectors_path,
        fund_path,
        dead_zone_xs,
        universe,
        src,
    )
    info: dict = {
        "cache_hit": False,
        "inputs_hash": expected,
        "path": str(parquet_path),
        "macro_source": src,
        "cache_version": CACHE_NAME,
    }

    if not force_rebuild and parquet_path.exists() and meta_matches(meta_path, expected):
        feats = pd.read_parquet(parquet_path)
        info["cache_hit"] = True
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for key in (
                    "fundamentals_status",
                    "fundamentals_included",
                    "macro_source",
                ):
                    if key in meta:
                        info[key] = meta[key]
                        feats.attrs[key] = meta[key]
            except Exception:  # noqa: BLE001
                pass
        feats.attrs.setdefault("macro_source", src)
        return feats, info

    feats = build_feature_matrix(
        prices,
        macro,
        sectors,
        tickers=universe,
        dead_zone_xs=dead_zone_xs,
        fundamentals=fundamentals,
        macro_source=src,
    )
    fund_status = feats.attrs.get("fundamentals_status")
    fund_included = bool(feats.attrs.get("fundamentals_included"))
    processed.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(parquet_path, index=False)
    meta = {
        "version": CACHE_NAME,
        "registry_version": REGISTRY_VERSION,
        "inputs_hash": expected,
        "dead_zone_xs": dead_zone_xs,
        "universe": list(universe),
        "macro_source": src,
        "n_rows": int(len(feats)),
        "fundamentals_status": fund_status,
        "fundamentals_included": fund_included,
        "invalidation": (
            "Any change to prices/macro/sectors/fundamentals mtime, "
            "dead_zone, universe, macro_source, or registry version → rebuild"
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    info["cache_hit"] = False
    info["n_rows"] = int(len(feats))
    info["fundamentals_status"] = fund_status
    info["fundamentals_included"] = fund_included
    return feats, info
