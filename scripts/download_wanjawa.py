#!/usr/bin/env python3
"""Download Barack Wanjawa NSE Mendeley datasets (2007–2025) into data/raw/nse/.

Uses Mendeley public-api metadata + per-file download_url (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nse"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
HEADERS = {"User-Agent": "nse-predictor/0.1 (research; CC-BY attribution to Wanjawa)"}

DATASETS = [
    {"id": "5hk4zw32f5", "version": 1, "doi": "10.17632/5hk4zw32f5.1", "years": "2007-2012"},
    {"id": "73rb78pmzw", "version": 2, "doi": "10.17632/73rb78pmzw.2", "years": "2013-2020"},
    {
        "id": "97hkwn5y3x",
        "version": 5,
        "doi": "10.17632/97hkwn5y3x.5",
        "years": "2021",
        "version_fallbacks": [4, 3],
    },
    {"id": "jmcdmnyh2s", "version": 2, "doi": "10.17632/jmcdmnyh2s.2", "years": "2022"},
    {"id": "ss5pfw8xnk", "version": 2, "doi": "10.17632/ss5pfw8xnk.2", "years": "2023-2024"},
    {
        "id": "2b63rx67xt",
        "version": 2,
        "doi": "10.17632/2b63rx67xt.2",
        "years": "2025",
        "version_fallbacks": [1],
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_dataset_meta(dataset_id: str, version: int | None = None) -> dict:
    url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
    if version is not None:
        url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}/versions/{version}"
    r = requests.get(url, timeout=60, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def resolve_meta(ds: dict) -> dict:
    versions = [ds["version"]] + ds.get("version_fallbacks", [])
    last_err: Exception | None = None
    for ver in versions:
        try:
            # Prefer versioned endpoint; fall back to latest
            try:
                meta = fetch_dataset_meta(ds["id"], ver)
            except requests.HTTPError:
                meta = fetch_dataset_meta(ds["id"])
                if int(meta.get("version", -1)) != int(ver) and ver != ds["version"]:
                    # keep trying fallbacks
                    continue
            if meta.get("files"):
                meta["_resolved_version"] = meta.get("version", ver)
                return meta
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err:
        raise last_err
    raise RuntimeError(f"No files for {ds['id']}")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300, headers=HEADERS, allow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as out:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    out.write(chunk)


def download_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        out_dir = RAW_DIR / ds["id"]
        existing = list(out_dir.glob("*.csv")) if out_dir.exists() else []
        if existing and not force:
            print(f"[skip] {ds['years']} already has {len(existing)} csvs in {out_dir}")
            continue
        print(f"[meta] {ds['years']} {ds['id']}")
        meta = resolve_meta(ds)
        print(f"[ok] version={meta.get('_resolved_version')} files={len(meta.get('files', []))}")
        for fobj in meta.get("files", []):
            name = fobj["filename"]
            url = fobj["content_details"]["download_url"]
            dest = out_dir / name
            if dest.exists() and not force:
                print(f"  [skip] {name}")
                continue
            print(f"  [download] {name}")
            download_file(url, dest)


def _parse_dates_from_csv(path: Path) -> dict:
    import pandas as pd

    sample = pd.read_csv(path, nrows=5)
    date_col = sample.columns[0]
    for c in sample.columns:
        if str(c).strip().lower() in {"date", "trade_date", "dated"}:
            date_col = c
            break
    dates = pd.read_csv(path, usecols=[date_col])[date_col]
    parsed = pd.to_datetime(dates, errors="coerce", dayfirst=False)
    return {
        "rows": int(parsed.notna().sum()),
        "min_date": None if parsed.isna().all() else str(parsed.min().date()),
        "max_date": None if parsed.isna().all() else str(parsed.max().date()),
    }


def build_manifest(raw_dir: Path) -> dict:
    csv_files = sorted(raw_dir.rglob("*.csv"))
    files_meta = []
    year_ranges: dict = {}
    for p in csv_files:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        entry = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
        try:
            entry.update(_parse_dates_from_csv(p))
            if "sector" not in p.name.lower():
                m = re.search(r"(20\d{2})", p.name)
                if m:
                    year_ranges[m.group(1)] = {
                        "min_date": entry.get("min_date"),
                        "max_date": entry.get("max_date"),
                        "rows": entry.get("rows"),
                        "file": rel,
                    }
        except Exception as exc:  # noqa: BLE001
            entry["parse_error"] = str(exc)
        files_meta.append(entry)

    gap_2020 = None
    if "2020" in year_ranges:
        max_d = year_ranges["2020"].get("max_date")
        gap_2020 = {
            "max_date": max_d,
            "h2_missing": bool(max_d and max_d < "2020-07-01"),
            "note": "If h2_missing, fill Jul–Dec 2020 before treating the panel as continuous.",
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0",
        "attribution": "Barack Wanjawa – Nairobi Securities Exchange daily prices (Mendeley Data)",
        "datasets": DATASETS,
        "files": files_meta,
        "year_ranges": year_ranges,
        "coverage_2020": gap_2020,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.manifest_only:
        try:
            download_all(force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] download failed: {exc}", file=sys.stderr)
            return 1

    manifest = build_manifest(RAW_DIR)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {MANIFEST_PATH} ({len(manifest.get('files', []))} files)")
    if manifest.get("coverage_2020"):
        print(f"[coverage_2020] {json.dumps(manifest['coverage_2020'])}")
    if not manifest.get("files"):
        print("[warn] no CSVs found — run generate_sample_panel.py for offline testing")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
