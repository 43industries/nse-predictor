#!/usr/bin/env python3
"""Download company favicons into data/external/logos/{TICKER}.png for the dashboard.

Uses public favicon endpoints (gstatic). Prefer replacing with licensed brand
assets when you have rights to use them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LOGOS = ROOT / "data" / "external" / "logos"
CSV_PATH = LOGOS / "companies.csv"

# Prefer these domains when the CSV entry fails or is weak
DOMAIN_OVERRIDES = {
    "EQTY": "equitygroupholdings.com",
    "KCB": "www.kcbgroup.com",
    "ABSA": "absa.africa",
    "HAFR": "hf.co.ke",
    "COOP": "co-opbank.co.ke",
}


def favicon_urls(domain: str) -> list[str]:
    domain = domain.removeprefix("https://").removeprefix("http://").strip("/")
    url = f"https://{domain}"
    return [
        (
            "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON"
            f"&fallback_opts=TYPE,SIZE,URL&url={url}&size=128"
        ),
        f"https://www.google.com/s2/favicons?domain={domain}&sz=128",
        f"https://icon.horse/icon/{domain}",
    ]


def fetch(url: str) -> bytes | None:
    req = Request(url, headers={"User-Agent": "nse-predictor-logo-fetch/1.0"})
    try:
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
            if not data or len(data) < 50:
                return None
            return data
    except (HTTPError, URLError, TimeoutError, OSError):
        return None


def main() -> int:
    LOGOS.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    ok = 0
    fail: list[str] = []
    for row in rows:
        code = (row.get("stock_code") or "").strip().upper()
        domain = DOMAIN_OVERRIDES.get(code) or (row.get("domain") or "").strip()
        if not code or not domain:
            continue
        dest = LOGOS / f"{code}.png"
        # Keep existing non-trivial local assets
        if dest.exists() and dest.stat().st_size > 200:
            print(f"skip {code} (exists)")
            ok += 1
            continue
        data = None
        used = None
        for url in favicon_urls(domain):
            data = fetch(url)
            if data:
                used = url
                break
        if not data:
            fail.append(code)
            print(f"FAIL {code} ({domain})")
            continue
        dest.write_bytes(data)
        ok += 1
        print(f"OK   {code} <- {used} ({len(data)} bytes)")

    print(f"done: {ok} saved, {len(fail)} failed: {fail}")
    return 0 if not fail else 0  # non-fatal; monogram covers gaps


if __name__ == "__main__":
    raise SystemExit(main())
