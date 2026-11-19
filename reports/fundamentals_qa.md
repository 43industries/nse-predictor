# Fundamentals QA

Curated fundamentals for the investable NSE panel (manual / published results — **no website scrape**).

## Coverage

- Source file: `data/external/fundamentals.csv`
- Template: `data/external/fundamentals_template.csv`
- Includes banks (EQTY, KCB, COOP, ABSA, NCBA, HFCK, …) with FY2025 seed metrics where available
- Extended ordinary names: SCOM, EABL, BRIT, CIC, KENO, SCAN, NSE, KEGN, KNRE, KPLC, CTUM, KQ

## Notes

- Pre-2022 history remains thin for many names — `full_gated` features are often missing in early train years (imputer drops all-null columns).
- Bank peer medians (`roe_vs_bank_med`, etc.) only activate when peer rows exist on/near the as-of date.
- Refresh: edit `fundamentals.csv`, then `python scripts/build_feature_cache.py` and retrain.

## Licensing

Do not scrape kenyanbanks.com or other licensed sites from this repo. Paste published FY/HY figures manually into the CSV.
