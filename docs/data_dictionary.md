# NSE Predictor – Data Dictionary

## Disclaimer

Forecasts and datasets in this project are for **internal research and decision support only**. They are **not investment advice**.

## Source: Wanjawa NSE daily prices (Mendeley Data, CC BY 4.0)

Contributor: Barack Wanjawa. Historical daily prices scraped from MyStocks (NSE-licensed public site).

| Years | DOI | Notes |
| --- | --- | --- |
| 2007–2012 | 10.17632/5hk4zw32f5.1 | Six yearly CSVs |
| 2013–2020 | 10.17632/73rb78pmzw.2 | Verify 2020 H2 coverage |
| 2021 | 10.17632/97hkwn5y3x | Use latest version (full year) |
| 2022 | 10.17632/jmcdmnyh2s.2 | |
| 2023–2024 | 10.17632/ss5pfw8xnk.2 | |
| 2025 | 10.17632/2b63rx67xt.2 | Full year |

### Raw CSV columns (`prices`)

| Column | Type | Description |
| --- | --- | --- |
| Date / trade_date | date | Trading session date |
| Code / stock_code | text | Ticker symbol |
| Name / stock_name | text | Company or index name |
| 12-month Low / low_12m | float | Trailing 12-month low |
| 12-month High / high_12m | float | Trailing 12-month high |
| Day's Low / day_low | float | Session low |
| Day's High / day_high | float | Session high |
| Day's Final Price / day_close | float | Closing / final price |
| Previous / prev_price | float | Previous session price |
| Change / change_value | float | Absolute change |
| Change % / change_pct | float | Percent change |
| Volume / volume | float | Shares traded |
| Adjusted / adj_price | float | Adjusted price if provided |

### Sector map

| Column | Description |
| --- | --- |
| market_sector | NSE sector classification |
| stock_code | Ticker |
| stock_name | Name |

### Macro (`macro`)

Place official series in `data/external/macro_cbk.csv` (same columns). Ingest records `macro_source` as `official` | `synthetic` in `data/processed/ingest_meta.json` and QA/train reports. Official requires a valid file **plus** marker file `macro_cbk.csv.official` (CSV path + `.official`). Without the marker, populated approx files stay labeled synthetic (no silent mix).

| Column | Description |
| --- | --- |
| observe_date | Observation date |
| cbk_rate | Central Bank Rate (%) |
| tbill_91d | 91-day T-bill yield (%) |
| usd_kes | USD/KES exchange rate |
| inflation_yoy | YoY inflation (%) |
| lending_rate | Commercial lending rate (%) |

**Where to get official data (manual download — no scraper in-repo):**

- Central Bank of Kenya statistics / monetary policy (CBR, T-bill yields, lending rates)
- CBK or published FX tables for USD/KES
- KNBS for CPI / inflation YoY

**Leakage rule:** join with `merge_asof(..., direction="backward")` so each trading day only sees the latest macro observation **already published** as of that date. Do not use revised future prints of the same vintage.

Empty official template: `python scripts/build_macro_template.py`. Approximate anchors (still synthetic): `python scripts/build_macro_template.py --approx`.

### Fundamentals (`data/external/fundamentals.csv`)

Point-in-time company fundamentals. Expand beyond `fundamentals_sample.csv`.

| Column | Description |
| --- | --- |
| stock_code | Ticker |
| report_date | Period end date |
| publish_date | When the print was knowable (preferred as-of); falls back to report_date |
| eps | Earnings per share |
| dividend | Dividend per share for the period |
| book_value | Book value per share |
| period | Label e.g. FY2023, H1-2024 |

**Leakage rule:** `merge_asof` on `publish_date` (fallback `report_date`) backward to `trade_date`. Derived features `eps_yoy`, `div_yield_proxy` are included only if non-null coverage ≥ 30% of rows; otherwise train logs “skipped low coverage”.

### Feature cache (`features_v2`)

`python scripts/build_feature_cache.py` writes `data/processed/features_v2.parquet` + `features_v2.meta.json` (hash of prices/macro/sectors/fundamentals mtimes, dead_zone, universe, `macro_source`, registry version). Train/score load the cache when the meta hash matches; any input change forces rebuild.

### Model feature registry (`features/registry.py`)

Single source of truth. Groups: `CORE_PRICE`, `RELATIVE`, `LIQUIDITY`, `CALENDAR`, `MACRO_OFFICIAL`, `FUNDAMENTALS`.

| Feature | Group | Definition (past-only) |
| --- | --- | --- |
| ret_1d, ma*_ratio, vol_20, rsi_14, macd* | CORE_PRICE | Price technicals |
| nasi_ret, sector_ret | CORE_PRICE | Market / sector context |
| xs_ret_1d, xs_ret_5d | RELATIVE | Stock minus NASI trailing returns |
| sector_xs, stock_vs_sector_1d | RELATIVE | Sector relative |
| rsi_vs_nasi, ma20_xs | RELATIVE | RSI / MA20 vs NASI |
| adv_20, rel_volume, amihud_20, jump_flag_1d, zero_vol_streak | LIQUIDITY | Volume / illiquidity |
| dow, month, results_season | CALENDAR | Calendar |
| cbk_*, usd_kes*, inflation_*, tbill_91d, lending_rate | MACRO_OFFICIAL | Only in **model** when `macro_source=official` |
| eps, eps_yoy, div_yield_proxy, book_value, book_to_price | FUNDAMENTALS | As-of join; model only if coverage ≥ 30% |

Ablation presets (valid only): `core_only`, `core_relative`, `full_gated` → `reports/feature_ablation.json`. Feature QA: `python scripts/run_feature_qa.py`.

### Error history (`error_history.parquet`)

Written by `scripts/train_evaluate.py` from the held-out test panel.

| Column | Description |
| --- | --- |
| trade_date | Forecast as-of date |
| stock_code | Ticker |
| prob_up | Model P(5d excess vs NASI > +1%) |
| pred_label | up / down (excess dead-zone labels) |
| y_dir_5d | Realised excess direction (NaN if \|excess\| ≤ 1%) |
| correct | Whether prediction matched label |
| horizon_days | Always 5 for v1 |

### Feature / label targets (research panel)

| Column | Description |
| --- | --- |
| y_ret_5d | Raw forward 5-day return |
| y_ret_5d_xs | Excess vs NASI 5-day return (label helper) |
| y_dir_5d | 1 if xs > +1%, 0 if xs < −1%, else excluded |
| y_vol_20d | Forward realized vol |

### Fundamentals (`fundamentals`)

Template: `python scripts/build_fundamentals_template.py` → `data/external/fundamentals_template.csv` (fill → `fundamentals.csv`; do not invent EPS).

| Column | Description |
| --- | --- |
| stock_code | Ticker |
| report_date | Period end |
| publish_date | When knowable (preferred as-of) |
| eps | Earnings per share |
| dividend | Dividend per share |
| book_value | Book value per share |
| period | e.g. FY2023, H1-2024 |

### Forecasts (`forecasts`)

| Column | Description |
| --- | --- |
| as_of_date | Date features were known |
| stock_code | Ticker |
| horizon_days | Prediction horizon |
| target_type | direction_5d_excess_nasi, return_range, volatility_20d |
| model_name | Baseline or model id |
| pred_label | up / down |
| prob_up | P(excess up beyond +1% dead zone) |
| pred_value / lower / upper | Point / interval predictions |

### Company logos (`data/external/logos/`)

| Asset | Description |
| --- | --- |
| `companies.csv` | `stock_code`, `company`, `domain` for favicon lookup |
| `{TICKER}.png` (etc.) | Optional local logo override |

### News (`data/external/news.csv`)

| Column | Description |
| --- | --- |
| stock_code | Ticker (e.g. SCOM) |
| date | Article / announcement date |
| headline | Short title |
| source | Publisher or “manual” |
| url | Optional link |
| sentiment | Optional: positive / neutral / negative |

Drop new files in `data/external/news/incoming/` and run `python scripts/collect_news.py`. Research context only — not a trading signal. Do not scrape NSE/news sites unless ToS allow it.

### Daily drop-folder collection

| Path | Role |
| --- | --- |
| `data/raw/nse/incoming/*.csv` | Drop new daily files here (licensed export / manual) |
| `data/raw/nse/incoming/archive/` | Processed drops moved here |
| `data/raw/nse/daily/NSE_data_all_stocks_YYYY.csv` | Append-only live history |
| `data/processed/collection_status.json` | Last run metadata |

Incoming CSVs must use Wanjawa-compatible headers (`Date`, `Code`, `Day Price`, `Volume`, …). Do **not** scrape NSE/MyStocks unless licensing/ToS allow it. `incoming/` and `archive/` are excluded from historical ingest globs so archives are not double-counted.

Live scoring (`scripts/score_latest.py`) reloads saved `.joblib` models and overwrites `latest_forecasts.parquet` without retraining.

## Feature leakage rule

Every feature must use **only information available before `as_of_date`**. No future closes, volumes, or macro revisions.
