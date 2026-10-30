# NSE Stock Predictor

Decision-support forecasts for the Nairobi Securities Exchange (NSE). Combines price history, macro context, and fundamentals into **probabilities and ranges** — not a machine that claims to beat the market.

## Day-one expectation

Stock prices are noisy. Most short-term moves are close to random. NSE adds thin trading and index concentration (e.g. Safaricom). A good project:

1. Beats simple baselines on **unseen** time periods  
2. Reports how often and by how much it is wrong  
3. Shows the facts behind each forecast so a **human** decides  

**Out of scope:** automated trading, real money, investment advice to end users.

## Disclaimer

Forecasts are for **internal research and decision support only**. They are **not investment advice**. Past performance does not guarantee future results. Any real investment decision should involve a licensed adviser regulated by the Capital Markets Authority.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Data: prefer real Mendeley downloads
python scripts/download_wanjawa.py

# If download is blocked, use synthetic panel for pipeline testing
python scripts/generate_sample_panel.py

# Macro: starter CBK/FX/inflation template (replace with official CBK/KNBS CSVs)
python scripts/build_macro_template.py
# or drop your own file at data/external/macro_cbk.csv

python scripts/run_ingest.py
python scripts/train_evaluate.py
# Train → calibrate + tune threshold / return quantiles on valid → refit train+valid → test once
# Writes: models/artifacts/*.joblib, reports/evaluation_results.json,
# reports/train_val_report.md, reports/per_stock_direction.csv,
# data/processed/latest_forecasts.parquet + error_history.parquet
# (train once; daily refresh uses score_latest only)

# API
uvicorn api.main:app --reload --app-dir .
# accuracy: GET /accuracy/SCOM
# live score: POST /score/latest
# daily refresh: POST /admin/refresh

# Dashboard (watchlist + per-stock hit rate)
streamlit run dashboard/app.py
```

## Daily collector (drop-folder, no scrape)

Licensing-safe path: export a daily OHLCV CSV from your licensed feed (or manual download) and drop it here:

`data/raw/nse/incoming/`

Expected columns (Wanjawa-compatible): `Date`, `Code`, `Name`, `12m Low`, `12m High`, `Day Low`, `Day High`, `Day Price`, `Previous`, `Change`, `Change%`, `Volume`, `Adjusted Price`.

### Checklist: daily CSV drop → run_daily

1. Export a **full-board** session CSV (not a 3-row sample).
2. Drop it into `data/raw/nse/incoming/`.
3. Run `python scripts/run_daily.py` (= collect → ingest → **QA** → score_latest).
4. Confirm `data/processed/collection_status.json` shows `max_date`, `n_tickers`, and `coverage_pct` ≥ 80 vs investable.
5. Confirm `reports/data_qa.md` and that `latest_forecasts` as-of matches the collected date for most investable names.
6. Optional retrain after material history updates: `python scripts/train_evaluate.py`.

```bash
python scripts/run_daily.py
# = collect_daily → run_ingest → run_data_qa → score_latest → run_monitor
```

Or step-by-step:

```bash
python scripts/collect_daily.py   # appends to data/raw/nse/daily/, archives incoming files
python scripts/run_ingest.py      # also prints non-fatal data QA summary
python scripts/run_data_qa.py     # reports/data_qa.json + .md
python scripts/score_latest.py    # refreshes forecasts with saved .joblib models (no retrain)
python scripts/run_monitor.py     # pipeline / score hygiene / lagged outcomes / gate watch
```

**Windows Task Scheduler** (weekdays ~17:30 EAT):

- Program: `C:\Users\...\nse predictor\.venv\Scripts\python.exe`
- Arguments: `scripts\run_daily.py`
- Start in: `C:\Users\...\nse predictor`

Status files (also shown on the dashboard top bar / Model health):

| File | Role |
| --- | --- |
| `data/processed/collection_status.json` | Last collector run |
| `data/processed/daily_run.json` | Last daily pipeline summary |
| `data/processed/monitor_status.json` | Latest monitor check + alerts |
| `data/processed/monitor_history.parquet` | Append-only monitor history |
| `data/processed/forecast_log.parquet` | Scored forecast snapshots for lagged outcomes |

`run_monitor.py` always exits 0 (Task Scheduler friendly). Grep stderr for `MONITOR_STATUS=fail` / `warn`.

## Company logos

Dashboard cards and stock detail show logos when available:

1. **Local files (preferred):** `data/external/logos/{TICKER}.png` — seed with:

```bash
python scripts/fetch_logos.py
```

   Or drop your own licensed brand assets with those filenames.
2. **Domain map:** `data/external/logos/companies.csv` — used by the fetcher / remote fallback
3. **Fallback:** two-letter monogram if the image is missing or fails to load

Use only logos you have rights to display. Cached/local files are embedded as data URIs in the UI.

## Stock news (drop-folder)

Curated headlines for the selected ticker (no scraping):

1. Drop CSVs into `data/external/news/incoming/` with columns  
   `stock_code,date,headline,source,url,sentiment` (`sentiment` optional: positive/neutral/negative)
2. Run:

```bash
python scripts/collect_news.py
```

Merged file: `data/external/news.csv` (seed sample included). The dashboard **Related news** panel filters by the sidebar stock.

Postgres (optional, when Docker is available):

```bash
docker compose up -d
# set DATABASE_URL in .env to postgresql+psycopg2://nse:nse_dev@localhost:5432/nse_predictor
```

Without Docker the default `.env` uses **SQLite** at `data/processed/nse_predictor.db`.

## Data sources (prices 2007–2025)

Barack Wanjawa Mendeley Data (CC BY 4.0):

| Years | DOI |
| --- | --- |
| 2007–2012 | 10.17632/5hk4zw32f5.1 |
| 2013–2020 | 10.17632/73rb78pmzw.2 |
| 2021 | 10.17632/97hkwn5y3x |
| 2022 | 10.17632/jmcdmnyh2s.2 |
| 2023–2024 | 10.17632/ss5pfw8xnk.2 |
| 2025 | 10.17632/2b63rx67xt.2 |

See [docs/data_dictionary.md](docs/data_dictionary.md). After download, check `data/manifest.json` for the **2020 H2 gap** flag.

## Prediction targets

1. Direction over next 5 trading days (primary)  
2. Expected return range 5–20 days  
3. Volatility over next 20 days  
4. Next-day exact close — **not** the main target  

## Modelling ladder

**ML problem:** 5-day **excess return vs NASI** direction with a **±1% dead zone** (flat excess moves excluded from labels). Return intervals still target raw 5d return. Investable universe is liquidity-/price-screened (`reports/universe_investable.json`).

Baselines → ARIMA/GARCH → LightGBM. Keep complexity only if it beats MA / majority baselines on walk-forward and shows positive Rank IC. SHAP explains tree drivers on the dashboard.

## Repo layout

```
api/                 FastAPI app
dashboard/           Streamlit UI
data/raw/nse/        Wanjawa CSVs (gitignored)
data/processed/      Parquet + local DB
docs/                Data dictionary, NSE notes
notebooks/           EDA
reports/             Evaluation JSON + final report
scripts/             Download, ingest, train
src/nse_predictor/   Library code
```

## Cite

Wanjawa, B. — Nairobi Securities Exchange all-stocks daily prices, Mendeley Data (CC BY 4.0).
