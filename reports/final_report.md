# NSE Predictor — Final Evaluation Report

**Generated from** `reports/evaluation_results.json` after gated retrain (Rank IC + ship gates) and XGBoost direction ablation. Wanjawa Mendeley prices (2007–2025) + curated fundamentals.

## Purpose

Internal decision-support research for NSE equities. **Not investment advice.**

## Method

1. Daily prices from Wanjawa Mendeley (CC BY 4.0) + official-marked CBK/macro series  
2. Leakage-safe features: returns, MAs, RSI, MACD, vol, liquidity, NASI/NSE20 relatives, calendar, news, fundamentals  
3. Curated fundamentals CSV (banks + ordinary investable names) — **no scrape**  
4. Baselines: always-up, MA(5/20), zero-excess  
5. Models: LightGBM binary / LightGBM ranker / **XGBoost binary**; return intervals; 20d vol; LSTM stretch  
6. Feature presets selected on **valid Rank IC** subject to ship gates (non-degenerate up/down labels + beats always-up)  
7. Direction head selected the same way; degenerate ranker heads are rejected even if Rank IC is higher  
8. Time split train ≤2019 / valid 2020–2022 / test 2023+; walk-forward; stress slices  

## Universe

Investable ordinary shares (~18) + ^NASI / ^N20I context. See `reports/universe_investable.json`.

## Feature + head selection (this run)

- Selection metric: **valid Rank IC** with gates `non_degenerate_labels`, `beats_always_up`  
- Feature preset: **`full_gated`** (59 cols; gated winners included this preset)  
- Ranker: **rejected** (`degenerate_labels up=0`)  
- LightGBM binary: passes gates (valid Rank IC ≈ 0.149)  
- **XGBoost binary: selected** (valid Rank IC ≈ 0.151, utility > 0)  
- Production head: **`xgb_binary`** (`models/artifacts/direction.joblib`)

## Results vs baselines (test holdout, LightGBM binary artifact from full `train_evaluate`)

Full pipeline evaluation below reflects the gated **LightGBM binary** train_evaluate pass before the post-hoc XGB head swap. XGB won the subsequent gated ablation on the same features; latest scores use XGB.

| Model | Metric | Value | Baseline | Beats? |
| --- | --- | ---: | ---: | --- |
| LightGBM direction (gated binary) | Accuracy | **52.8%** | Always-up 51.0% | **Yes** (+1.8 pp) |
| Recall (up) | | 28.9% | — | non-zero (usable desk) |
| Rank IC (test) | | **0.118** | — | positive |
| Walk-forward mean Rank IC | | **0.131** | — | IC>0 on 7/8; beats always-up 7/8 |
| 5d excess return MAE | | ~0.045 | Zero-excess ~0.045 | **No** |

Confusion (test non-flat): TP 1309 · FP 1003 · TN 3412 · FN 3216 — **both up and down calls present**.

### Where the model does **not** beat the baseline

- Point 5d excess return MAE does not beat naïve zero-excess  
- Costed long/flat research book still trails buy-and-hold  
- ~47% of names flagged at-chance on the LightGBM eval (`validation_status=warn`)  
- Edge remains modest; desk should keep filters / caution banner  

### Costed long/flat book (test window)

- Cost assumption: **50.0 bps** per switch; lot size note **100**  
- Strategy total return: **41.5%**  
- Buy-and-hold: **170.7%**  
- Max drawdown: **-6.8%**  
- Beats buy-and-hold: **False**  
- Research illustration only — not a trading mandate  

## Blueprint gap closeout

| Gap | Status |
| --- | --- |
| Usable direction head (desk setups) | Restored via gates; ranker rejected; XGB/binary emit ups |
| Postgres primary store | `docker compose` + `scripts/sync_processed_to_db.py`; SQLite fallback when Docker absent |
| XGBoost in horse race | Ablated; **selected** on valid Rank IC under gates |
| News depth | Drop-folder (`data/external/news/incoming` + `data/raw/news/incoming`); no scrape |
| Fundamentals depth | Extended investable CSV + `reports/fundamentals_qa.md` |
| Live vendor scrape | Intentionally out of scope (licensing) |

## Limitations

- Short-horizon equity direction remains noisy  
- Fundamentals/news history still sparse in early years  
- Docker not required; without it the DB is SQLite  
- XGB production head was selected after the full eval JSON was written — re-run `python scripts/train_evaluate.py` for a single end-to-end artifact if needed  

## How to rerun

```bash
docker compose up -d   # optional Postgres
# ensure .env DATABASE_URL matches compose when using Postgres
python scripts/run_ingest.py
python scripts/sync_processed_to_db.py
python scripts/collect_news.py
python scripts/build_feature_cache.py
python scripts/train_evaluate.py
python scripts/score_latest.py
uvicorn api.main:app --reload --app-dir .
cd web && npm run dev
```

## Disclaimer

Forecasts are for internal research and decision support only. They are not investment advice. Past performance does not guarantee future results. Any real investment decision should involve a licensed adviser regulated by the Capital Markets Authority.
