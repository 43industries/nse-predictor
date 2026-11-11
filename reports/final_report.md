# NSE Predictor — Final Evaluation Report

**Generated from** `reports/evaluation_results.json` after training on Wanjawa Mendeley daily prices (2007–2025), including curated bank FY2025 fundamentals.

## Purpose

Internal decision-support research for NSE equities. **Not investment advice.**

## Method

1. Daily prices from Wanjawa Mendeley (CC BY 4.0) + official-marked CBK/macro series when present  
2. Leakage-safe features: returns, MAs, RSI, MACD, rolling vol, liquidity flags, NASI context, macro, calendar, news  
3. Curated bank fundamentals (kenyanbanks.com FY table → `fundamentals.csv`, no scrape): ROE, ROA, NIM, CIR, NPL, PAT YoY, loans/deposits + peer medians  
4. Baselines: always-up, MA(5/20) cross, tomorrow=today (zero return)  
5. Models: LightGBM direction / 5d excess return intervals / 20d volatility; LSTM stretch; ARIMA & GARCH examples  
6. Time split train ≤2019 / valid 2020–2022 / test 2023+; yearly walk-forward; stress slices  

## Universe (liquid ordinary shares)

`pick_universe(n=18)`: SCOM, EQTY, KCB, KENO, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, NCBA, HFCK, SCAN, NSE (+ ^NASI / ^N20I for context).

Bank candidates with FY2025 metrics on file but not forced into the book unless liquidity admits: SCBK, DTK, IMH, SBIC.

## Bank fundamentals upgrade

- Seeded FY2025 bank rows for EQTY, KCB, COOP, ABSA, NCBA, IMH, SBIC, SCBK, DTK (`reports/fundamentals_qa.md`).  
- Feature ablation on **valid**: `core_relative` won (acc **57.7%**, Rank IC **0.114**).  
- `full_gated` (includes bank metrics + peers + macro/news/liquidity) had valid acc **56.0%** but higher Rank IC **0.146**.  
- Selected production head therefore uses **core_relative** (bank features available in the panel / dashboard, not in the selected LightGBM feature list).  
- Primary value of the upgrade: **dashboard bank as-of strip** + QA + richer CSV for future ablations / longer-horizon work.

## Results vs baselines (test, scored non-flat)

| Model | Metric | Value | Baseline | Beats? |
| --- | --- | ---: | ---: | --- |
| LightGBM direction | Accuracy | **55.1%** | Always-up 50.8% | **Yes** (+4.3 pp) |
| LightGBM 5d excess return | MAE | 0.0457 | Zero-excess 0.0455 | **No** |
| LightGBM 20d vol | MAE | 0.0109 | — | — |

### Walk-forward

- Mean Rank IC ≈ **0.141**; IC-positive folds **8/8**; beat always-up on **7/8** folds (`evaluation_results.json`).

### Where the model does **not** beat the baseline

- **5-day return point forecasts** do not beat naïve zero-excess on MAE. Prefer **direction probabilities** and **intervals**.  
- Costed long/flat book still trails buy-and-hold (research illustration only).



### Costed long/flat book (test window)

- Cost assumption: **50.0 bps** per switch; lot size note **100**
- Strategy total return: **94.1%**
- Buy-and-hold: **170.7%**
- Max drawdown: **-5.1%**
- Win rate (name avg): **48.6%**
- Beats buy-and-hold: **False**
- Research illustration only — not a trading mandate.

## Limitations

- Short-horizon equity direction remains noisy; many 5d moves are still misclassified  
- Bank FY metrics are sparse historically (mostly FY2025 seed) — limited train-period signal  
- Rights issues and illiquids excluded; results do not generalize to thin counters  
- No live licensed market-data scrape of kenyanbanks.com  

## Demo checklist

- [x] Dashboard shows forecast + P(up) + return range  
- [x] Dashboard bank fundamentals strip (as-of PAT YoY / ROE / NPL / EPS / DPS + peer ROE note)  
- [x] SHAP drivers CSV available  
- [x] Disclaimer on API and dashboard  
- [x] Fundamentals QA report  
- [x] Evaluation JSON reviewed  

## How to rerun

```bash
python scripts/download_wanjawa.py
python scripts/run_ingest.py
python scripts/_seed_bank_fundamentals.py   # optional: refresh curated FY seed
python scripts/build_feature_cache.py
python scripts/train_evaluate.py
uvicorn api.main:app --reload --app-dir .
streamlit run dashboard/app.py
```

## Disclaimer

Forecasts are for internal research and decision support only. They are not investment advice. Past performance does not guarantee future results. Any real investment decision should involve a licensed adviser regulated by the Capital Markets Authority.
