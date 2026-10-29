# NSE Predictor — Final Evaluation Report

**Generated from** `reports/evaluation_results.json` after training on Wanjawa Mendeley daily prices (2007–2025).

## Purpose

Internal decision-support research for NSE equities. **Not investment advice.**

## Method

1. Daily prices from Wanjawa Mendeley (CC BY 4.0) + synthetic CBK macro placeholder pending official series  
2. Leakage-safe features: returns, MAs, RSI, MACD, rolling vol, liquidity flags, NASI context, macro, calendar  
3. Baselines: always-up, MA(5/20) cross, tomorrow=today (zero return)  
4. Models: LightGBM direction / 5d return intervals / 20d volatility; ARIMA & GARCH examples  
5. Time split train ≤2019 / valid 2020–2022 / test 2023–2025; yearly walk-forward; stress slices 2020 & 2023  

## Universe (liquid ordinary shares)

SCOM, EQTY, KCB, COOP, ABSA, MSC, KEGN, KNRE, KPLC, BRIT, CIC, EABL, HAFR, CTUM, KQ (+ ^NASI / ^N20I for context).

## Results vs baselines (test 2023–2025, n=11,760)

| Model | Metric | Value | Baseline | Beats? |
| --- | --- | ---: | ---: | --- |
| LightGBM direction | Accuracy | **58.7%** | Always-up 44.1% | **Yes** (+14.6 pp) |
| MA(5/20) direction | Accuracy | 55.2% | Always-up 44.1% | Yes |
| LightGBM 5d return | MAE | 0.0353 | Tomorrow=today 0.0344 | **No** |
| Return ±1σ interval | Coverage | 96.1% | — | Wide intervals |
| LightGBM 20d vol | MAE | 0.0082 | — | — |

### Walk-forward

Direction model beat always-up on multiple yearly folds (see `evaluation_results.json`). Folds and stress-period metrics are recorded there for audit.

### Where the model does **not** beat the baseline

- **5-day return point forecasts** do not beat the naïve “expected return = 0” baseline on MAE/RMSE. Prefer **direction probabilities** and **intervals**, not a single return number.  
- Interval coverage is high partly because bands are wide — useful for honesty, not for tight trading signals.

### SHAP drivers (global, test sample)

1. days_since_trade  
2. ma20_ratio  
3. vol_20  
4. nasi_ret  
5. macd_signal  

## Limitations

- Short-horizon equity direction remains noisy; ~41% of 5d moves are still misclassified on test  
- Macro series currently synthetic until official CBK CSVs are loaded  
- Rights issues and illiquids excluded; results do not generalize to thin counters  
- No live licensed market-data feed yet  
- Simulated costs not a license to trade  

## Demo checklist

- [x] Dashboard shows forecast + P(up) + return range  
- [x] SHAP drivers CSV available  
- [x] Disclaimer on API and dashboard  
- [x] Evaluation JSON reviewed  

## How to rerun

```bash
python scripts/download_wanjawa.py
python scripts/run_ingest.py
python scripts/train_evaluate.py
uvicorn api.main:app --reload --app-dir .
streamlit run dashboard/app.py
```

## Disclaimer

Forecasts are for internal research and decision support only. They are not investment advice. Past performance does not guarantee future results. Any real investment decision should involve a licensed adviser regulated by the Capital Markets Authority.
