# Train / validation report

Decision-support research only - not investment advice.

## Problem definition
- **Direction:** P(5d excess vs NASI **> +1%**); label **down** if excess **< -1%**; rows with |excess| <= 1% are excluded (dead zone).
- **Return model:** 5d **excess** vs NASI + residual quantile band.
- **Universe:** investable ordinary shares (liquidity / price / history gates).

- Investable names: SCOM, EQTY, KCB, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, NCBA, HFCK

> **validation_status=warn** — flagged_frac=53% ≥ 40%; macro_source=synthetic

## Protocol
- Embargo: **True** · horizon **5d** · train_end `2019-12-31` · valid_end `2022-12-31`
- Fit rows after embargo: train 29032 (raw 29058), valid 7241 (raw 7301)

## Time splits
- **train**: ['2008-01-04', '2019-12-24', 29032]
- **valid**: ['2020-01-06', '2022-12-23', 7241]
- **test**: ['2023-01-05', '2025-12-22', 8437]

## Selection (from validation)
- Direction head: **binary**
- Decision mode: **threshold**
- Direction threshold: **0.38**
- Calibration: **isotonic** (isotonic)
- Best params: `{'num_leaves': 15, 'min_child_samples': 60, 'learning_rate': 0.05, 'n_estimators': 500}`
- Valid accuracy: **57.3%** (lift vs always-up +10.6%)
- Return residual quantiles (excess): [-0.069, 0.048] (target ~80% coverage)

## Test holdout (direction on excess labels)
- Always up: 51.3%
- Majority class: 51.3%
- MA 5/20: 47.3%
- LightGBM (selected): **54.8%** (precision_up 55.1%, recall_up 60.1%, n_scored 6343, abstain 0.24819248548062106)
- Abstain rate (flat): **24.8%**
- Rank IC (prob_up vs y_ret_5d_xs): **mean 0.133** (median 0.143, days 730)

## Calibration (test, non-flat)
- Brier **0.2528** · ECE **0.0754** · std(prob) **0.138** · unique probs **39**

## Confusion (test, non-flat only)
- TP 1937 · FP 1578 · TN 1540 · FN 1288

## Excess return interval (test)
- Coverage **74.2%**, mean width **11.7%**

## Stress Rank IC (OOS retrain)
- covid_2020: {'mean_ic': 0.12870598528836788, 'median_ic': 0.12967032967032965, 'n_days': 211, 'ic_std': 0.320988374766932, 'abstain_rate': 0.29200652528548127}
- fx_2023: {'mean_ic': 0.023004181535341235, 'median_ic': 0.03922851696456023, 'n_days': 242, 'ic_std': 0.3002055481252739, 'abstain_rate': 0.24737945492662475}

## Per-stock test accuracy (top / bottom)
| Stock | Accuracy | N | Flag |
| --- | --- | --- | --- |
| NCBA | 59.1% | 450 |  |
| KPLC | 58.3% | 482 |  |
| KCB | 58.3% | 405 |  |
| CIC | 58.0% | 417 |  |
| ABSA | 57.1% | 385 | at_chance |
| … | | | |
| KEGN | 52.3% | 390 | at_chance |
| CTUM | 51.3% | 429 | at_chance |
| KNRE | 51.2% | 457 | at_chance |
| HFCK | 51.2% | 506 | at_chance |
| SCOM | 50.7% | 343 | at_chance |

- Flagged names: ABSA, EABL, KQ, KEGN, CTUM, KNRE, HFCK, SCOM

## Walk-forward
- Mean Rank IC: **0.120** · IC>0 folds: **8/8** · beats always-up: **7/8**
- wf_2018: acc 56.1% · Rank IC 0.133 · abstain 20.8% · beats=True
- wf_2019: acc 54.1% · Rank IC 0.009 · abstain 18.1% · beats=True
- wf_2020: acc 59.5% · Rank IC 0.139 · abstain 37.1% · beats=True
- wf_2021: acc 56.2% · Rank IC 0.130 · abstain 42.9% · beats=True
- wf_2022: acc 55.8% · Rank IC 0.182 · abstain 9.5% · beats=True
- wf_2023: acc 51.3% · Rank IC 0.039 · abstain 14.3% · beats=False
- wf_2024: acc 57.1% · Rank IC 0.155 · abstain 38.2% · beats=True
- wf_2025: acc 56.8% · Rank IC 0.177 · abstain 15.2% · beats=True

## Research backtest (not for selection)
- Equal-weight long/flat · total return **15.5%** · buy-hold **136.8%** · max DD **-22.0%** · names 15
- Disclaimer: `research_only_not_for_selection`

