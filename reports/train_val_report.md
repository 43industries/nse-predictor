# Train / validation report

Decision-support research only - not investment advice.

## Problem definition
- **Direction:** P(5d excess vs NASI **> +1%**); label **down** if excess **< -1%**; rows with |excess| <= 1% are excluded (dead zone).
- **Return model:** 5d **excess** vs NASI + residual quantile band.
- **Universe:** investable ordinary shares (liquidity / price / history gates).

- Investable names: SCOM, EQTY, KCB, KENO, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, NCBA, HFCK, SCAN, NSE

> **validation_status=warn** — flagged_frac=47% ≥ 40%

## Protocol
- Embargo: **True** · horizon **5d** · train_end `2019-12-31` · valid_end `2022-12-31`
- Fit rows after embargo: train 34588 (raw 34620), valid 8297 (raw 8364)

## Time splits
- **train**: ['2008-01-04', '2019-12-24', 34588]
- **valid**: ['2020-01-06', '2022-12-23', 8297]
- **test**: ['2023-01-05', '2025-12-24', 9662]

## Selection (from validation)
- Direction head: **binary**
- Decision mode: **threshold**
- Decision metric: **utility**
- Direction threshold: **0.44**
- Valid utility (costed book): **13.0%**
- Calibration: **isotonic** (isotonic)
- Best params: `{'num_leaves': 15, 'min_child_samples': 60, 'learning_rate': 0.05, 'n_estimators': 500}`
- Valid accuracy: **60.9%** (lift vs always-up +16.1%)
- Return residual quantiles (excess): [-0.058, 0.057] (target ~80% coverage)

## Test holdout (direction on excess labels)
- Always up: 51.0%
- Majority class: 51.0%
- MA 5/20: 47.3%
- LightGBM (selected): **52.8%** (precision_up 56.6%, recall_up 28.9%, n_scored 8940, abstain 0.07472572966259573)
- Abstain rate (flat): **7.5%**
- Rank IC (prob_up vs y_ret_5d_xs): **mean 0.118** (median 0.126, days 727) · 95% bootstrap CI [0.094, 0.139]
- At-chance names: SCOM, COOP, KNRE, NCBA, ABSA, EQTY, KEGN, KPLC

## Calibration (test, non-flat)
- Brier **0.2809** · ECE **0.1545** · std(prob) **0.212** · unique probs **138**

## Confusion (test, non-flat only)
- TP 1309 · FP 1003 · TN 3412 · FN 3216

## Excess return interval (test)
- Coverage **87.1%**, mean width **19.4%**

## Stress Rank IC (OOS retrain)
- covid_2020: {'mean_ic': 0.15546225248229056, 'median_ic': 0.1607142857142857, 'n_days': 211, 'ic_std': 0.28267520903299925, 'abstain_rate': 0.16494478090488066}
- fx_2023: {'mean_ic': 0.05910717404342555, 'median_ic': 0.07352941176470588, 'n_days': 243, 'ic_std': 0.2998321630062424, 'abstain_rate': 0.08356460067838421}

## Per-stock test accuracy (top / bottom)
| Stock | Accuracy | N | Flag |
| --- | --- | --- | --- |
| SCAN | 59.8% | 579 |  |
| EABL | 56.2% | 539 |  |
| KQ | 56.1% | 531 |  |
| HFCK | 55.5% | 569 |  |
| CTUM | 54.7% | 534 |  |
| … | | | |
| NCBA | 50.7% | 537 | at_chance |
| ABSA | 50.1% | 471 | at_chance |
| EQTY | 50.0% | 490 | at_chance |
| KEGN | 47.4% | 492 | at_chance |
| KPLC | 46.9% | 588 | at_chance |

- Flagged names: SCOM, COOP, KNRE, NCBA, ABSA, EQTY, KEGN, KPLC

## Walk-forward
- Mean Rank IC: **0.131** · IC>0 folds: **7/8** · beats always-up: **7/8**
- wf_2018: acc 55.1% · Rank IC 0.127 · abstain 9.7% · beats=True
- wf_2019: acc 51.1% · Rank IC -0.010 · abstain 19.4% · beats=True
- wf_2020: acc 59.3% · Rank IC 0.140 · abstain 17.3% · beats=True
- wf_2021: acc 53.0% · Rank IC 0.162 · abstain 7.6% · beats=True
- wf_2022: acc 57.2% · Rank IC 0.213 · abstain 13.1% · beats=True
- wf_2023: acc 48.3% · Rank IC 0.080 · abstain 5.5% · beats=False
- wf_2024: acc 56.8% · Rank IC 0.153 · abstain 25.8% · beats=True
- wf_2025: acc 58.9% · Rank IC 0.182 · abstain 20.6% · beats=True

## Research backtest (not for selection)
- Equal-weight long/flat · total return **41.5%** · buy-hold **170.7%** · max DD **-6.8%** · names 17
- Disclaimer: `research_only_not_for_selection`

