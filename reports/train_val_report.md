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
- Direction threshold: **0.60**
- Valid utility (costed book): **1.8%**
- Calibration: **isotonic** (isotonic)
- Best params: `{'num_leaves': 31, 'min_child_samples': 40, 'learning_rate': 0.05, 'colsample_bytree': 0.7, 'n_estimators': 500}`
- Valid accuracy: **56.8%** (lift vs always-up +11.4%)
- Return residual quantiles (excess): [-0.052, 0.054] (target ~80% coverage)

## Test holdout (direction on excess labels)
- Always up: 51.0%
- Majority class: 51.0%
- MA 5/20: 47.3%
- LightGBM (selected): **55.1%** (precision_up 59.7%, recall_up 35.7%, n_scored 8709, abstain 0.09863382322500518)
- Abstain rate (flat): **9.9%**
- Rank IC (prob_up vs y_ret_5d_xs): **mean 0.166** (median 0.180, days 732) · 95% bootstrap CI [0.144, 0.185]
- At-chance names: NCBA, KQ, ABSA, COOP, KPLC, CTUM, KNRE, EQTY

## Calibration (test, non-flat)
- Brier **0.2506** · ECE **0.0630** · std(prob) **0.130** · unique probs **23**

## Confusion (test, non-flat only)
- TP 1582 · FP 1068 · TN 3214 · FN 2845

## Excess return interval (test)
- Coverage **72.8%**, mean width **12.0%**

## Stress Rank IC (OOS retrain)
- covid_2020: {'mean_ic': 0.08312467709230856, 'median_ic': 0.08241758241758242, 'n_days': 211, 'ic_std': 0.30421581461341424, 'abstain_rate': 0.29070181688635555}
- fx_2023: {'mean_ic': 0.12948263070154858, 'median_ic': 0.14521460932458408, 'n_days': 243, 'ic_std': 0.2761484167068252, 'abstain_rate': 0.230958988590811}

## Per-stock test accuracy (top / bottom)
| Stock | Accuracy | N | Flag |
| --- | --- | --- | --- |
| KCB | 60.3% | 496 |  |
| HFCK | 59.1% | 560 |  |
| CIC | 58.9% | 521 |  |
| SCAN | 58.0% | 574 |  |
| SCOM | 57.6% | 413 |  |
| … | | | |
| BRIT | 53.0% | 547 |  |
| KPLC | 52.9% | 548 | at_chance |
| CTUM | 51.6% | 527 | at_chance |
| KNRE | 50.2% | 538 | at_chance |
| EQTY | 49.0% | 486 | at_chance |

- Flagged names: NCBA, KQ, ABSA, COOP, KPLC, CTUM, KNRE, EQTY

## Walk-forward
- Mean Rank IC: **0.141** · IC>0 folds: **8/8** · beats always-up: **7/8**
- wf_2018: acc 54.4% · Rank IC 0.113 · abstain 26.7% · beats=True
- wf_2019: acc 56.1% · Rank IC 0.035 · abstain 22.7% · beats=True
- wf_2020: acc 63.9% · Rank IC 0.114 · abstain 50.2% · beats=True
- wf_2021: acc 57.3% · Rank IC 0.098 · abstain 48.4% · beats=True
- wf_2022: acc 55.3% · Rank IC 0.245 · abstain 5.2% · beats=True
- wf_2023: acc 45.9% · Rank IC 0.142 · abstain 8.7% · beats=False
- wf_2024: acc 53.6% · Rank IC 0.136 · abstain 21.5% · beats=True
- wf_2025: acc 60.4% · Rank IC 0.245 · abstain 25.6% · beats=True

## Research backtest (not for selection)
- Equal-weight long/flat · total return **94.1%** · buy-hold **170.7%** · max DD **-5.1%** · names 17
- Disclaimer: `research_only_not_for_selection`

