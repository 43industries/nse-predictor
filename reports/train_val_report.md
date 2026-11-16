# Train / validation report

Decision-support research only - not investment advice.

## Problem definition
- **Direction:** P(5d excess vs NASI **> +1%**); label **down** if excess **< -1%**; rows with |excess| <= 1% are excluded (dead zone).
- **Return model:** 5d **excess** vs NASI + residual quantile band.
- **Universe:** investable ordinary shares (liquidity / price / history gates).

- Investable names: SCOM, EQTY, KCB, KENO, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, NCBA, HFCK, SCAN, NSE

> **validation_status=warn** — flagged_frac=100% ≥ 40%

## Protocol
- Embargo: **True** · horizon **5d** · train_end `2019-12-31` · valid_end `2022-12-31`
- Fit rows after embargo: train 34588 (raw 34620), valid 8297 (raw 8364)

## Time splits
- **train**: ['2008-01-04', '2019-12-24', 34588]
- **valid**: ['2020-01-06', '2022-12-23', 8297]
- **test**: ['2023-01-05', '2025-12-24', 9662]

## Selection (from validation)
- Direction head: **ranker**
- Decision mode: **threshold**
- Decision metric: **utility**
- Direction threshold: **0.59**
- Valid utility (costed book): **0.0%**
- Calibration: **isotonic** (isotonic)
- Best params: `{'num_leaves': 15, 'min_child_samples': 60, 'learning_rate': 0.05, 'n_estimators': 500}`
- Valid accuracy: **55.1%** (lift vs always-up +10.2%)
- Return residual quantiles (excess): [-0.058, 0.057] (target ~80% coverage)

## Test holdout (direction on excess labels)
- Always up: 51.0%
- Majority class: 51.0%
- MA 5/20: 47.3%
- LightGBM (selected): **50.0%** (precision_up 0.0%, recall_up 0.0%, n_scored 7073, abstain 0.26795694473193954)
- Abstain rate (flat): **26.8%**
- Rank IC (prob_up vs y_ret_5d_xs): **mean 0.114** (median 0.122, days 734) · 95% bootstrap CI [0.094, 0.135]
- At-chance names: KQ, SCOM, SCAN, EABL, KCB, CIC, HFCK, KNRE, BRIT, COOP, KEGN, EQTY, CTUM, NSE, KPLC, NCBA, ABSA

## Calibration (test, non-flat)
- Brier **0.2522** · ECE **0.0595** · std(prob) **0.077** · unique probs **13**

## Confusion (test, non-flat only)
- TP 0 · FP 0 · TN 3540 · FN 3533

## Excess return interval (test)
- Coverage **87.0%**, mean width **19.4%**

## Stress Rank IC (OOS retrain)
- covid_2020: {'mean_ic': 0.15546225248229056, 'median_ic': 0.1607142857142857, 'n_days': 211, 'ic_std': 0.28267520903299925, 'abstain_rate': 0.16494478090488066}
- fx_2023: {'mean_ic': 0.06347069748864607, 'median_ic': 0.0857142857142857, 'n_days': 243, 'ic_std': 0.29904379810302395, 'abstain_rate': 0.08356460067838421}

## Per-stock test accuracy (top / bottom)
| Stock | Accuracy | N | Flag |
| --- | --- | --- | --- |
| KQ | 56.5% | 492 | at_chance |
| SCOM | 55.4% | 379 | at_chance |
| SCAN | 53.6% | 435 | at_chance |
| EABL | 53.4% | 440 | at_chance |
| KCB | 53.4% | 369 | at_chance |
| … | | | |
| CTUM | 49.0% | 455 | at_chance |
| NSE | 47.0% | 411 | at_chance |
| KPLC | 44.5% | 452 | at_chance |
| NCBA | 44.2% | 448 | at_chance |
| ABSA | 43.7% | 359 | at_chance |

- Flagged names: KQ, SCOM, SCAN, EABL, KCB, CIC, HFCK, KNRE, BRIT, COOP, KEGN, EQTY, CTUM, NSE, KPLC, NCBA, ABSA

## Walk-forward
- Mean Rank IC: **0.123** · IC>0 folds: **8/8** · beats always-up: **5/8**
- wf_2018: acc 60.7% · Rank IC 0.141 · abstain 71.7% · beats=True
- wf_2019: acc 55.3% · Rank IC 0.043 · abstain 46.7% · beats=True
- wf_2020: acc 61.3% · Rank IC 0.126 · abstain 31.6% · beats=True
- wf_2021: acc 54.0% · Rank IC 0.169 · abstain 25.3% · beats=True
- wf_2022: acc 50.3% · Rank IC 0.195 · abstain 31.4% · beats=False
- wf_2023: acc 46.9% · Rank IC 0.042 · abstain 54.8% · beats=False
- wf_2024: acc 47.1% · Rank IC 0.085 · abstain 82.5% · beats=False
- wf_2025: acc 60.9% · Rank IC 0.184 · abstain 70.3% · beats=True

## Research backtest (not for selection)
- Equal-weight long/flat · total return **0.0%** · buy-hold **170.7%** · max DD **0.0%** · names 17
- Disclaimer: `research_only_not_for_selection`

