# Feature QA report

- macro_source: **synthetic**
- fundamentals: skipped low ticker overlap=22.2% < 30%
- model features (28): gated full set

> Macro features **excluded from model** (synthetic): cbk_rate, cbk_chg, tbill_91d, usd_kes, usd_kes_chg, inflation_yoy, inflation_trend, lending_rate

## Relative coverage (investable labeled)

- `xs_ret_1d`: 100%
- `xs_ret_5d`: 100%
- `sector_xs`: 100%
- `stock_vs_sector_1d`: 100%
- `rsi_vs_nasi`: 98%
- `ma20_xs`: 99%

## Per-feature (top coverage issues)

- `eps` (FUNDAMENTALS): cov=0% in_model=False rank_ic=n/a
- `eps_yoy` (FUNDAMENTALS): cov=0% in_model=False rank_ic=n/a
- `div_yield_proxy` (FUNDAMENTALS): cov=0% in_model=False rank_ic=n/a
- `book_value` (FUNDAMENTALS): cov=0% in_model=False rank_ic=n/a
- `book_to_price` (FUNDAMENTALS): cov=0% in_model=False rank_ic=n/a
- `dist_52w_high` (CORE_PRICE): cov=27% in_model=True rank_ic=-0.0045
- `dist_52w_low` (CORE_PRICE): cov=27% in_model=True rank_ic=0.0020
- `adv_20` (LIQUIDITY): cov=27% in_model=True rank_ic=-0.0166
- `rel_volume` (LIQUIDITY): cov=27% in_model=True rank_ic=0.0295
- `volume_chg` (LIQUIDITY): cov=27% in_model=True rank_ic=0.0381
- `rsi_vs_nasi` (RELATIVE): cov=98% in_model=True rank_ic=-0.1515
- `rsi_14` (CORE_PRICE): cov=98% in_model=True rank_ic=-0.1534
- `amihud_20` (LIQUIDITY): cov=99% in_model=True rank_ic=-0.0332
- `ma50_ratio` (CORE_PRICE): cov=99% in_model=True rank_ic=-0.1693
- `ma20_xs` (RELATIVE): cov=99% in_model=True rank_ic=-0.1963
