# Proposed liquid universe (v1 → v1.1)

Ranked by average daily volume after ingest of Wanjawa 2007–2025 panel.

Default screen size raised from **15 → 18** equities (`pick_universe`) so additional liquid banks can enter without forcing thin names. Gates unchanged: ≥1500 sessions, last trade ≥2023-01-01, recent median price ≥ 1.0, ADV above median preferred.

| Rank | Code | Avg volume | Zero-vol days | Span |
| --- | --- | ---: | ---: | --- |
| 1 | SCOM | 10,648,634 | 1 | 2008-01-07 → 2025-12-31 |
| 2 | EQTY | 1,934,144 | 0 | 2007-01-02 → 2025-12-31 |
| 3 | KCB | 1,674,154 | 0 | 2007-01-02 → 2025-12-31 |
| 4 | KENO | 1,012,412 | 239 | 2007-01-02 → 2019-10-11 |
| 5 | COOP | 858,862 | 1 | 2008-11-13 → 2025-12-31 |
| 6 | ABSA | 696,506 | 2 | 2013-01-02 → 2025-12-31 |
| 7 | MSC | 670,731 | 999 | 2007-01-02 → 2025-12-31 |
| 8 | KEGN | 661,698 | 1 | 2007-01-02 → 2025-12-31 |
| 9 | KNRE | 618,385 | 2 | 2007-01-10 → 2025-12-31 |
| 10 | KPLC | 599,410 | 0 | 2007-01-02 → 2025-12-31 |
| 11 | BRIT | 443,123 | 1 | 2011-01-11 → 2025-12-31 |
| 12 | CIC | 372,177 | 0 | 2012-01-02 → 2025-12-31 |
| 13 | BBK | 351,625 | 0 | 2007-01-02 → 2012-12-31 |
| 14 | ACCS | 318,693 | 1 | 2007-01-08 → 2012-12-31 |
| 15 | LAPR | 299,270 | 672 | 2023-03-22 → 2025-12-31 |

Indices retained for context features when present: NASI, NSE20.

## Bank candidates (kenyanbanks.com map)

Curated ticker map in `nse_predictor.ingest.banks.BANK_TICKERS`:

| Code | Name | FY2025 bank metrics seeded | Notes |
| --- | --- | --- | --- |
| EQTY | Equity Group | Yes | Core investable |
| KCB | KCB Group | Yes | Core investable |
| COOP | Co-operative Bank | Yes | Core investable |
| ABSA | Absa Bank Kenya | Yes | Core investable |
| NCBA | NCBA Group | Yes | Often in top-ADV set |
| HFCK | HF Group | Partial (legacy EPS/DPS only) | FY2025 pending on source dashboard |
| SCBK | Standard Chartered Kenya | Yes | Enters if liquidity screen admits |
| DTK | Diamond Trust Bank | Yes | Enters if liquidity screen admits |
| IMH | I&M Group | Yes | Enters if liquidity screen admits |
| SBIC | Stanbic Holdings | Yes | Enters if liquidity screen admits |

**Rule:** do not force Family Bank or other pending/thin names. Re-run `pick_universe` / `train_evaluate` / `build_feature_cache` to refresh `reports/universe_investable.json` after the n=18 change.

### After n=18 refresh (train_evaluate)

Investable equities admitted: SCOM, EQTY, KCB, KENO, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, **NCBA**, **HFCK**, SCAN, NSE.

Bank tickers with FY2025 metrics that did **not** clear the ADV screen this run: SCBK, DTK, IMH, SBIC (still on file for dashboard / future screens).
