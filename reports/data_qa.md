# Data QA report

Prices: 303984 rows, 2007-01-02 → 2026-11-03, 107 tickers.

Investable focus (18): SCOM, EQTY, KCB, KENO, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, NCBA, HFCK, SCAN, NSE

## Duplicates
- duplicate_rows: **0** (ok=True)

## Calendar gaps
- gap events: **44**

## Zero volume / illiquidity
- flagged names: **2**
  - KENO: zero_vol=8%, max_dst=165
  - KQ: zero_vol=24%, max_dst=1117

## Large jumps
- |ret_1d| > 40%: **458** (showing 50)
- adj vs close disagreements on investable (>5%): **12295**

## Ticker hygiene
- rights issues: 10
- indices: ^FNK15, ^FNK25, ^N10I, ^N20I, ^N25I, ^NASI, ^NBDI, ^NBSI, ^ZKEQTK, ^ZKEQTU

## Known remaps (manual review)
- BBK → ABSA: Barclays Kenya rebranded to Absa (from=True, to=True)
- CFC → CFCI: Historical ticker variants — verify before merge (from=True, to=True)
- SAF → SCOM: Safaricom sometimes coded SAF in older lists (from=False, to=True)

## Macro
- source: **official** ok=True
- asof leak check: leaks=0 (checked=500)

## Sectors
- coverage of investable: **100%**
- orphans: none

## Fundamentals
- path: `C:\Users\Marllone.Mayieka\Desktop\nse predictor\data\external\fundamentals.csv`
- rows / tickers: **62** / **19**
- investable coverage (any fundamentals): **83%**
- bank metric coverage (ROE/ROA/NIM/PAT YoY): **100%**
- missing publish_date tickers: none
- banks missing latest ROE: none
- banks missing latest NPL: none
- |eps| > 100 outliers: **0**
