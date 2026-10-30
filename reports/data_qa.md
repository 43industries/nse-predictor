# Data QA report

> **SYNTHETIC MACRO — not official CBK/KNBS**

Prices: 303770 rows, 2007-01-02 → 2026-01-01, 107 tickers.

Investable focus (15): SCOM, EQTY, KCB, COOP, ABSA, KEGN, KNRE, KPLC, BRIT, CIC, EABL, CTUM, KQ, NCBA, HFCK

## Duplicates
- duplicate_rows: **0** (ok=True)

## Calendar gaps
- gap events: **25**

## Zero volume / illiquidity
- flagged names: **1**
  - KQ: zero_vol=24%, max_dst=1117

## Large jumps
- |ret_1d| > 40%: **388** (showing 50)
- adj vs close disagreements on investable (>5%): **9945**

## Ticker hygiene
- rights issues: 10
- indices: ^FNK15, ^FNK25, ^N10I, ^N20I, ^N25I, ^NASI, ^NBDI, ^NBSI, ^ZKEQTK, ^ZKEQTU

## Known remaps (manual review)
- BBK → ABSA: Barclays Kenya rebranded to Absa (from=True, to=True)
- CFC → CFCI: Historical ticker variants — verify before merge (from=True, to=True)
- SAF → SCOM: Safaricom sometimes coded SAF in older lists (from=False, to=True)

## Macro
- source: **synthetic** ok=True
- asof leak check: leaks=0 (checked=500)

## Sectors
- coverage of investable: **100%**
- orphans: none
