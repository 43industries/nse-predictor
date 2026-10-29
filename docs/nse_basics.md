# NSE basics (Week 1 notes)

## Market structure

- **NSE** (Nairobi Securities Exchange) lists equities, bonds, and ETFs.
- Trading is continuous on business days; many counters are **illiquid** (zero volume for days).
- Indices: **NASI** (all share), **NSE 20** (blue chips). A few large names (notably Safaricom / SCOM) dominate market-cap weighted moves.

## Corporate actions

- Splits, bonus issues, rights issues, and dividends change price levels.
- Prefer **adjusted** prices when available; flag discontinuous jumps in cleaning.

## Dividends and book closure

- Dividend entitlement uses a **book closure** / record date.
- Calendar features can mark results season (typically early calendar year and mid-year for many issuers).

## Costs (for backtests)

- Brokerage, exchange fees, and **minimum lot** sizes matter more on NSE than on deep US markets.
- Thin names: slippage can dominate any signal.

## Liquidity screen for v1

Keep ~10–15 names with consistently high average daily volume; exclude tickers with frequent zero-volume days. Always include NASI / NSE20 for context features.
