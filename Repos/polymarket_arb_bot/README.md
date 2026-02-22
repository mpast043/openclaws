# Polymarket Sports Arb Bot V2

This is the hardened V2 scaffold for complementary-outcome arbitrage on Polymarket sports markets.

What is new in V2

1. Depth-aware sizing (VWAP and worst-price across book levels)
2. Lower-bound edge math (fees + slippage reserve + ops cost)
3. Larger-size-safe sizing rule: choose the largest `q` with positive lower-bound edge
4. Second-leg failure hedge fallback (best-effort FOK sell on leg A)
5. Bundle ledger (`bundles.csv`) for expected locked PnL tracking
6. Trade log reconciliation command (`python main.py reconcile`)
7. Optional websocket quote cache scaffold (fallback remains REST polling)

## Core math (lower-bound)

For a paired complementary trade of `q` shares:

- `C_gross(q)` = executable total buy cost across both books (depth-aware)
- `F(q)` = estimated fees (conservative, uses worst fill price)
- `S(q)` = slippage reserve = `q * 2 * buffer_per_leg`
- `O` = per-bundle ops reserve
- payout = `q`

Lower-bound locked edge:

`Edge_lb(q) = q - [C_gross(q) + F(q) + S(q) + O]`

V2 trades only if `Edge_lb(q) > 0` and selects the largest `q` that remains positive and within budget.

## Important caveat

Two-leg execution is still not truly atomic unless your infrastructure supports it.
V2 reduces risk by:
- FOK on each leg
- using depth-derived worst prices
- attempting an immediate hedge on leg A if leg B fails

But a true atomic multi-order route would still be better.

## Files

- `main.py` CLI (`scan`, `paper`, `live`, `loop`, `reconcile`)
- `polyarb_bot/scanner.py` depth-aware opportunity scoring
- `polyarb_bot/executor.py` paper/live execution + hedge fallback + logs
- `polyarb_bot/reconcile.py` trade log summary
- `polyarb_bot/ws_quotes.py` optional websocket quote cache scaffold

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python main.py scan --verbose
python main.py paper --verbose
python main.py reconcile
```

## Key env vars (V2)

- `LATENCY_SLIPPAGE_BUFFER_PER_LEG`
- `OPS_COST_PER_BUNDLE`
- `DEPTH_LEVELS_LIMIT`
- `MIN_EDGE_TOTAL_USDC`
- `HEDGE_ON_LEG_FAIL`
- `HEDGE_PRICE_BUFFER`
- `BUNDLE_LOG_PATH`
- `ENABLE_WEBSOCKET_QUOTES` (optional, off by default)
