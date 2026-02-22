# SPEC_V2.md

## Polymarket Complementary Arb Bot V2 Specification

### Objective

Exploit complementary-outcome pricing inefficiencies in two-outcome sports markets while minimizing execution risk from:
- depth slippage
- fee misestimation
- one-leg exposure (legging risk)
- rule mismatch / invalid pairing

### Supported pair types

1. OUTCOME_PAIR
   - Single binary market with exactly two mutually exclusive outcomes
   - Buy both outcomes if combined executable cost < payout after fees

2. YES_PAIR
   - Two related YES/NO markets under the same event
   - Buy YES on both if they are complementary and executable cost supports locked edge

3. NO_PAIR
   - Two related YES/NO markets under the same event
   - Buy NO on both under same complementarity assumptions

### V2 Improvements

#### 1) Depth-aware sizing
V1 often sized off top-of-book only.
V2 computes executable cost using order book depth and VWAP/worst-price at a target size `q`.

For each leg:
- walk asks from best to worst
- compute:
  - total executable cost
  - VWAP
  - worst fill price
  - number of levels consumed

This makes the edge estimate robust to real liquidity.

#### 2) Lower-bound edge math
For candidate size `q`, define:

- `C_gross(q)` = sum of executable costs for both legs
- `F(q)` = sum of fee estimates (conservative, using worst fill price)
- `S(q)` = slippage reserve = `q * 2 * latency_buffer_per_leg`
- `O` = fixed ops reserve per bundle
- payout = `q`

Then:

`Edge_lb(q) = q - [C_gross(q) + F(q) + S(q) + O]`

A trade is valid only if:
- `Edge_lb(q) > 0`
- `Edge_lb(q)/q >= MIN_EDGE_USDC_PER_SHARE`
- `Edge_lb(q) >= MIN_EDGE_TOTAL_USDC`
- net cost <= `CAPITAL_PER_TRADE_USDC`

#### 3) Mathematically conservative sizing rule
V2 evaluates multiple size breakpoints (depth breakpoints + interior points) and selects:

- the largest `q` with positive lower-bound edge

This aligns with your requirement to size by mathematical confidence, not top-of-book optimism.

#### 4) Pair validation guardrails
V2 adds:
- negative-risk skipping (configurable)
- sports filtering
- red-flag rule text screening (draw/tie/regulation-only/void/postponed keywords)
- pair rule hashing for traceability (`rule_hash`)

#### 5) Execution hardening
Live mode uses:
- FOK per-leg limit buys
- price caps derived from worst observed executable depth + buffer

If leg A fills and leg B fails:
- V2 attempts immediate hedge on leg A (best-effort FOK sell using best bid minus hedge buffer)

This is not fully atomic, but it is a meaningful risk reduction.

#### 6) Fill and PnL tracking
V2 writes:
- `trades.csv`: execution-level logs, sizing math, costs, fees, expected edge
- `bundles.csv`: bundle-level locked-PnL ledger

Reconciliation command:
- `python main.py reconcile`
- summarizes counts by status and expected locked PnL from trade logs

#### 7) Optional websocket quote cache
V2 includes an optional websocket quote cache scaffold:
- `polyarb_bot/ws_quotes.py`

If websocket is disabled or schema changes:
- scanner falls back to REST polling

### Safety model and known limitations

1. Complementarity is assumed from market structure
   - Bot does not formally prove semantic complementarity
   - Human review still recommended for new leagues/market templates

2. Fees endpoint shape can vary
   - Client attempts multiple endpoint shapes and caches results
   - Missing fee response defaults to 0 (conservative config should account for this)

3. Execution is not atomic
   - True atomic bundled order placement would be superior
   - V2 uses FOK + hedge fallback as mitigation

4. Settlement PnL is logged as expected locked PnL at entry
   - Full post-settlement realization tracking can be added later by ingesting fills/settlements directly

### CLI

- `python main.py scan`
- `python main.py paper`
- `python main.py live`
- `python main.py loop`
- `python main.py reconcile`

### Environment variables to tune first

- `CAPITAL_PER_TRADE_USDC`
- `MIN_EDGE_USDC_PER_SHARE`
- `MIN_EDGE_TOTAL_USDC`
- `LATENCY_SLIPPAGE_BUFFER_PER_LEG`
- `OPS_COST_PER_BUNDLE`
- `DEPTH_LEVELS_LIMIT`
- `HEDGE_ON_LEG_FAIL`
- `HEDGE_PRICE_BUFFER`

### Recommended rollout path

1. `scan` mode for several hours
2. `paper` mode with logs
3. review `trades.csv` and `bundles.csv`
4. tighten thresholds and buffers
5. only then test `live` with very small size
