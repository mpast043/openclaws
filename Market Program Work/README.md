# Spectral Dimension Market Test

Core pipeline: S&P 500 correlation network → ds(t) → validation (VIX, lead/lag) → portfolio eval → memo. Decoupled: robustness (signed/absolute, alpha, windows), event-timing diagnostics, granular crisis tables.

## Run (core pipeline)

```bash
cd spectral_dimension_market_test
pip install -r requirements.txt
python run_pipeline.py
```

**Benchmark market (optional):** Validation and portfolio eval use one benchmark index. Default is S&P 500 (`spx`). Run against another market with `--market`:

```bash
python run_pipeline.py --market qqq    # Nasdaq (QQQ)
python run_pipeline.py --market efa    # Developed ex-US (EFA)
python run_pipeline.py --market eem    # Emerging Markets (EEM)
python run_pipeline.py --market iwm    # Russell 2000 (IWM)
```

When `--market` is not `spx`, outputs are suffixed so runs for different markets do not overwrite each other.

**Decoupled (run separately):**

- **Robustness** (signed vs absolute, alpha tuning, multiple window sizes):  
  `python scripts/run_robustness_only.py`  
  Requires `data/sp500_prices.csv`. Writes `data/robustness_sensitivity.csv`.

- **Event-timing diagnostics** (drawdown onset, ds-drop, lead table; coarse crisis summary):  
  `python scripts/run_event_timing_only.py [--market spx|qqq|efa|eem|iwm] [--no-plot] [--full-table]`  
  Requires `data/ds_series.csv`. Writes `data/event_timing_crisis_summary[_suffix].csv` (coarse) and optionally updates the overlay plot; full granular table is in `data/event_timing_summary[_suffix].csv` when run.

- **Cross-asset structural validation** (foreign equity + commodity ETFs; fixed rules, no tuning):  
  `python scripts/run_cross_asset_validation.py`  
  Requires `data/ds_series.csv`. Tests EFA, EEM, EWJ, VGK, GLD, DBC, USO with same ds overlay (20th pct) and VIX overlay (80th pct). Writes `data/cross_asset_comparison.csv`, `cross_asset_overlay_vs_baseline_diffs.csv`, `cross_asset_ds_vs_vix_overlay.csv`, `cross_asset_regime_overlap.csv`.

**Outputs (core pipeline)** in `data/`: `sp500_prices.csv`, `ds_series.csv`, `ds_with_crisis_overlays[.png|_qqq.png]`, `cross_corr_ds_vix[.csv|_qqq.csv]`, `lead_lag_summary[.csv|_qqq.csv]`, `portfolio_eval_metrics[.csv|_qqq.csv]`, `portfolio_eval_differences[.csv|_qqq.csv]`, `portfolio_eval_equity_curves[.png|_qqq.png]`, `research_memo[.txt|_qqq.txt]`.

## Definitions

- Correlation shrinkage: C_shrunk = (1 − α)C + αI, α = 0.05.
- Adjacency: W_ij = |C_ij|, diagonal 0.
- Laplacian: L = D − W, D_ii = Σ_j W_ij.
- Heat trace: Z(σ) = Σ_i exp(−σ λ_i).
- Spectral dimension: ds(σ) = −2 d(log Z)/d(log σ); plateau = longest contiguous region with sliding IQR (window 5) < 0.05, mean ds over that region.
