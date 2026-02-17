#!/usr/bin/env python3
"""
Spectral dimension market test — single entry script.
Core pipeline: data -> returns -> ds(t) -> validation (VIX, lead/lag) -> portfolio eval -> memo.
Decoupled (run separately): robustness, event-timing diagnostics. See scripts/.
Use --market to run against a specific benchmark: spx (default), qqq, efa, eem, iwm.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Benchmark ticker by market name
MARKET_TICKER = {
    "spx": "^GSPC",   # S&P 500
    "qqq": "QQQ",     # Nasdaq
    "efa": "EFA",     # Developed ex-US
    "eem": "EEM",     # Emerging Markets
    "iwm": "IWM",     # Russell 2000
}

from src.data_loader import load_prices
from src.correlations import log_returns, rolling_correlation_matrices, correlation_to_adjacency
from src.spectral import compute_ds_series
from src.validation import run_validation, plot_ds_with_overlays
from src.portfolio_eval import run_portfolio_eval


def main():
    parser = argparse.ArgumentParser(description="Spectral dimension market test. Optionally select benchmark market.")
    parser.add_argument(
        "--market",
        choices=list(MARKET_TICKER.keys()),
        default="spx",
        help="Benchmark market for validation, event-timing, and portfolio eval: spx (default), qqq, efa, eem, iwm",
    )
    args = parser.parse_args()
    benchmark_ticker = MARKET_TICKER[args.market]

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_" + benchmark_ticker.replace("^", "").lower() if benchmark_ticker != "^GSPC" else ""
    print(f"Benchmark market: {args.market} ({benchmark_ticker})")

    # Phase 1 — Data
    print("Phase 1: Loading S&P 500 prices...")
    prices, n_tickers, (first_date, last_date) = load_prices(
        start_date="2000-01-01",
        end_date="2024-12-31",
        max_missing_frac=0.10,
        data_dir=data_dir,
    )
    print(f"  Tickers: {n_tickers}, Range: {first_date} to {last_date}")

    # Phase 2 & 3 — Returns and spectral dimension time series (slow: ~5k+ windows)
    n_windows = len(prices) - 60 + 1
    print(f"Phase 2–3: Returns and ds(t) series (window=60, α=0.05, ~{n_windows} windows)...")
    returns = log_returns(prices)
    # Edge density (one representative window)
    _, C0 = next(rolling_correlation_matrices(returns, window=60, alpha=0.05, use_abs=True))
    W0 = correlation_to_adjacency(C0)
    edge_density = np.count_nonzero(W0) / (W0.shape[0] ** 2 - W0.shape[0])
    print(f"  Edge density: {edge_density:.4f}")
    ds_list = [
        (d, v)
        for d, v in compute_ds_series(
            returns, window=60, alpha=0.05, use_abs=True, progress_interval=500
        )
        if v is not None
    ]
    ds_series = pd.Series({d: v for d, v in ds_list}).sort_index()
    ds_series.to_csv(data_dir / "ds_series.csv", header=True)
    print(f"  ds(t) length: {len(ds_series)}")

    # Phase 4 — Validation
    print("Phase 4: Validation (VIX, overlays, lead/lag)...")
    val = run_validation(ds_series, data_dir=data_dir, benchmark_ticker=benchmark_ticker)
    lead_lag = val["lead_lag_summary"]
    print("  Lead/lag summary:")
    print(lead_lag.to_string(index=False))
    if not lead_lag.empty:
        lead_lag.to_csv(data_dir / f"lead_lag_summary{suffix}.csv", index=False)
    # Plot: ds(t) + VIX + crisis shading (no event-timing overlays; run scripts/run_event_timing_only.py for that)
    plot_ds_with_overlays(
        ds_series,
        data_dir / f"ds_with_crisis_overlays{suffix}.png",
        vix=val["vix"],
    )

    # Portfolio evaluation (fixed rules: 20th pct compression, 70/30 overlay; no curve-fit)
    print("Phase 4c: Portfolio evaluation (baseline vs overlay)...")
    port = run_portfolio_eval(ds_series, data_dir=data_dir, benchmark_ticker=benchmark_ticker)
    print("  Comparison (baseline | ds overlay 20% | VIX overlay 80%):")
    print(port["comparison_df"].to_string(index=False))
    print("  Regime overlap (% days ds active, % VIX_high active, % both):")
    print(port["overlap_df"].to_string(index=False))
    print(f"  Saved: {port['comparison_path']}, {port['overlap_path']}, {port['metrics_path']}, {port['diff_path']}, {port['plot_path']}")

    # Research memo (no robustness / event-timing; run scripts for those)
    memo_path = data_dir / f"research_memo{suffix}.txt"
    write_memo(ds_series=ds_series, lead_lag_df=lead_lag, val=val, output_path=memo_path)
    print(f"\nMemo written: {memo_path}")
    print("Done.")


def write_memo(
    ds_series,
    lead_lag_df,
    val,
    output_path: Path,
):
    """Short research memo: ds in crises, lead/lag vs VIX, redundancy, conclusion."""

    lines = [
        "SPECTRAL DIMENSION MARKET TEST — RESEARCH MEMO",
        "=" * 50,
        "",
        "1. Does ds meaningfully change during crises?",
        "",
    ]
    # Simple crisis means vs non-crisis
    crisis_dates = [
        ("2008", "2008-01-01", "2009-06-30"),
        ("Mar 2020", "2020-02-01", "2020-05-31"),
        ("2022", "2022-01-01", "2022-12-31"),
    ]
    for name, start, end in crisis_dates:
        mask = (ds_series.index >= start) & (ds_series.index <= end)
        if mask.any():
            c_mean = float(ds_series[mask].mean())
            lines.append(f"   {name}: mean ds = {c_mean:.4f}")
        else:
            lines.append(f"   {name}: no data in range")
    non_crisis = ds_series.copy()
    for _, start, end in crisis_dates:
        non_crisis = non_crisis[(non_crisis.index < start) | (non_crisis.index > end)]
    if len(non_crisis) > 0:
        lines.append(f"   Non-crisis mean ds = {float(non_crisis.mean()):.4f}")
    lines.extend(["", "2. Is ds leading, coincident, or lagging (vs VIX)?", ""])
    if not lead_lag_df.empty:
        lag = int(lead_lag_df["best_lag_days"].iloc[0])
        corr = float(lead_lag_df["corr_at_best"].iloc[0])
        interp = lead_lag_df["interpretation"].iloc[0]
        lines.append(f"   Best lag: {lag} days, corr = {corr:.4f} — {interp}.")
    else:
        lines.append("   No lead/lag computed (insufficient overlap).")
    lines.extend(["", "3. Is ds redundant with volatility?", ""])
    if "vix" in val and val["vix"] is not None:
        common = ds_series.align(val["vix"], join="inner")
        valid = common[0].notna() & common[1].notna()
        if valid.sum() > 20:
            r = np.corrcoef(common[0][valid], common[1][valid])[0, 1]
            red = "High |r| suggests redundancy." if abs(r) > 0.3 else "Low |r|: not strongly redundant with VIX."
            lines.append(f"   Correlation ds–VIX (lag 0): {r:.4f}. {red}")
        else:
            lines.append("   Insufficient overlap for ds–VIX correlation.")
    lines.extend(["", "4. Evidence of structural regime detection?", ""])
    crisis_means = [float(ds_series[(ds_series.index >= s) & (ds_series.index <= e)].mean())
                   for (_, s, e) in crisis_dates
                   if ((ds_series.index >= s) & (ds_series.index <= e)).any()]
    non_crisis_mean = float(non_crisis.mean()) if len(non_crisis) > 0 else float("nan")
    if crisis_means and not np.isnan(non_crisis_mean):
        all_means = crisis_means + [non_crisis_mean]
        spread = max(all_means) - min(all_means) if all_means else 0
        lines.append(
            f"   Crisis vs non-crisis mean ds spread: {spread:.4f}. "
            + ("No meaningful difference." if spread < 0.01 else "Some difference.")
        )
    lines.extend(["", "5. Brutally honest summary", ""])
    # Verdict from data (no contradiction)
    crisis_spread = max(crisis_means) - min(crisis_means) if crisis_means else 0
    ds_effectively_constant = crisis_spread < 0.01 and (len(non_crisis) == 0 or abs(float(non_crisis.mean()) - (crisis_means[0] if crisis_means else 0)) < 0.01)
    if ds_effectively_constant:
        lines.append("   ds does NOT meaningfully change during crises: crisis and non-crisis mean ds are")
        lines.append("   effectively the same. No detectable regime shift in this metric.")
    else:
        lines.append("   ds shows some variation between crisis and non-crisis periods; inspect the plot.")
    r_vix = None
    if "vix" in val and val.get("vix") is not None:
        c0, c1 = ds_series.align(val["vix"], join="inner")
        v = c0.notna() & c1.notna()
        if v.sum() > 20:
            r_vix = np.corrcoef(c0[v], c1[v])[0, 1]
    if r_vix is not None:
        lines.append(f"   ds–VIX correlation (lag 0) is {r_vix:.3f}: " + ("moderate redundancy with volatility." if abs(r_vix) > 0.3 else "low redundancy."))
    lines.append("   Conclusion: This pipeline does not provide evidence that spectral dimension of")
    lines.append("   rolling correlation networks detects structural market regime shifts.")
    lines.append("")

    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
