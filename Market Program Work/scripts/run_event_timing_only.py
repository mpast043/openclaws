#!/usr/bin/env python3
"""
Standalone event-timing diagnostics: drawdown onset, ds-drop, lead table.
Reads data/ds_series.csv. Writes a coarse crisis summary (one row per major episode at -20% only)
plus optional full lead table. Use --market for benchmark (spx, qqq, efa, eem, iwm).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MARKET_TICKER = {"spx": "^GSPC", "qqq": "QQQ", "efa": "EFA", "eem": "EEM", "iwm": "IWM"}

import pandas as pd

from src.event_timing import run_event_timing
from src.validation import plot_ds_with_overlays, load_vix_and_benchmark


def main():
    parser = argparse.ArgumentParser(description="Event-timing: ds-drop vs drawdown onset.")
    parser.add_argument("--market", choices=list(MARKET_TICKER.keys()), default="spx")
    parser.add_argument("--no-plot", action="store_true", help="Skip updating the overlay plot.")
    parser.add_argument("--full-table", action="store_true", help="Also write full granular lead table (default: coarse summary only).")
    args = parser.parse_args()
    benchmark_ticker = MARKET_TICKER[args.market]
    suffix = "_" + benchmark_ticker.replace("^", "").lower() if benchmark_ticker != "^GSPC" else ""

    data_dir = PROJECT_ROOT / "data"
    ds_path = data_dir / "ds_series.csv"
    if not ds_path.exists():
        print("Missing data/ds_series.csv. Run the main pipeline first.")
        sys.exit(1)
    df = pd.read_csv(ds_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ds_series = df.iloc[:, 0] if df.ndim > 1 else df.squeeze()
    print(f"Loaded ds_series: {len(ds_series)} dates.")

    event = run_event_timing(ds_series, data_dir=data_dir, benchmark_ticker=benchmark_ticker)
    lead_df = event["lead_table"]

    # Coarse crisis summary: one row per major episode, -20% threshold only (no granular -10/-20/-30)
    coarse = lead_df[lead_df["drawdown_threshold_pct"] == -20.0].copy() if not lead_df.empty and "drawdown_threshold_pct" in lead_df.columns else lead_df
    if coarse.empty and not lead_df.empty:
        coarse = lead_df.drop_duplicates(subset=["drawdown_date"], keep="first").head(20)
    coarse_path = data_dir / f"event_timing_crisis_summary{suffix}.csv"
    coarse.to_csv(coarse_path, index=False)
    print("Coarse crisis summary (e.g. -20% only):")
    print(coarse.to_string(index=False))
    print(f"Saved: {coarse_path}")

    if args.full_table:
        print(f"Full table: {event['csv_path']}")
    else:
        # Overwrite the full granular file only if we're not keeping it; actually we always write it in run_event_timing. So we just don't print it.
        pass

    print(f"Avg lead (ds before drawdown): {event['summary'].get('avg_lead_days')}; % ds leads: {event['summary'].get('pct_crises_ds_leads')}; FP/yr: {event['summary'].get('false_positives_per_year')}")

    if not args.no_plot:
        start = ds_series.index.min().strftime("%Y-%m-%d")
        end = ds_series.index.max().strftime("%Y-%m-%d")
        vix, _ = load_vix_and_benchmark(start, end, benchmark_ticker=benchmark_ticker)
        plot_ds_with_overlays(
            ds_series,
            data_dir / f"ds_with_crisis_overlays{suffix}.png",
            vix=vix,
            drawdown=event["drawdown"],
            ds_drop_dates=event["ds_drop_dates"],
            drawdown_crossing_dates=event["onset_table"]["drawdown_date"].tolist() if not event["onset_table"].empty else [],
        )
        print(f"Updated plot: ds_with_crisis_overlays{suffix}.png")


if __name__ == "__main__":
    main()
