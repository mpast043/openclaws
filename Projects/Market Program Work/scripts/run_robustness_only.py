#!/usr/bin/env python3
"""
Standalone robustness: signed vs absolute, alpha tuning, multiple window sizes.
Reads data/sp500_prices.csv and data/ds_series.csv; recomputes ds for each config and compares to baseline.
Run after the main pipeline has produced prices and baseline ds.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.correlations import log_returns
from src.robustness import run_robustness


def main():
    data_dir = PROJECT_ROOT / "data"
    prices_path = data_dir / "sp500_prices.csv"
    if not prices_path.exists():
        print("Missing data/sp500_prices.csv. Run the main pipeline first.")
        sys.exit(1)
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    print(f"Loaded {prices.shape[1]} tickers, {len(prices)} dates.")
    robustness_df = run_robustness(prices, data_dir=data_dir)
    print(robustness_df.to_string(index=False))
    print(f"Saved: {data_dir / 'robustness_sensitivity.csv'}")


if __name__ == "__main__":
    main()
