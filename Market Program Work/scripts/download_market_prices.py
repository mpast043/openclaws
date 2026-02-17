#!/usr/bin/env python3
"""Download historical adjusted close prices from Yahoo Finance and save to CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex) and 'Adj Close' in raw.columns.levels[0]:
        data = raw['Adj Close']
    else:
        data = raw
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how='all')
    data.columns = [str(col).upper() for col in data.columns]
    return data

def main():
    parser = argparse.ArgumentParser(description="Download Yahoo Finance prices for multiple tickers")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols (e.g., SPY QQQ EEM IWM)")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--suffix", default="", help="Optional file suffix (e.g., _qqq)")
    args = parser.parse_args()

    prices = download_prices(args.tickers, args.start, args.end)
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix.strip()
    filename = f"prices{suffix}.csv" if suffix else "prices.csv"
    out_path = args.output / filename
    prices.to_csv(out_path, float_format="%.6f")
    print(f"Saved {len(prices)} rows to {out_path}")

if __name__ == "__main__":
    main()
