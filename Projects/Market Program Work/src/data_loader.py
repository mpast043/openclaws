"""
Phase 1 — Data: S&P 500 daily adjusted close, 2000-01-01 to 2024-12-31.
Drop tickers with >10% missing. Save to data/sp500_prices.csv.
"""

from __future__ import annotations

import io
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple


def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 constituent symbols from Wikipedia."""
    import requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def load_prices(
    start_date: str = "2000-01-01",
    end_date: str = "2024-12-31",
    max_missing_frac: float = 0.10,
    data_dir: Union[str, Path] = "data",
) -> Tuple[pd.DataFrame, int, Tuple[str, str]]:
    """
    Download daily adjusted close for S&P 500 constituents via yfinance.
    Drop tickers with more than max_missing_frac missing. Save cleaned prices.
    Returns (prices_df, n_tickers, (first_date, last_date)).
    """
    import yfinance as yf

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "sp500_prices.csv"

    tickers = get_sp500_tickers()
    print(f"Downloading {len(tickers)} S&P 500 constituents...")
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance with group_by="ticker" -> columns (Ticker, OHLC); some tickers may fail
    if isinstance(prices.columns, pd.MultiIndex):
        # Level 1 = Open/High/Low/Close/Volume; extract Close for each ticker
        closes = prices.xs("Close", axis=1, level=1).copy()
        if isinstance(closes.columns, pd.MultiIndex):
            closes = closes.droplevel(1, axis=1)
    else:
        closes = prices[["Close"]].copy()
        closes.columns = [tickers[0]] if len(tickers) == 1 else tickers
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    # Keep only columns that have data (drop failed downloads)
    closes = closes.dropna(axis=1, how="all").dropna(axis=0, how="all")
    prices = closes

    prices = prices.loc[start_date:end_date]
    n_expected = len(prices)
    missing_frac = prices.isna().sum() / n_expected
    keep = missing_frac <= max_missing_frac
    prices = prices.loc[:, keep]
    n_tickers = prices.shape[1]
    prices = prices.dropna(how="all")
    first_date = prices.index.min().strftime("%Y-%m-%d")
    last_date = prices.index.max().strftime("%Y-%m-%d")

    prices.to_csv(out_path)
    print(f"Saved {n_tickers} tickers, {first_date} to {last_date} -> {out_path}")
    return prices, n_tickers, (first_date, last_date)
