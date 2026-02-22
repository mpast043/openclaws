"""
Data fetching utilities.
"""

import yfinance as yf
import pandas as pd


def fetch_crypto_data(
    symbol: str = 'BTC-USD',
    period: str = '2y',
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Fetch cryptocurrency data from Yahoo Finance.
    
    Parameters
    ----------
    symbol : str
        Crypto symbol (BTC-USD, ETH-USD, etc.)
    period : str
        Data period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
    interval : str
        Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
    
    Returns
    -------
    data : pd.DataFrame
        OHLCV data with datetime index.
    """
    print(f"Fetching {symbol} data ({period}, {interval})...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
    
    # Clean up column names (remove spaces)
    data.columns = [c.replace(' ', '') for c in data.columns]
    
    print(f"Loaded {len(data)} rows: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    return data


def fetch_forex_data(
    symbol: str = 'EURUSD=X',
    period: str = '2y',
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Fetch forex data from Yahoo Finance.
    
    Note: Yahoo forex data has 1-hour delay and limited history.
    """
    return fetch_crypto_data(symbol, period, interval)


def fetch_equity_data(
    symbol: str = 'SPY',
    period: str = '2y',
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Fetch equity ETF data (proxy for market).
    """
    return fetch_crypto_data(symbol, period, interval)
