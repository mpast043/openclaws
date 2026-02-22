"""
Trading signals with capacity filtering.
"""

import numpy as np
import pandas as pd
from .filter import capacity_filter_time_series


def ma_signal(
    data: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Simple moving average crossover signal.
    
    Returns:
        DataFrame with 'short_ma', 'long_ma', 'signal' columns.
        signal: 1 = long, -1 = short, 0 = neutral
    """
    result = data.copy()
    result['short_ma'] = data[price_col].rolling(window=short_window).mean()
    result['long_ma'] = data[price_col].rolling(window=long_window).mean()
    
    # Generate signals
    result['signal'] = np.where(
        result['short_ma'] > result['long_ma'], 1,
        np.where(result['short_ma'] < result['long_ma'], -1, 0)
    )
    
    return result


def capacity_ma_signal(
    data: pd.DataFrame,
    capacity: float = 0.5,
    short_window: int = 5,
    long_window: int = 20,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Moving average crossover with capacity-constrained filtering.
    
    First filters data to information-rich subset, then computes MAs.
    """
    # Filter to capacity-constrained subset
    filtered = capacity_filter_time_series(data, capacity, window=short_window, price_col=price_col)
    
    # Compute MAs on filtered data (sparse, irregular timestamps)
    filtered['short_ma'] = filtered[price_col].rolling(window=short_window).mean()
    filtered['long_ma'] = filtered[price_col].rolling(window=long_window).mean()
    
    # Generate signals
    filtered['signal'] = np.where(
        filtered['short_ma'] > filtered['long_ma'], 1,
        np.where(filtered['short_ma'] < filtered['long_ma'], -1, 0)
    )
    
    return filtered


def momentum_signal(
    data: pd.DataFrame,
    lookback: int = 10,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Simple momentum signal: positive if price up over lookback period.
    """
    result = data.copy()
    result['momentum'] = data[price_col] - data[price_col].shift(lookback)
    result['signal'] = np.where(result['momentum'] > 0, 1, -1)
    return result


def capacity_momentum_signal(
    data: pd.DataFrame,
    capacity: float = 0.5,
    lookback: int = 10,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Momentum signal with capacity filtering.
    """
    filtered = capacity_filter_time_series(data, capacity, window=lookback, price_col=price_col)
    filtered['momentum'] = filtered[price_col] - filtered[price_col].shift(lookback)
    filtered['signal'] = np.where(filtered['momentum'] > 0, 1, -1)
    return filtered
