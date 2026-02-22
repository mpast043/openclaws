"""
Simple backtest engine for strategy comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def backtest_strategy(
    data: pd.DataFrame,
    signal_col: str = 'signal',
    price_col: str = 'Close',
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001  # 0.1% per trade
) -> Dict:
    """
    Simple backtest: buy/sell based on signals.
    
    Parameters
    ----------
    data : pd.DataFrame
        Must have signal_col (-1, 0, or 1) and price_col
    signal_col : str
        Column with trading signals
    price_col : str
        Column with price data
    initial_capital : float
        Starting capital
    transaction_cost : float
        Cost per trade as fraction of position
    
    Returns
    -------
    results : dict
        - total_return: float
        - sharpe_ratio: float
        - max_drawdown: float
        - num_trades: int
        - win_rate: float
        - equity_curve: pd.Series
    """
    prices = data[price_col].values
    signals = data[signal_col].values
    
    capital = initial_capital
    position = 0  # -1, 0, or 1
    trades = []
    equity = [initial_capital]
    
    for i in range(1, len(prices)):
        # Check for signal change
        if signals[i] != position and signals[i] != 0:
            # Execute trade
            if position != 0:
                # Close existing position
                capital *= (1 - transaction_cost)
            
            position = signals[i]
            capital *= (1 - transaction_cost)  # Entry cost
            trades.append({
                'time': data.index[i],
                'action': 'BUY' if position == 1 else 'SELL',
                'price': prices[i]
            })
        
        # Update equity
        if position == 1:
            # Long: profit from price increase
            pnl = (prices[i] / prices[i-1] - 1) * capital
        elif position == -1:
            # Short: profit from price decrease
            pnl = (prices[i-1] / prices[i] - 1) * capital
        else:
            pnl = 0
        
        capital += pnl
        equity.append(capital)
    
    # Calculate metrics
    equity_series = pd.Series(equity, index=data.index)
    returns = equity_series.pct_change().dropna()
    
    total_return = (equity[-1] / initial_capital) - 1
    
    # Sharpe ratio (annualized, assuming daily data)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = sum(1 for t in trades if t['action'] == 'BUY')  # Simplified
    win_rate = len(trades) > 0 and winning_trades / len(trades) or 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'equity_curve': equity_series,
        'trades': trades
    }


def compare_strategies(
    data: pd.DataFrame,
    strategies: Dict[str, pd.DataFrame],
    initial_capital: float = 10000.0
) -> pd.DataFrame:
    """
    Compare multiple strategies on same data.
    
    Parameters
    ----------
    strategies : dict
        {name: dataframewith_signal_column, ...}
    
    Returns
    -------
    comparison : pd.DataFrame
        Metrics for each strategy.
    """
    results = []
    
    for name, strat_data in strategies.items():
        metrics = backtest_strategy(strat_data, initial_capital=initial_capital)
        results.append({
            'strategy': name,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate']
        })
    
    return pd.DataFrame(results)
