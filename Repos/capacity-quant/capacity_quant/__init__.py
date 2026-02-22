"""Capacity-Quant: Signal filtering using capacity-constrained information geometry."""

from .filter import capacity_filter_time_series, spectral_selectivity
from .signals import ma_signal, momentum_signal
from .backtest import backtest_strategy
from .data import fetch_crypto_data

__all__ = [
    'capacity_filter_time_series',
    'spectral_selectivity',
    'ma_signal',
    'momentum_signal',
    'backtest_strategy',
    'fetch_crypto_data',
]
