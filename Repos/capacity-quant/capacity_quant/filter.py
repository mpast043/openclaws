"""
Capacity-constrained filtering for time series data.

Translates the capacity framework to financial signals:
- Treat time series as observations on a temporal manifold
- Apply capacity constraints to select maximally-informative observations
- Return filtered subset for signal generation
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema


def spectral_entropy(signal: np.ndarray) -> float:
    """
    Compute spectral entropy of a signal.
    Higher entropy = more information content = more predictive value.
    """
    # FFT to frequency domain
    spectrum = np.abs(fft(signal))
    # Normalize to probability distribution
    power = spectrum ** 2
    prob = power / (power.sum() + 1e-10)
    # Shannon entropy
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    return entropy


def local_information_density(
    series: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Compute local information density at each point.
    Uses spectral entropy in a sliding window.
    
    Returns array of information scores (higher = more valuable).
    """
    n = len(series)
    scores = np.zeros(n)
    
    for i in range(window, n - window):
        # Local window around point i
        local_window = series[i - window:i + window]
        # Spectral entropy as information proxy
        scores[i] = spectral_entropy(local_window)
    
    # Edge cases: use nearest valid score
    scores[:window] = scores[window]
    scores[-window:] = scores[-window - 1]
    
    return scores


def capacity_select(
    indices: np.ndarray,
    scores: np.ndarray,
    capacity: float
) -> np.ndarray:
    """
    Select top `capacity` fraction of points by information score.
    
    Parameters
    ----------
    indices : np.ndarray
        All available indices.
    scores : np.ndarray
        Information density scores for each index.
    capacity : float
        Fraction of points to select (0 to 1).
    
    Returns
    -------
    selected_indices : np.ndarray
        Indices of selected points.
    """
    if not 0 < capacity <= 1:
        raise ValueError(f"Capacity must be in (0, 1], got {capacity}")
    
    n_select = max(1, int(len(indices) * capacity))
    
    # Select top n_select by score
    top_indices = np.argsort(scores)[-n_select:]
    selected = indices[top_indices]
    
    # Return in original temporal order
    return np.sort(selected)


def capacity_filter_time_series(
    data: pd.DataFrame,
    capacity: float = 0.5,
    window: int = 20,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Filter time series to capacity-constrained information-rich subset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Market data with datetime index and price column.
    capacity : float
        Fraction of data to keep (0 to 1). Lower = more selective.
    window : int
        Window size for local information density calculation.
    price_col : str
        Column name for price data.
    
    Returns
    -------
    filtered : pd.DataFrame
        Subset of original data with highest information density.
    """
    prices = data[price_col].values
    
    # Compute local information density
    info_scores = local_information_density(prices, window)
    
    # Capacity-constrained selection
    all_indices = np.arange(len(data))
    selected_indices = capacity_select(all_indices, info_scores, capacity)
    
    # Return filtered dataframe
    return data.iloc[selected_indices]


def spectral_selectivity(
    data: pd.DataFrame,
    price_col: str = 'Close'
) -> dict:
    """
    Analyze spectral properties of time series.
    Returns diagnostic metrics for tuning capacity.
    
    Returns
    -------
    metrics : dict
        - dominant_frequency
        - spectral_entropy
        - information_peak_ratio
    """
    prices = data[price_col].values
    
    # FFT analysis
    spectrum = np.abs(fft(prices))
    freqs = fftfreq(len(prices))
    
    # Dominant frequency
    positive_idx = freqs > 0
    dominant_idx = np.argmax(spectrum[positive_idx])
    dominant_freq = freqs[positive_idx][dominant_idx]
    
    # Spectral entropy
    power = spectrum ** 2
    prob = power / (power.sum() + 1e-10)
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    normalized_entropy = entropy / np.log(len(prob))  # 0 to 1
    
    # Information peak ratio (concentrated vs diffuse spectrum)
    peak_ratio = spectrum.max() / spectrum.mean()
    
    return {
        'dominant_frequency': dominant_freq,
        'spectral_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'peak_ratio': peak_ratio,
        'recommendation': 'low_capacity' if normalized_entropy > 0.7 else 'high_capacity'
    }
