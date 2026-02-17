"""
Phase 2 — Returns & correlations.
Log returns, rolling 60-day window, correlation + shrinkage (α=0.05).
Symmetric, no full 3D tensor stored; yields (date, C_shrunk) per window.
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def shrink_correlation(C: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    C_shrunk = (1 - α) * C + α * I.
    Matrix must be symmetric and positive semi-definite.
    """
    n = C.shape[0]
    C_s = (1.0 - alpha) * C + alpha * np.eye(n)
    C_s = 0.5 * (C_s + C_s.T)
    eigs = np.linalg.eigvalsh(C_s)
    if eigs.min() < -1e-10:
        C_s = C_s + (1e-9 - eigs.min()) * np.eye(n)
    return C_s


def correlation_to_adjacency(C: np.ndarray, use_abs: bool = True) -> np.ndarray:
    """
    tau = 0.4. W_ij = max(|C_ij| - tau, 0), diagonal 0, symmetric.
    """
    tau = 0.4
    W = np.maximum(np.abs(C) - tau, 0)
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    return W


def rolling_correlation_matrices(
    returns: pd.DataFrame,
    window: int = 60,
    alpha: float = 0.05,
    use_abs: bool = True,
):
    """
    For each rolling window: compute correlation, shrink, ensure symmetry.
    Yields (window_end_date, C_shrunk) to avoid storing full 3D tensor.
    """
    r = returns.values
    dates = returns.index
    n_days, n_assets = r.shape
    for i in range(window, n_days + 1):
        block = r[i - window : i]
        if np.any(np.isnan(block)):
            continue
        C = np.corrcoef(block.T)
        if np.any(np.isnan(C)):
            continue
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C = 0.5 * (C + C.T)
        C_s = shrink_correlation(C, alpha=alpha)
        yield dates[i - 1], C_s
