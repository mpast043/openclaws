"""
Phase 3 — Spectral dimension time series.
L = D - W, eigenvalues, heat trace Z(σ), ds(σ) = -2 d(log Z)/d(log σ),
plateau extraction (sliding IQR 5, IQR < 0.05, longest region, mean ds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


# Log-spaced σ grid: 10^-2 to 10^2, 50 points
SIGMA_GRID = np.logspace(-2, 2, 50)
IQR_WINDOW = 5
IQR_THRESHOLD = 0.05


def laplacian(W: np.ndarray) -> np.ndarray:
    """L = D - W, symmetric. D_ii = sum_j W_ij."""
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    L = 0.5 * (L + L.T)
    return L


def heat_trace(eigenvalues: np.ndarray, sigma: float) -> float:
    """Z(σ) = sum_i exp(-σ λ_i)."""
    return np.sum(np.exp(-sigma * eigenvalues))


def spectral_dimension_curve(eigenvalues: np.ndarray, sigma_grid: np.ndarray) -> np.ndarray:
    """
    ds(σ) = -2 * d(log Z)/d(log σ).
    Central differences in log-space.
    """
    log_sigma = np.log(sigma_grid)
    log_Z = np.array([np.log(heat_trace(eigenvalues, s)) for s in sigma_grid])
    n = len(sigma_grid)
    ds = np.full(n, np.nan)
    for i in range(1, n - 1):
        d_log_Z = (log_Z[i + 1] - log_Z[i - 1]) / (log_sigma[i + 1] - log_sigma[i - 1])
        ds[i] = -2.0 * d_log_Z
    return ds


def sliding_iqr(x: np.ndarray, window: int) -> np.ndarray:
    """IQR over sliding window of `window` adjacent points."""
    n = len(x)
    iqr = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 2:
            iqr[i] = np.percentile(seg, 75) - np.percentile(seg, 25)
    return iqr


def plateau_region(ds: np.ndarray, sigma_grid: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Sliding IQR over 5 adjacent σ. Identify region where IQR < 0.05.
    If multiple, choose longest contiguous. Return (start_idx, end_idx) inclusive.
    """
    iqr = sliding_iqr(ds, IQR_WINDOW)
    below = iqr < IQR_THRESHOLD
    n = len(ds)
    best_start, best_end = None, None
    best_len = 0
    i = 0
    while i < n:
        if not below[i] or np.isnan(iqr[i]):
            i += 1
            continue
        start = i
        while i < n and below[i]:
            i += 1
        end = i - 1
        length = end - start + 1
        if length > best_len:
            best_len = length
            best_start, best_end = start, end
    return (best_start, best_end) if best_start is not None else None


def plateau_mean_ds(ds: np.ndarray, sigma_grid: np.ndarray) -> Optional[float]:
    """
    Longest contiguous plateau (IQR < 0.05 over 5 σ), mean ds over that region.
    Single ds value per window. None if no plateau.
    """
    region = plateau_region(ds, sigma_grid)
    if region is None:
        return None
    start, end = region
    seg = ds[start : end + 1]
    seg = seg[~np.isnan(seg)]
    if len(seg) == 0:
        return None
    return float(np.mean(seg))


def eigenvalues_laplacian(L: np.ndarray, n_assets: int) -> np.ndarray:
    """Full eigendecomposition if N < 600; else symmetric eigh (all eigenvalues for heat trace)."""
    eigs = np.linalg.eigvalsh(L)
    return np.maximum(eigs, 0.0)


def compute_ds_series(
    returns: pd.DataFrame,
    window: int = 60,
    alpha: float = 0.05,
    use_abs: bool = True,
    sigma_grid: Optional[np.ndarray] = None,
    progress_interval: Optional[int] = 500,
):
    """
    For each rolling window: Laplacian -> eigenvalues -> ds plateau.
    Yields (date, ds_value). ds_value is None if no plateau.
    If progress_interval is set, print progress every N windows.
    """
    from .correlations import rolling_correlation_matrices, correlation_to_adjacency

    if sigma_grid is None:
        sigma_grid = SIGMA_GRID
    n_done = 0
    for date, C in rolling_correlation_matrices(returns, window=window, alpha=alpha, use_abs=use_abs):
        W = correlation_to_adjacency(C, use_abs=use_abs)
        L = laplacian(W)
        n = L.shape[0]
        eigs = eigenvalues_laplacian(L, n)
        ds_curve = spectral_dimension_curve(eigs, sigma_grid)
        ds_val = plateau_mean_ds(ds_curve, sigma_grid)
        n_done += 1
        if progress_interval and n_done % progress_interval == 0:
            print(f"    windows done: {n_done} (last date {date.date()})", flush=True)
        yield date, ds_val
