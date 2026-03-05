"""
Unified spectral-dimension computation from eigenvalues + spectral filter.

Generic framework pipeline:
    eigenvalues + filter g_C -> P_C(sigma) -> d_s^C(sigma) -> plateau

Reuses the existing derivative estimator (np.gradient) from spectral.py.
Works with arbitrary substrates and filters.
"""

import numpy as np
from numpy.typing import NDArray

from .spectral import spectral_dimension


# ---------------------------------------------------------------------------
# Core: P_C(sigma) from eigenvalues and filter weights
# ---------------------------------------------------------------------------

def filtered_log_return_probability(
    eigenvalues: NDArray[np.float64],
    filter_weights: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Log filtered return probability.

    ln P_C(sigma) = ln [ (1/|V|) sum_i g_C(lambda_i) exp(-sigma lambda_i) ]

    Parameters
    ----------
    eigenvalues : ndarray of shape (n,)
        Sorted Laplacian eigenvalues of the substrate.
    filter_weights : ndarray of shape (n,)
        Filter values g_C(lambda_i) in [0, 1].
    sigma_values : ndarray of shape (M,)
        Diffusion time grid (log-spaced recommended).

    Returns
    -------
    ln_P : ndarray of shape (M,)
    """
    n = len(eigenvalues)
    active = filter_weights > 1e-300
    if not np.any(active):
        return np.full(len(sigma_values), -np.inf)

    log_w = np.full(len(filter_weights), -np.inf)
    log_w[active] = np.log(filter_weights[active])

    # log_summands[m, i] = -sigma[m]*lambda[i] + log(g[i])
    log_summands = -sigma_values[:, None] * eigenvalues[None, :] + log_w[None, :]

    # log-sum-exp over eigenvalue axis (active modes only)
    active_summands = log_summands[:, active]
    max_val = np.max(active_summands, axis=1, keepdims=True)
    log_sum = max_val[:, 0] + np.log(
        np.sum(np.exp(active_summands - max_val), axis=1)
    )
    ln_P = log_sum - np.log(n)

    return ln_P


def filtered_return_probability(
    eigenvalues: NDArray[np.float64],
    filter_weights: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """P_C(sigma) = exp(ln P_C(sigma))."""
    return np.exp(filtered_log_return_probability(
        eigenvalues, filter_weights, sigma_values
    ))


def filtered_spectral_dimension(
    eigenvalues: NDArray[np.float64],
    filter_weights: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute d_s^C(sigma) from eigenvalues and filter weights.

    Returns (ds, ln_P) tuple.
    """
    ln_P = filtered_log_return_probability(eigenvalues, filter_weights, sigma_values)
    ds = spectral_dimension(sigma_values, ln_P)
    return ds, ln_P


# ---------------------------------------------------------------------------
# Plateau extraction (fixed window, same logic as canonical demo)
# ---------------------------------------------------------------------------

def extract_plateau(
    ds: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
    sigma_lo: float,
    sigma_hi: float,
) -> dict:
    """
    Extract plateau summary from d_s(sigma) in a fixed window.

    The window [sigma_lo, sigma_hi] is set per-run, NOT per-C.
    """
    mask = (sigma_values >= sigma_lo) & (sigma_values <= sigma_hi)
    n_pts = int(np.sum(mask))

    if n_pts < 3:
        n = len(sigma_values)
        mask = np.zeros(n, dtype=bool)
        mask[n // 3: 2 * n // 3] = True
        n_pts = int(np.sum(mask))

    ds_window = ds[mask]
    return {
        "ds_plateau": float(np.median(ds_window)),
        "ds_mean": float(np.mean(ds_window)),
        "ds_std": float(np.std(ds_window)),
        "n_points": n_pts,
    }


# ---------------------------------------------------------------------------
# Scaling-window assumption check
# ---------------------------------------------------------------------------

def check_scaling_assumption(
    ln_P: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
    sigma_lo: float,
    sigma_hi: float,
) -> dict:
    """
    Check the power-law scaling assumption in a declared window.

    Fit ln P vs ln sigma linearly:
        ln P(sigma) ~ a - (d_eff/2) ln sigma

    Report R^2, max absolute residual, fitted slope.
    """
    mask = (sigma_values >= sigma_lo) & (sigma_values <= sigma_hi)
    n_pts = int(np.sum(mask))

    if n_pts < 5:
        return {
            "r_squared": float("nan"),
            "max_residual": float("nan"),
            "fitted_slope": float("nan"),
            "fitted_d_eff": float("nan"),
            "n_points": n_pts,
            "warning": "Too few points in scaling window",
        }

    ln_sigma = np.log(sigma_values[mask])
    ln_P_win = ln_P[mask]

    finite = np.isfinite(ln_P_win) & np.isfinite(ln_sigma)
    if np.sum(finite) < 5:
        return {
            "r_squared": float("nan"),
            "max_residual": float("nan"),
            "fitted_slope": float("nan"),
            "fitted_d_eff": float("nan"),
            "n_points": int(np.sum(finite)),
            "warning": "Too few finite points in scaling window",
        }

    ln_sigma = ln_sigma[finite]
    ln_P_win = ln_P_win[finite]

    coeffs = np.polyfit(ln_sigma, ln_P_win, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    fitted = intercept + slope * ln_sigma
    residuals = ln_P_win - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ln_P_win - np.mean(ln_P_win))**2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    max_residual = float(np.max(np.abs(residuals)))
    fitted_d_eff = -2.0 * slope

    return {
        "r_squared": round(r_squared, 6),
        "max_residual": round(max_residual, 6),
        "fitted_slope": round(slope, 6),
        "fitted_d_eff": round(fitted_d_eff, 4),
        "n_points": int(np.sum(finite)),
    }
