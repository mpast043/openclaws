"""
Spectral dimension computation via heat-kernel return probability.

Mathematical basis
------------------
On a D-dimensional periodic cubic lattice with side N:

    Eigenvalues:  λ(k₁,...,k_D) = Σ_d  2(1 - cos(2π k_d / N))
    Return prob:  P(σ) = (1/N^D) Σ_k exp(-σ λ(k))
    Spectral dim: d_s(σ) = -2 d(ln P) / d(ln σ)

Capacity filtering scales each dimension's eigenvalue contribution:

    λ_w(k) = Σ_d  w_d · λ_1D(k_d)

Because the eigenvalues separate, P factorises exactly:

    P(σ) = Π_{d=1}^D  P_1D(w_d σ)

where  P_1D(t) = (1/N) Σ_{k=0}^{N-1} exp(-t λ_1D(k)).

Complexity: O(D · N · |Σ|) where |Σ| is the number of σ points.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def eigenvalues_1d(N: int) -> NDArray[np.float64]:
    """
    1D periodic lattice Laplacian eigenvalues.

    λ_1D(k) = 2(1 - cos(2πk/N)),  k = 0, ..., N-1.

    Parameters
    ----------
    N : int
        Lattice side length (must be >= 2).

    Returns
    -------
    eigs : ndarray of shape (N,)
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    k = np.arange(N, dtype=np.float64)
    return 2.0 * (1.0 - np.cos(2.0 * np.pi * k / N))


def p1d(eigs_1d: NDArray[np.float64], t_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    1D return probability on a periodic chain.

    P_1D(t) = (1/N) Σ_{k=0}^{N-1} exp(-t λ_1D(k))

    Parameters
    ----------
    eigs_1d : ndarray of shape (N,)
        Precomputed 1D eigenvalues from eigenvalues_1d().
    t_values : ndarray of shape (M,)
        Effective diffusion times (w_d · σ).

    Returns
    -------
    P : ndarray of shape (M,)
    """
    # eigs_1d: (N,), t_values: (M,) → exp term: (N, M)
    return np.mean(np.exp(-eigs_1d[:, None] * t_values[None, :]), axis=0)


def log_return_probability(
    eigs_1d: NDArray[np.float64],
    weights: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Log return probability with capacity-filtered factorisation.

    ln P(σ) = Σ_{d=1}^D  ln P_1D(w_d σ)

    Computed in log space for numerical stability.

    Parameters
    ----------
    eigs_1d : ndarray of shape (N,)
        Precomputed 1D eigenvalues.
    weights : ndarray of shape (D,)
        Per-dimension capacity weights.
    sigma_values : ndarray of shape (M,)
        Diffusion time grid.

    Returns
    -------
    ln_P : ndarray of shape (M,)
    """
    ln_P = np.zeros_like(sigma_values)
    for w in weights:
        if w > 1e-15:
            P_d = p1d(eigs_1d, w * sigma_values)
            ln_P += np.log(np.maximum(P_d, 1e-300))
        # w ≈ 0  →  P_1D(0) = 1  →  ln(1) = 0, no contribution
    return ln_P


def batch_log_return_probability(
    eigs_1d: NDArray[np.float64],
    weights_matrix: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
    *,
    eig_sigma: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Batch log return probability for multiple capacity settings.

    Parameters
    ----------
    eigs_1d : ndarray of shape (N,)
        1D eigenvalues.
    weights_matrix : ndarray of shape (n_C, D)
        Capacity weights for each C_geo value.
    sigma_values : ndarray of shape (M,)
        Diffusion grid shared by all sweeps.
    eig_sigma : ndarray of shape (N, M), optional
        Precomputed outer product eigs_1d[:, None] * sigma_values[None, :].

    Returns
    -------
    ln_P_matrix : ndarray of shape (n_C, M)
    """
    n_C = weights_matrix.shape[0]
    if eig_sigma is None:
        eig_sigma = eigs_1d[:, None] * sigma_values[None, :]
    ln_P = np.zeros((n_C, len(sigma_values)), dtype=np.float64)

    D = weights_matrix.shape[1]
    for d in range(D):
        weights = weights_matrix[:, d]
        mask = weights > 1e-15
        if not np.any(mask):
            continue
        w_active = weights[mask][:, None, None]
        scaled = np.exp(-eig_sigma[None, :, :] * w_active)
        P_d = scaled.mean(axis=1)
        ln_P[mask] += np.log(np.maximum(P_d, 1e-300))
    return ln_P


def return_probability(
    eigs_1d: NDArray[np.float64],
    weights: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Return probability P(σ) = exp(ln P(σ)).

    Parameters
    ----------
    eigs_1d : ndarray of shape (N,)
    weights : ndarray of shape (D,)
    sigma_values : ndarray of shape (M,)

    Returns
    -------
    P : ndarray of shape (M,)
    """
    return np.exp(log_return_probability(eigs_1d, weights, sigma_values))


def p1d_infinite_lattice(t_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Infinite discrete lattice (N→∞) closed form for the 1D return probability.

    On an infinite 1D lattice (N→∞ ring) with spacing a=1:

        P_1D^∞(t) = (1/(2π)) ∫_0^{2π} exp(-2t(1-cos θ)) dθ
                   = exp(-2t) I₀(2t)

    where I₀ is the modified Bessel function of the first kind.
    This is NOT the continuum limit on ℝ; it is the N→∞ limit on ℤ.

    For large t this simplifies to  P_1D(t) ~ 1/√(4πt).

    Requires SciPy (scipy.special.i0e).

    Parameters
    ----------
    t_values : ndarray of shape (M,)
        Effective diffusion times (w_d * sigma).  Must be > 0.

    Returns
    -------
    P : ndarray of shape (M,)
        Exact Bessel-form return probability for the infinite lattice.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    """
    from scipy.special import i0e  # exponentially scaled I_0
    # P_1D(t) = exp(-2t) I_0(2t) = i0e(2t) since i0e(x) = exp(-|x|) I_0(x)
    return i0e(2.0 * t_values)


def spectral_dimension(
    sigma_values: NDArray[np.float64],
    ln_P_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Spectral dimension from log return probability.

    d_s(σ) = -2 d(ln P) / d(ln σ)

    Uses central finite differences on the log-log data, with
    forward/backward at endpoints.

    Parameters
    ----------
    sigma_values : ndarray of shape (M,)
        Log-spaced diffusion time grid.
    ln_P_values : ndarray of shape (M,)
        ln P(σ) values (use log_return_probability for stability).

    Returns
    -------
    ds : ndarray of shape (M,)
    """
    ln_sigma = np.log(sigma_values)
    d_lnP = np.gradient(ln_P_values, ln_sigma)
    return -2.0 * d_lnP
