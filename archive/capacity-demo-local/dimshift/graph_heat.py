"""
Heat-kernel trace estimation on sparse graphs via Stochastic Lanczos Quadrature.

For graphs small enough (n_total < EXACT_THRESHOLD), uses full eigendecomposition.
For larger graphs, uses SLQ with Rademacher probe vectors.

This module provides the same interface as spectral.py but works with arbitrary
sparse Laplacians (not just separable lattice products).
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh


# Graphs below this size use exact eigendecomposition
EXACT_THRESHOLD = 8000


# ---------------------------------------------------------------------------
# Exact trace via full eigendecomposition
# ---------------------------------------------------------------------------

def _trace_exact(
    L: sparse.csr_matrix,
    sigma_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Exact Tr(exp(-σL)) / n via full eigendecomposition.

    Returns ln_P(σ) = ln( (1/n) Σ_k exp(-σ λ_k) ) for each σ.
    """
    n = L.shape[0]
    # For small matrices, convert to dense and use numpy eigh
    if n <= 2000:
        eigs = np.linalg.eigvalsh(L.toarray())
    else:
        # Use sparse eigsh for all eigenvalues
        eigs = eigsh(L, k=n - 1, which="SM", return_eigenvectors=False)
        eigs = np.append(eigs, 0.0)  # zero eigenvalue

    eigs = np.sort(np.maximum(eigs, 0.0))  # clip tiny negatives

    # ln P(σ) = ln( (1/n) Σ_k exp(-σ λ_k) )
    # Use log-sum-exp for stability
    ln_P = np.zeros(len(sigma_values))
    for i, sigma in enumerate(sigma_values):
        exponents = -sigma * eigs
        max_exp = np.max(exponents)
        ln_P[i] = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))

    return ln_P


# ---------------------------------------------------------------------------
# Lanczos iteration
# ---------------------------------------------------------------------------

def _lanczos(
    L: sparse.csr_matrix,
    v0: NDArray[np.float64],
    m: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Lanczos iteration: L, v0 → tridiagonal (alpha, beta).

    Parameters
    ----------
    L : sparse matrix (n, n)
    v0 : starting vector (n,), assumed unit norm
    m : number of Lanczos steps

    Returns
    -------
    alpha : ndarray (m,) — diagonal of tridiagonal T
    beta : ndarray (m-1,) — off-diagonal of tridiagonal T
    """
    n = L.shape[0]
    m = min(m, n)

    alpha = np.zeros(m)
    beta = np.zeros(m - 1)

    v_prev = np.zeros(n)
    v_curr = v0.copy()

    for j in range(m):
        w = L @ v_curr
        alpha[j] = np.dot(v_curr, w)
        if j > 0:
            w -= beta[j - 1] * v_prev
        w -= alpha[j] * v_curr

        # Partial reorthogonalization against v_curr and v_prev
        w -= np.dot(v_curr, w) * v_curr
        if j > 0:
            w -= np.dot(v_prev, w) * v_prev

        beta_val = np.linalg.norm(w)
        if beta_val < 1e-14:
            # Early termination — invariant subspace found
            alpha = alpha[:j + 1]
            beta = beta[:j]
            break

        if j < m - 1:
            beta[j] = beta_val
            v_prev = v_curr.copy()
            v_curr = w / beta_val

    return alpha, beta


# ---------------------------------------------------------------------------
# SLQ trace estimation
# ---------------------------------------------------------------------------

def _trace_slq(
    L: sparse.csr_matrix,
    sigma_values: NDArray[np.float64],
    n_probes: int,
    m_lanczos: int,
    seed: int,
) -> NDArray[np.float64]:
    """
    Stochastic Lanczos Quadrature for Tr(exp(-σL)) / n.

    Returns ln_P(σ) for each σ.
    """
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    n_sigma = len(sigma_values)

    # Accumulate in log space
    ln_P_accum = np.full(n_sigma, -np.inf)

    for p in range(n_probes):
        # Rademacher probe: ±1 with equal probability
        v = rng.choice([-1.0, 1.0], size=n)
        v_norm = np.linalg.norm(v)
        v_unit = v / v_norm

        alpha, beta = _lanczos(L, v_unit, m_lanczos)
        m_actual = len(alpha)

        # Eigendecompose the tridiagonal matrix T once per probe
        if m_actual == 1:
            theta = alpha.copy()
            S = np.ones((1, 1))
        else:
            theta, S = eigh_tridiagonal(alpha, beta[:m_actual - 1])

        theta = np.maximum(theta, 0.0)  # clip negatives

        # Quadrature weights: τ_j = S[0, j] (first component of each eigenvector)
        tau_sq = (S[0, :]) ** 2

        # Single Lanczos pass, multi-sigma quadrature
        # For all σ at once: exp_terms shape (n_sigma, m_actual)
        exponents = -np.outer(sigma_values, theta)
        max_exp = np.max(exponents, axis=1)
        exp_terms = np.exp(exponents - max_exp[:, None])
        quad_vals = exp_terms @ tau_sq
        quad_vals = np.maximum(quad_vals, 1e-300)
        ln_contrib = max_exp + np.log(quad_vals)

        if p == 0:
            ln_P_accum = ln_contrib
        else:
            a = np.maximum(ln_P_accum, ln_contrib)
            b = np.minimum(ln_P_accum, ln_contrib)
            ln_P_accum = a + np.log1p(np.exp(b - a))

    # Average over probes: ln(sum/n_probes) = ln(sum) - ln(n_probes)
    ln_P = ln_P_accum - np.log(n_probes)

    return ln_P


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_return_probability_sparse(
    L: sparse.csr_matrix,
    sigma_values: NDArray[np.float64],
    n_probes: int = 30,
    m_lanczos: int = 80,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Log return probability ln P(σ) = ln(Tr(exp(-σL)) / n) on a sparse graph.

    Uses exact eigendecomposition for n < EXACT_THRESHOLD, SLQ otherwise.

    Parameters
    ----------
    L : sparse CSR matrix (n, n) — graph Laplacian
    sigma_values : diffusion time grid
    n_probes : number of Rademacher probe vectors (SLQ only)
    m_lanczos : Lanczos iteration depth (SLQ only)
    seed : RNG seed for probe vectors

    Returns
    -------
    ln_P : ndarray of shape (n_sigma,)
    """
    sigma_values = np.asarray(sigma_values, dtype=float)
    if np.any(sigma_values <= 0):
        raise ValueError("sigma_values must be strictly positive")
    if np.any(np.diff(sigma_values) < 0):
        raise ValueError("sigma_values must be non-decreasing for multi-sigma SLQ")

    n = L.shape[0]
    if n < EXACT_THRESHOLD:
        return _trace_exact(L, sigma_values)
    else:
        return _trace_slq(L, sigma_values, n_probes, m_lanczos, seed)


def spectral_dimension_sparse(
    sigma_values: NDArray[np.float64],
    ln_P_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Spectral dimension from log return probability.

    d_s(σ) = -2 d(ln P) / d(ln σ)

    Same formula as spectral.spectral_dimension but accepts any ln_P source.
    """
    ln_sigma = np.log(sigma_values)
    d_lnP = np.gradient(ln_P_values, ln_sigma)
    return -2.0 * d_lnP
