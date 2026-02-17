"""
Monotone spectral filters for Framework v4.5 Option B.

Each filter maps (eigenvalues, capacity C) -> weights g_C(λ) in [0, 1].
Monotonicity: for C1 <= C2, g_{C1}(λ) <= g_{C2}(λ) for all λ.

Within any single sweep, ONLY C varies.  All other filter parameters
(λ0, steepness s, baseline dimension d0) are fixed per-run and logged
in metadata.

Three filters implemented:
  1. HardCutoffFilter  -- rank-based low-pass, top-m modes
  2. SoftCutoffFilter  -- logistic taper by rank
  3. PowerLawFilter    -- power-law reweight (changes scaling exponent)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Weights returned by a filter application."""
    weights: NDArray[np.float64]   # shape (n_eigenvalues,), values in [0, 1]
    metadata: dict                 # filter params (all C-independent except C itself)


class SpectralFilter:
    """Base class for monotone spectral filters."""

    name: str = "base"

    def apply(self, eigenvalues: NDArray[np.float64], C: float) -> FilterResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Filter 1: Hard cutoff by rank (low-pass)
# ---------------------------------------------------------------------------

class HardCutoffFilter(SpectralFilter):
    """
    Keep the m(C) smallest eigenvalue modes with a fixed minimum floor.

    g_C(lambda_i) = 1  if i < m(C)  (sorted ascending)
    g_C(lambda_i) = 0  otherwise

    Parameters
    ----------
    floor_fraction : float
        Fraction of modes always kept (capacity-independent). Prevents the
        low-capacity limit from collapsing to zero dimension while staying
        capacity-only (constant per run).
    """

    name = "hard_cutoff"

    def __init__(self, floor_fraction: float = 0.05):
        self.floor_fraction = max(0.0, min(floor_fraction, 1.0))

    def apply(self, eigenvalues: NDArray[np.float64], C: float) -> FilterResult:
        n = len(eigenvalues)
        min_modes = max(1, int(np.ceil(self.floor_fraction * n)))
        m = max(min_modes, int(np.floor(C * n)))
        m = min(m, n)

        weights = np.zeros(n, dtype=np.float64)
        weights[:m] = 1.0

        return FilterResult(
            weights=weights,
            metadata={
                "filter": self.name,
                "C": C,
                "m": m,
                "n": n,
                "floor_fraction": self.floor_fraction,
            },
        )

    def describe(self) -> dict:
        return {"name": self.name, "type": "rank-based hard cutoff",
                "formula": "g_C(i) = 1 if i < floor(C*n), else 0"}


# ---------------------------------------------------------------------------
# Filter 2: Soft cutoff by rank (logistic taper)
# ---------------------------------------------------------------------------

class SoftCutoffFilter(SpectralFilter):
    """
    Logistic taper around a capacity-weighted rank with a fixed base fraction.

    g_C(i) = sigmoid((m_eff(C) - i) / s)

    where:
        m_eff(C) = n * (base_fraction + (1 - base_fraction) * C)
        s = steepness_frac * n (capacity-independent)

    The base_fraction ensures a minimal mass of modes stays active even at
    C -> 0, improving stability on noisy spectra while remaining
    capacity-only (base_fraction constant per run).
    """

    name = "soft_cutoff"

    def __init__(self, steepness_frac: float = 0.05, base_fraction: float = 0.05):
        self.steepness_frac = max(0.001, steepness_frac)
        self.base_fraction = min(max(base_fraction, 0.0), 0.5)

    def apply(self, eigenvalues: NDArray[np.float64], C: float) -> FilterResult:
        n = len(eigenvalues)
        m_eff = n * (self.base_fraction + (1.0 - self.base_fraction) * C)
        s = max(self.steepness_frac * n, 1.0)
        indices = np.arange(n, dtype=np.float64)
        z = (m_eff - indices) / s
        weights = 1.0 / (1.0 + np.exp(-z))
        weights = np.clip(weights, 0.0, 1.0)

        return FilterResult(
            weights=weights,
            metadata={
                "filter": self.name,
                "C": C,
                "m_eff": round(m_eff, 4),
                "s": round(s, 4),
                "steepness_frac": self.steepness_frac,
                "base_fraction": self.base_fraction,
                "n": n,
            },
        )

    def describe(self) -> dict:
        return {"name": self.name, "type": "rank-based logistic taper",
                "formula": "g_C(i) = sigmoid((C*n - i) / s)",
                "steepness_frac": self.steepness_frac}


# ---------------------------------------------------------------------------
# Filter 3: Power-law reweight (changes scaling exponent)
# ---------------------------------------------------------------------------

class PowerLawFilter(SpectralFilter):
    """
    Power-law spectral reweight that modifies the scaling exponent.

    Define:
        beta(C) = beta_max * (1 - C**gamma)
        lambda0 = quantile of nonzero eigenvalues (fixed per substrate)
        g_C(lambda) = (1 + lambda/lambda0)^{-beta(C)}

    Parameters
    ----------
    d0 : float
        Baseline dimensional prior (used if beta_max not provided).
    beta_max : float | None
        Maximum beta at C=0. Defaults to d0/2 to match prior behavior.
    gamma : float
        Shapes capacity response (gamma > 1 makes early capacities gentler).
    lambda0 : float | None
        Optional fixed scale. If None, computed once from eigenvalues.
    lambda0_quantile : float
        Quantile (0–1) used when estimating lambda0 from eigenvalues.
    """

    name = "powerlaw_reweight"

    def __init__(
        self,
        d0: float,
        beta_max: float | None = None,
        gamma: float = 1.0,
        lambda0: float | None = None,
        lambda0_quantile: float = 0.5,
    ):
        self.d0 = d0
        self.beta_max = beta_max if beta_max is not None else (d0 / 2.0)
        self.gamma = max(0.1, gamma)
        self.lambda0_quantile = min(max(lambda0_quantile, 0.0), 1.0)
        self._lambda0 = lambda0

    @property
    def lambda0(self) -> float | None:
        return self._lambda0

    def _ensure_lambda0(self, eigenvalues: NDArray[np.float64]) -> float:
        """Compute and cache lambda0 from eigenvalues (once per substrate)."""
        if self._lambda0 is not None:
            return self._lambda0
        nonzero = eigenvalues[eigenvalues > 1e-12]
        if len(nonzero) == 0:
            self._lambda0 = 1.0
        else:
            q = max(0.0, min(self.lambda0_quantile, 1.0))
            if q <= 0.0:
                self._lambda0 = float(np.min(nonzero))
            elif q >= 1.0:
                self._lambda0 = float(np.max(nonzero))
            else:
                self._lambda0 = float(np.quantile(nonzero, q))
        return self._lambda0

    def apply(self, eigenvalues: NDArray[np.float64], C: float) -> FilterResult:
        lam0 = self._ensure_lambda0(eigenvalues)
        beta = self.beta_max * (1.0 - float(C) ** self.gamma)

        ratio = 1.0 + eigenvalues / lam0
        if beta < 1e-15:
            weights = np.ones_like(eigenvalues)
        else:
            weights = np.power(ratio, -beta)

        weights = np.clip(weights, 0.0, 1.0)

        return FilterResult(
            weights=weights,
            metadata={
                "filter": self.name,
                "C": C,
                "beta": round(beta, 6),
                "beta_max": round(self.beta_max, 6),
                "gamma": self.gamma,
                "d0": self.d0,
                "lambda0": round(lam0, 6),
                "lambda0_quantile": self.lambda0_quantile,
                "n": len(eigenvalues),
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "type": "power-law spectral reweight",
            "formula": "g_C(lam) = (1 + lam/lam0)^{-beta_max * (1 - C^gamma)}",
            "d0": self.d0,
            "beta_max": self.beta_max,
            "gamma": self.gamma,
            "lambda0": self._lambda0,
            "lambda0_quantile": self.lambda0_quantile,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_FILTERS = {
    "hard_cutoff": HardCutoffFilter,
    "soft_cutoff": SoftCutoffFilter,
    "powerlaw_reweight": PowerLawFilter,
}


def get_filter(name: str, **kwargs) -> SpectralFilter:
    """Get a filter instance by name."""
    if name not in ALL_FILTERS:
        raise ValueError(f"Unknown filter: {name}. Available: {list(ALL_FILTERS.keys())}")
    return ALL_FILTERS[name](**kwargs)
