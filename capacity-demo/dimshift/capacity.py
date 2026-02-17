"""
Capacity weights and clamping logic.

The capacity C_geo ∈ [0, 1] maps to per-dimension weights w_d via:
    d_nom = C_geo * D
    w_d = clamp(d_nom - (d-1), 0, 1)   for d = 1, ..., D

This is the ONLY knob that changes across a sweep.
"""

import numpy as np
from numpy.typing import NDArray


def clamp01(x: float) -> float:
    """Clamp scalar x to [0, 1]."""
    return min(1.0, max(0.0, x))


def capacity_weights(C_geo: float, D: int) -> NDArray[np.float64]:
    """
    Per-dimension weights for capacity C_geo on a D-dimensional lattice.

    Uses nominal capacity dimension:
        d_nom = C_geo * D
        w_i   = clip(d_nom - i, 0, 1)   for i = 0, ..., D-1

    Parameters
    ----------
    C_geo : float
        Geometric capacity in [0, 1].
    D : int
        Lattice dimension (must be positive).

    Returns
    -------
    weights : ndarray of shape (D,)
        w_d = clamp(C_geo * D - (d-1), 0, 1) for d = 1, ..., D.

    Examples (D=3)
    --------------
    C_geo=1/3 → [1.0, 0.0, 0.0]   (1 active dim)
    C_geo=0.50 → [1.0, 0.5, 0.0]  (1.5 nominal)
    C_geo=2/3 → [1.0, 1.0, 0.0]   (2 active dims)
    C_geo=1.00 → [1.0, 1.0, 1.0]  (3 active dims)
    """
    if D <= 0:
        raise ValueError("D must be positive")
    if not np.isfinite(C_geo):
        raise ValueError("C_geo must be finite")
    d_nom = float(C_geo) * D
    return np.clip(d_nom - np.arange(D, dtype=np.float64), 0.0, 1.0)
