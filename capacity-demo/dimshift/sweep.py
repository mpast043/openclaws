"""
Capacity sweep runner — produces deterministic, self-reporting experiment results.

Holds everything fixed except C_geo.  Writes CSV, JSON metadata, and returns
structured results for plotting and analysis.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .spectral import eigenvalues_1d, batch_log_return_probability

CAPACITY_AXES = ("C_geo", "C_int", "C_gauge", "C_ptr", "C_obs")


def _make_capacity_vector(C_geo: float, overrides: Optional[Dict[str, float]] = None) -> Dict[str, Optional[float]]:
    """Return a dict covering all Framework capacity axes."""
    vector: Dict[str, Optional[float]] = {axis: None for axis in CAPACITY_AXES}
    vector["C_geo"] = float(C_geo)
    if overrides:
        for key, value in overrides.items():
            if key in vector:
                vector[key] = value
    return vector


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepConfig:
    """All fixed parameters for a capacity sweep experiment."""

    # Lattice
    D: int = 3
    N: int = 64

    # Capacity grid (only C_geo varies)
    C_geo_min: float = 0.05
    C_geo_max: float = 1.0
    C_geo_steps: int = 30

    # Diffusion time grid
    sigma_min: float = 0.1
    sigma_max: float = 200.0
    n_sigma: int = 400

    # Plateau detection window (fixed, not capacity-dependent)
    plateau_sigma_lo: float = 5.0     # lower bound of plateau window
    plateau_sigma_hi_frac: float = 0.4  # fraction of N^2/(4π²) as upper bound

    # Optional placeholder values for non-geometric capacity axes
    capacity_overrides: Optional[Dict[str, float]] = None

    def sigma_grid(self) -> NDArray[np.float64]:
        return np.geomspace(self.sigma_min, self.sigma_max, self.n_sigma)

    def C_geo_grid(self) -> NDArray[np.float64]:
        return np.linspace(self.C_geo_min, self.C_geo_max, self.C_geo_steps)

    def plateau_window(self) -> tuple[float, float]:
        """
        Fixed plateau detection window [sigma_lo, sigma_hi].

        sigma_hi = min(plateau_sigma_hi_frac * N^2/(4pi^2), sigma_max * 0.6)
        The sigma_max * 0.6 cap prevents the window from extending into the
        finite-size saturation tail.  A floor of sigma_lo + 1.0 ensures the
        window always contains at least a minimal range.
        """
        sigma_hi = self.plateau_sigma_hi_frac * self.N**2 / (4 * np.pi**2)
        sigma_hi = min(sigma_hi, self.sigma_max * 0.6)
        return (self.plateau_sigma_lo, max(sigma_hi, self.plateau_sigma_lo + 1.0))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SweepResult:
    """Complete results of a capacity sweep."""

    config: SweepConfig
    C_geo_values: NDArray[np.float64]
    sigma_values: NDArray[np.float64]

    # Per-(C_geo, sigma) data
    ds_matrix: NDArray[np.float64]       # shape (n_C, n_sigma)
    ln_P_matrix: NDArray[np.float64]     # shape (n_C, n_sigma)

    # Per-C_geo summary
    weights_list: list[NDArray[np.float64]]
    ds_plateau: NDArray[np.float64]      # shape (n_C,)
    d_eff_nominal: NDArray[np.float64]   # shape (n_C,) = sum of weights
    capacity_vectors: list[Dict[str, Optional[float]]]

    # Thresholds
    thresholds: list[dict]

    # Metadata
    run_id: str = ""
    timestamp: str = ""
    elapsed_s: float = 0.0

    def plateau_at(self, C_geo: float) -> float:
        """Interpolate ds_plateau at a given C_geo value."""
        return float(np.interp(C_geo, self.C_geo_values, self.ds_plateau))


# ---------------------------------------------------------------------------
# Plateau detection (fixed window, not capacity-dependent)
# ---------------------------------------------------------------------------

def _compute_plateau(
    ds_row: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
    window: tuple[float, float],
) -> float:
    """
    Median d_s within the fixed plateau window.

    If the window contains fewer than 3 sigma points (e.g. very small N),
    falls back to the middle third of the sigma grid.  Both paths are
    deterministic and independent of C_geo.
    """
    lo, hi = window
    mask = (sigma_values >= lo) & (sigma_values <= hi)
    if np.sum(mask) >= 3:
        return float(np.median(ds_row[mask]))
    # Fallback: middle third of sigma grid
    n = len(sigma_values)
    return float(np.median(ds_row[n // 3 : 2 * n // 3]))


# ---------------------------------------------------------------------------
# Threshold detection
# ---------------------------------------------------------------------------

def _detect_thresholds(
    C_geo_values: NDArray[np.float64],
    ds_plateau: NDArray[np.float64],
    D: int,
) -> list[dict]:
    """
    Detect C_geo thresholds where d_s crosses integer/half-integer values.

    Assumes d_s is roughly monotone-increasing with C_geo (staircase shape).
    Reports only the first crossing for each target; if noise causes multiple
    crossings, only the earliest is captured.
    """
    targets = [v for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] if v <= D + 0.5]
    thresholds = []

    for target in targets:
        for i in range(len(C_geo_values) - 1):
            c1, d1 = float(C_geo_values[i]), float(ds_plateau[i])
            c2, d2 = float(C_geo_values[i + 1]), float(ds_plateau[i + 1])
            if (d1 < target <= d2) or (d2 < target <= d1):
                if abs(d2 - d1) > 1e-10:
                    c_thresh = c1 + (target - d1) * (c2 - c1) / (d2 - d1)
                    thresholds.append({
                        "target_dimension": target,
                        "C_geo_threshold": round(c_thresh, 6),
                        "bracket": [round(c1, 6), round(c2, 6)],
                    })
                break

    return thresholds


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_capacity_sweep(config: Optional[SweepConfig] = None) -> SweepResult:
    """
    Run a deterministic capacity sweep experiment.

    Holds fixed: D, N, lattice spacing (a=1), diffusion convention,
    sigma grid, derivative estimator, plateau window.

    Only C_geo varies.

    Parameters
    ----------
    config : SweepConfig, optional
        Experiment configuration. Uses defaults if None.

    Returns
    -------
    SweepResult
        Full results including per-point d_s values, plateau summaries,
        detected thresholds, and metadata.
    """
    if config is None:
        config = SweepConfig()

    t0 = time.time()
    sigma_values = config.sigma_grid()
    C_geo_values = config.C_geo_grid()
    window = config.plateau_window()

    # Precompute 1D eigenvalues once (O(N))
    eigs_1d = eigenvalues_1d(config.N)
    eig_sigma = eigs_1d[:, None] * sigma_values[None, :]

    n_C = len(C_geo_values)
    n_sigma = len(sigma_values)

    # Capacity weights for all C_geo values at once
    weights_matrix = np.clip(
        C_geo_values[:, None] * config.D - np.arange(config.D, dtype=np.float64),
        0.0,
        1.0,
    )
    d_eff_arr = np.sum(weights_matrix, axis=1)
    weights_list = [weights_matrix[i].copy() for i in range(n_C)]
    capacity_vectors = [
        _make_capacity_vector(float(C_geo_values[i]), config.capacity_overrides)
        for i in range(n_C)
    ]

    # Batch log return probability and spectral dimension
    ln_P_matrix = batch_log_return_probability(
        eigs_1d,
        weights_matrix,
        sigma_values,
        eig_sigma=eig_sigma,
    )
    ln_sigma = np.log(sigma_values)
    ds_matrix = -2.0 * np.gradient(ln_P_matrix, ln_sigma, axis=1)

    ds_plateau_arr = np.zeros(n_C)
    for i in range(n_C):
        ds_plateau_arr[i] = _compute_plateau(ds_matrix[i], sigma_values, window)

    thresholds = _detect_thresholds(C_geo_values, ds_plateau_arr, config.D)
    elapsed = time.time() - t0

    # Run ID from config hash
    config_str = json.dumps(config.to_dict(), sort_keys=True)
    run_id = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    return SweepResult(
        config=config,
        C_geo_values=C_geo_values,
        sigma_values=sigma_values,
        ds_matrix=ds_matrix,
        ln_P_matrix=ln_P_matrix,
        weights_list=weights_list,
        ds_plateau=ds_plateau_arr,
        d_eff_nominal=d_eff_arr,
        capacity_vectors=capacity_vectors,
        thresholds=thresholds,
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_s=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def write_artifacts(result: SweepResult, output_dir: str | Path) -> dict[str, Path]:
    """
    Write all experiment artifacts to output_dir/.

    Returns dict mapping artifact name → file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Metadata JSON (all fixed parameters + run info)
    meta = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "elapsed_s": result.elapsed_s,
        "config": result.config.to_dict(),
        "plateau_window": list(result.config.plateau_window()),
        "thresholds": result.thresholds,
        "C_geo_grid": result.C_geo_values.tolist(),
        "capacity_vectors": result.capacity_vectors,
        "n_sigma": len(result.sigma_values),
    }
    p = out / "metadata.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    paths["metadata"] = p

    # 2. CSV: C_geo, d_eff_nominal, ds_plateau, weights
    import csv
    p = out / "sweep_results.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["C_geo", "d_eff_nominal", "ds_plateau"]
        header += [f"w_{d+1}" for d in range(result.config.D)]
        writer.writerow(header)
        for i in range(len(result.C_geo_values)):
            row = [
                f"{result.C_geo_values[i]:.6f}",
                f"{result.d_eff_nominal[i]:.6f}",
                f"{result.ds_plateau[i]:.6f}",
            ]
            row += [f"{result.weights_list[i][d]:.6f}" for d in range(result.config.D)]
            writer.writerow(row)
    paths["csv"] = p

    # 3. Full d_s matrix as JSON (for reproducibility)
    p = out / "ds_matrix.json"
    with open(p, "w") as f:
        json.dump({
            "C_geo_values": result.C_geo_values.tolist(),
            "sigma_values": result.sigma_values.tolist(),
            "ds_matrix": result.ds_matrix.tolist(),
            "ln_P_matrix": result.ln_P_matrix.tolist(),
            "ds_plateau": result.ds_plateau.tolist(),
            "capacity_vectors": result.capacity_vectors,
        }, f)
    paths["ds_matrix"] = p

    # 4. Thresholds JSON
    p = out / "thresholds.json"
    with open(p, "w") as f:
        json.dump(result.thresholds, f, indent=2)
    paths["thresholds"] = p

    return paths
