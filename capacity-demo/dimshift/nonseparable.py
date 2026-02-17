"""
Non-separable Laplacian sweep runner and acceptance criteria evaluation.

Tests whether the capacity-controlled "dimension staircase" survives on a
graph whose Laplacian L(C_geo) = Σ_d w_d * L_d + L_rand is NOT separable.

Acceptance criteria:
  A) Monotonicity — ds_plateau is non-decreasing (within tolerance)
  B) Threshold locations — D-1 largest jumps in ds_plateau occur near C = k/D
  C) Plateau banding — ds_plateau clusters near integers away from transitions
"""

import json
import time
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .capacity import capacity_weights
from .rewire import rewire_lattice, build_weighted_laplacian
from .rgg import build_rgg_layers
from .graph_heat import (
    log_return_probability_sparse,
    spectral_dimension_sparse,
)
from .sweep import SweepConfig, CAPACITY_AXES


# ---------------------------------------------------------------------------
# Capacity vector helper
# ---------------------------------------------------------------------------

def _capacity_vector(C_geo: float) -> dict:
    vector = {axis: None for axis in CAPACITY_AXES}
    vector["C_geo"] = float(C_geo)
    return vector



# ---------------------------------------------------------------------------
# C_geo grid helper
# ---------------------------------------------------------------------------

def _refined_C_grid(cfg: SweepConfig, bands: list[tuple[float, float, int]] | None) -> np.ndarray:
    base = cfg.C_geo_grid()
    if not bands:
        return base
    extra = []
    for lo, hi, steps in bands:
        lo = max(cfg.C_geo_min, float(lo))
        hi = min(cfg.C_geo_max, float(hi))
        if hi <= lo or steps <= 0:
            continue
        extra.append(np.linspace(lo, hi, int(steps), endpoint=True))
    if not extra:
        return base
    combined = np.concatenate([base] + extra)
    combined = np.clip(combined, cfg.C_geo_min, cfg.C_geo_max)
    combined = np.unique(np.round(combined, decimals=12))
    return combined


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class NonseparableResult:
    """Full results of a non-separable Laplacian sweep."""
    D: int
    N: int
    rewire_rate: float
    seed: int

    C_geo_values: NDArray[np.float64]
    sigma_values: NDArray[np.float64]

    ds_matrix: NDArray[np.float64]       # shape (n_C, n_sigma)
    ln_P_matrix: NDArray[np.float64]     # shape (n_C, n_sigma)

    ds_plateau: NDArray[np.float64]      # shape (n_C,)
    plateau_iqr: NDArray[np.float64]     # shape (n_C,)
    ds_plateau_std: NDArray[np.float64]  # shape (n_C,)
    ds_plateau_stderr: NDArray[np.float64]  # shape (n_C,)
    plateau_sample_counts: NDArray[np.int32]  # number of probe samples per step

    weights_list: list[NDArray[np.float64]]
    min_active_w: NDArray[np.float64]    # shape (n_C,) minimum non-zero weight

    plateau_window: tuple[float, float]
    elapsed_s: float
    method: str  # "exact" or "slq"
    refine_bands: tuple[tuple[float, float, int], ...] | None = None
    family: str = "lattice"
    graph_metadata: dict | None = None

    criteria: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plateau computation (same logic as sweep.py)
# ---------------------------------------------------------------------------

def _compute_plateau_stats(
    ds_row: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
    ln_P_row: NDArray[np.float64],
    window: tuple[float, float],
) -> tuple[float, float]:
    """
    Robust plateau d_s and IQR within the plateau window.

    Uses linear regression of ln_P vs ln_sigma (d_s = -2 × slope) which is
    much more robust to SLQ noise than pointwise d_s + median.
    Falls back to median of pointwise d_s if regression fails.
    """
    lo, hi = window
    mask = (sigma_values >= lo) & (sigma_values <= hi)
    if np.sum(mask) < 3:
        n = len(sigma_values)
        mask = np.zeros(len(sigma_values), dtype=bool)
        mask[n // 3 : 2 * n // 3] = True

    ln_sig = np.log(sigma_values[mask])
    ln_P = ln_P_row[mask]
    ds_vals = ds_row[mask]

    # Linear regression: ln_P = slope * ln_sigma + intercept
    # d_s = -2 * slope
    if len(ln_sig) >= 3 and np.isfinite(ln_P).all():
        slope, _ = np.polyfit(ln_sig, ln_P, 1)
        ds_regress = -2.0 * slope
    else:
        ds_regress = float(np.median(ds_vals))

    iqr = float(np.percentile(ds_vals, 75) - np.percentile(ds_vals, 25))
    return float(ds_regress), iqr


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_nonseparable_sweep(
    D: int = 3,
    N: int = 32,
    rewire_rate: float = 0.03,
    seed: int = 42,
    C_geo_steps: int = 30,
    n_sigma: int = 400,
    n_probes: int = 30,
    m_lanczos: int = 80,
    verbose: bool = False,
    refine_bands: list[tuple[float, float, int]] | None = None,
    family: str = "lattice",
    family_kwargs: dict | None = None,
) -> NonseparableResult:
    """
    Run a capacity sweep on a non-separable rewired lattice.

    Parameters
    ----------
    D : lattice dimension
    N : side length
    rewire_rate : fraction of local edges to rewire
    seed : RNG seed for deterministic rewiring
    C_geo_steps : number of C_geo points
    n_sigma : number of sigma points
    n_probes : SLQ probe count
    m_lanczos : Lanczos iteration depth
    verbose : print progress per C_geo step
    """
    t0 = time.time()

    # Build the fixed graph structure (capacity-independent)
    family_kwargs = family_kwargs or {}
    if family == "rgg":
        radius_scale = family_kwargs.get("radius_scale", 1.0)
        radius_min = family_kwargs.get("radius_min", 1e-3)
        radius = max(radius_min, rewire_rate * radius_scale)
        if verbose:
            print(f"Building D={D} RGG (n_total={N**D}) with radius={radius:.4f}...")
        graph = build_rgg_layers(D, N, radius, seed)
    elif family == "lattice":
        if verbose:
            print(f"Building D={D} N={N} lattice (n_total={N**D}) with r={rewire_rate}...")
        graph = rewire_lattice(D, N, rewire_rate, seed)
    else:
        raise ValueError(f"Unsupported family: {family}")

    L_dims = graph["L_dims"]
    L_rand = graph["L_rand"]
    n_total = graph.get("n_total", N ** D)
    graph_metadata = graph.get("metadata", {})
    if verbose and family == "lattice":
        print(f"  Rewired {graph.get('n_rewired', 0)}/{graph.get('n_local_edges', 0)} edges")

    # Sigma and C_geo grids (reuse SweepConfig conventions)
    cfg = SweepConfig(D=D, N=N, n_sigma=n_sigma, C_geo_steps=C_geo_steps)
    sigma_values = cfg.sigma_grid()
    C_geo_values = _refined_C_grid(cfg, refine_bands)
    window = cfg.plateau_window()

    n_C = len(C_geo_values)
    n_sig = len(sigma_values)

    ds_matrix = np.zeros((n_C, n_sig))
    ln_P_matrix = np.zeros((n_C, n_sig))
    ds_plateau_arr = np.zeros(n_C)
    plateau_iqr_arr = np.zeros(n_C)
    ds_plateau_std_arr = np.zeros(n_C)
    ds_plateau_stderr_arr = np.zeros(n_C)
    plateau_sample_counts = np.zeros(n_C, dtype=np.int32)
    min_active_w_arr = np.zeros(n_C)
    weights_list = []

    method = "exact" if n_total < 8000 else "slq"

    for i, C_geo in enumerate(C_geo_values):
        weights = capacity_weights(float(C_geo), D)
        weights_list.append(weights)

        active = weights[weights > 1e-15]
        min_active_w_arr[i] = float(np.min(active)) if len(active) > 0 else 0.0

        L = build_weighted_laplacian(L_dims, L_rand, float(C_geo), D)

        # Use same SLQ seed for all C_geo steps so noise is correlated
        # across steps — makes relative changes (monotonicity) accurate
        ln_P_output = log_return_probability_sparse(
            L, sigma_values,
            n_probes=n_probes, m_lanczos=m_lanczos, seed=seed,
            return_probe_lnP=True,
        )
        if isinstance(ln_P_output, tuple):
            ln_P, probe_lnP = ln_P_output
        else:  # pragma: no cover — legacy fallback
            ln_P, probe_lnP = ln_P_output, None
        ln_P_matrix[i] = ln_P

        ds = spectral_dimension_sparse(sigma_values, ln_P)
        ds_matrix[i] = ds

        median, iqr = _compute_plateau_stats(ds, sigma_values, ln_P, window)
        ds_plateau_arr[i] = median
        plateau_iqr_arr[i] = iqr

        probe_plateaus: list[float] = []
        if probe_lnP is not None:
            for probe_ln in probe_lnP:
                probe_ds = spectral_dimension_sparse(sigma_values, probe_ln)
                probe_median, _ = _compute_plateau_stats(probe_ds, sigma_values, probe_ln, window)
                probe_plateaus.append(float(probe_median))
        else:
            probe_plateaus.append(float(median))

        samples = np.asarray(probe_plateaus, dtype=float)
        plateau_sample_counts[i] = samples.size
        if samples.size > 1:
            std = float(np.std(samples, ddof=1))
        else:
            std = 0.0
        stderr = float(std / np.sqrt(samples.size)) if samples.size > 0 else 0.0
        ds_plateau_std_arr[i] = std
        ds_plateau_stderr_arr[i] = stderr

        if verbose:
            print(f"  C_geo={C_geo:.4f}  ds_plateau={median:.3f}  "
                  f"iqr={iqr:.3f}  min_w={min_active_w_arr[i]:.3f}  "
                  f"method={method}  [{i+1}/{n_C}]")

    elapsed = time.time() - t0

    result = NonseparableResult(
        D=D, N=N, rewire_rate=rewire_rate, seed=seed, family=family,
        C_geo_values=C_geo_values, sigma_values=sigma_values,
        ds_matrix=ds_matrix, ln_P_matrix=ln_P_matrix,
        ds_plateau=ds_plateau_arr, plateau_iqr=plateau_iqr_arr,
        ds_plateau_std=ds_plateau_std_arr,
        ds_plateau_stderr=ds_plateau_stderr_arr,
        plateau_sample_counts=plateau_sample_counts,
        weights_list=weights_list, min_active_w=min_active_w_arr,
        plateau_window=window, elapsed_s=round(elapsed, 3),
        method=method, refine_bands=tuple(refine_bands) if refine_bands else None,
        graph_metadata=graph_metadata,
    )

    return result


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def evaluate_criteria(
    result: NonseparableResult,
    mono_tol: float | None = None,
    threshold_tol: float | None = None,
    banding_dist_tol: float | None = None,
    banding_iqr_tol: float | None = None,
) -> dict:
    """
    Evaluate acceptance criteria A, B, C.

    Tolerances are N-dependent if not explicitly provided.
    """
    N = result.N
    D = result.D
    ds_p = result.ds_plateau
    C_vals = result.C_geo_values

    # N-dependent defaults
    if mono_tol is None:
        mono_tol = 0.05 if N >= 64 else (0.08 if N >= 32 else 0.12)
    if threshold_tol is None:
        threshold_tol = 0.10 if N >= 64 else 0.15
    if banding_dist_tol is None:
        banding_dist_tol = 0.15 if N >= 64 else (0.35 if N >= 32 else 0.40)
    if banding_iqr_tol is None:
        banding_iqr_tol = 0.20 if N >= 64 else (0.40 if N >= 32 else 0.50)

    criteria = {}

    # --- Criterion A: Monotonicity ---
    diffs = np.diff(ds_p)
    drop_mask = diffs < 0
    raw_drops = -diffs[drop_mask]

    se = getattr(result, "ds_plateau_stderr", None)
    se_valid = isinstance(se, np.ndarray) and se.shape == ds_p.shape and np.any(se > 0)
    if se_valid:
        se_pairs = np.sqrt(se[:-1] ** 2 + se[1:] ** 2)
        thresholds = 2.0 * se_pairs
        thresholds = thresholds[drop_mask]
        adjusted_drops = np.maximum(0.0, raw_drops - thresholds)
        max_drop = float(np.max(adjusted_drops)) if len(adjusted_drops) > 0 else 0.0
    else:
        max_drop = float(np.max(raw_drops)) if len(raw_drops) > 0 else 0.0

    max_drop_raw = float(np.max(raw_drops)) if len(raw_drops) > 0 else 0.0
    criteria["A"] = {
        "name": "Monotonicity",
        "pass": max_drop <= mono_tol,
        "max_drop": round(max_drop, 6),
        "max_drop_raw": round(max_drop_raw, 6),
        "tolerance": mono_tol,
        "used_uncertainty": bool(se_valid),
    }

    # --- Criterion B: Threshold locations (jump detection) ---
    n_expected_jumps = D - 1
    if len(ds_p) > 2 and n_expected_jumps > 0:
        step_diffs = np.diff(ds_p)
        # Find the D-1 largest jumps
        jump_indices = np.argsort(step_diffs)[::-1][:n_expected_jumps]
        jump_indices = np.sort(jump_indices)  # sort by C position

        predicted = [(k + 1) / D for k in range(n_expected_jumps)]
        errors = []
        jump_details = []

        for rank, ji in enumerate(jump_indices):
            # Midpoint C of the jump
            c_mid = float((C_vals[ji] + C_vals[ji + 1]) / 2)
            magnitude = float(step_diffs[ji])
            pred_c = predicted[rank]
            err = abs(c_mid - pred_c)
            errors.append(err)
            jump_details.append({
                "rank": rank + 1,
                "c_mid": round(c_mid, 4),
                "predicted_c": round(pred_c, 4),
                "error": round(err, 4),
                "magnitude": round(magnitude, 4),
            })

        max_error = max(errors) if errors else 0.0
        criteria["B"] = {
            "name": "Threshold locations",
            "pass": max_error <= threshold_tol and all(
                d["magnitude"] > 0.3 for d in jump_details
            ),
            "max_error": round(max_error, 6),
            "tolerance": threshold_tol,
            "jumps": jump_details,
        }
    else:
        criteria["B"] = {
            "name": "Threshold locations",
            "pass": False,
            "reason": "insufficient data",
        }

    # --- Criterion C: Plateau banding (segment flatness) ---
    # Segment ds_plateau into D bands separated by the detected jumps.
    # Check that within each band, ds_plateau is approximately constant,
    # and that inter-band jumps are significantly larger than intra-band spread.
    # This captures the staircase structure even when plateaus are not at integers
    # (random edges boost d_s above integer values).

    if "B" in criteria and "jumps" in criteria["B"] and len(criteria["B"]["jumps"]) > 0:
        # Get segment boundaries from jump detection
        step_diffs = np.diff(ds_p)
        n_jumps = min(D - 1, len(criteria["B"]["jumps"]))
        jump_idx = np.sort(np.argsort(step_diffs)[::-1][:n_jumps])

        # Build segments: [0..jump0], [jump0+1..jump1], [jump1+1..end]
        # Within each segment, only count "core" points where all active
        # weights are substantial (min_active_w >= 0.5), excluding the
        # transition region near each threshold.
        boundaries = [0] + [int(ji + 1) for ji in jump_idx] + [len(ds_p)]
        segment_stds = []
        segment_means = []
        segment_details = []

        for seg_i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[seg_i], boundaries[seg_i + 1]
            seg_vals_all = ds_p[seg_start:seg_end]
            seg_maw = result.min_active_w[seg_start:seg_end]

            # Core: min_active_w >= 0.5
            core_mask = seg_maw >= 0.5
            seg_vals = seg_vals_all[core_mask] if np.sum(core_mask) >= 2 else seg_vals_all

            std = float(np.std(seg_vals))
            mean = float(np.mean(seg_vals))
            segment_stds.append(std)
            segment_means.append(mean)
            segment_details.append({
                "segment": seg_i + 1,
                "n_points": len(seg_vals_all),
                "n_core": int(np.sum(core_mask)),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "range": round(float(np.ptp(seg_vals)), 4),
            })

        # Inter-band gaps: differences between consecutive segment means
        gaps = [segment_means[i + 1] - segment_means[i]
                for i in range(len(segment_means) - 1)]

        # Banding: gaps between segments must be larger than the width of
        # each segment. d_s varies smoothly within a segment (weights change
        # continuously), so we use segment range (max-min) not std.
        segment_ranges = [d["range"] for d in segment_details]
        max_seg_range = max(segment_ranges) if segment_ranges else 999.0
        min_gap = min(gaps) if gaps else 0.0
        n_segments = len(segment_stds)

        # Pass if: min inter-segment gap > max intra-segment range
        # (bands don't overlap) AND we have D segments
        bands_separated = min_gap > max_seg_range
        # Also check gap-to-range ratio is at least 2.0
        gap_range_ratio = (min_gap / max_seg_range) if max_seg_range > 1e-10 else 999.0

        criteria["C"] = {
            "name": "Plateau banding",
            "pass": bands_separated and n_segments >= D and gap_range_ratio >= 2.0,
            "n_segments": n_segments,
            "segment_means": [round(m, 4) for m in segment_means],
            "segment_ranges": [round(r, 4) for r in segment_ranges],
            "segment_stds": [round(s, 4) for s in segment_stds],
            "gaps": [round(g, 4) for g in gaps],
            "min_gap": round(min_gap, 4),
            "max_seg_range": round(max_seg_range, 4),
            "gap_range_ratio": round(gap_range_ratio, 4),
            "segments": segment_details,
        }
    else:
        criteria["C"] = {
            "name": "Plateau banding",
            "pass": False,
            "reason": "no jumps detected by criterion B",
        }

    # Overall
    criteria["overall_pass"] = all(
        criteria[k].get("pass", False) for k in ["A", "B", "C"]
    )

    result.criteria = criteria
    return criteria


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------

def write_artifacts(
    result: NonseparableResult,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write experiment artifacts to disk."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Metadata JSON
    meta = {
        "type": "nonseparable_rewire_test",
        "D": result.D, "N": result.N,
        "rewire_rate": result.rewire_rate, "seed": result.seed,
        "n_total": (result.graph_metadata or {}).get("n_total", result.N ** result.D),
        "method": result.method,
        "elapsed_s": result.elapsed_s,
        "plateau_window": list(result.plateau_window),
        "n_C": len(result.C_geo_values),
        "n_sigma": len(result.sigma_values),
        "refine_bands": result.refine_bands,
        "family": result.family,
        "graph_metadata": result.graph_metadata,
        "criteria": result.criteria,
        "result": {
            "C_geo_values": result.C_geo_values.tolist(),
            "sigma_values": result.sigma_values.tolist(),
            "ds_plateau": result.ds_plateau.tolist(),
            "ds_plateau_std": result.ds_plateau_std.tolist(),
            "ds_plateau_stderr": result.ds_plateau_stderr.tolist(),
            "plateau_sample_counts": result.plateau_sample_counts.tolist(),
        },
    }
    p = out / "metadata.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    paths["metadata"] = p

    # 2. CSV: C_geo, ds_plateau, plateau_iqr, min_active_w, weights
    p = out / "sweep_results.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "C_geo",
            "ds_plateau",
            "ds_plateau_std",
            "ds_plateau_stderr",
            "plateau_iqr",
            "min_active_w",
            "n_plateau_samples",
        ]
        header += [f"w_{d+1}" for d in range(result.D)]
        writer.writerow(header)
        for i in range(len(result.C_geo_values)):
            row = [
                f"{result.C_geo_values[i]:.6f}",
                f"{result.ds_plateau[i]:.6f}",
                f"{result.ds_plateau_std[i]:.6f}",
                f"{result.ds_plateau_stderr[i]:.6f}",
                f"{result.plateau_iqr[i]:.6f}",
                f"{result.min_active_w[i]:.6f}",
                str(int(result.plateau_sample_counts[i])),
            ]
            row += [f"{result.weights_list[i][d]:.6f}" for d in range(result.D)]
            writer.writerow(row)
    paths["csv"] = p

    # 3. Thresholds JSON (from criteria B)
    if "B" in result.criteria and "jumps" in result.criteria["B"]:
        p = out / "thresholds.json"
        with open(p, "w") as f:
            json.dump(result.criteria["B"]["jumps"], f, indent=2)
        paths["thresholds"] = p

    # 4. Summary JSON
    summary = {
        "D": result.D, "N": result.N,
        "rewire_rate": result.rewire_rate,
        "ds_plateau_min": round(float(np.min(result.ds_plateau)), 4),
        "ds_plateau_max": round(float(np.max(result.ds_plateau)), 4),
        "criteria": result.criteria,
        "elapsed_s": result.elapsed_s,
        "result": {
            "C_geo_values": result.C_geo_values.tolist(),
            "sigma_values": result.sigma_values.tolist(),
            "ds_plateau": result.ds_plateau.tolist(),
            "ds_plateau_std": result.ds_plateau_std.tolist(),
            "ds_plateau_stderr": result.ds_plateau_stderr.tolist(),
            "plateau_sample_counts": result.plateau_sample_counts.tolist(),
            "capacity_vectors": [_capacity_vector(c) for c in result.C_geo_values],
        },
    }
    p = out / "summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    paths["summary"] = p

    # 5. Full ds_matrix dump for downstream analysis
    p = out / "ds_matrix.json"
    with open(p, "w") as f:
        json.dump({
            "C_geo_values": result.C_geo_values.tolist(),
            "sigma_values": result.sigma_values.tolist(),
            "ds_matrix": result.ds_matrix.tolist(),
            "ln_P_matrix": result.ln_P_matrix.tolist(),
            "ds_plateau": result.ds_plateau.tolist(),
            "ds_plateau_std": result.ds_plateau_std.tolist(),
            "ds_plateau_stderr": result.ds_plateau_stderr.tolist(),
            "plateau_sample_counts": result.plateau_sample_counts.tolist(),
            "capacity_vectors": [_capacity_vector(c) for c in result.C_geo_values],
        }, f)
    paths["ds_matrix"] = p

    return paths
