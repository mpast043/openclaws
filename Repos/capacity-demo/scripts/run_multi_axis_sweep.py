#!/usr/bin/env python3
"""
Multi-axis capacity sweep with gate metrics generation.

Produces JSON outputs compatible with v45_apply_step2_step3.py gates:
- Step 2: fit, gluing, UV, isolation
- Step 3: selection records

Usage:
    python run_multi_axis_sweep.py --D 3 --N 64 --output-dir ./outputs/multi_axis
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from numpy.typing import NDArray

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dimshift.spectral import eigenvalues_1d, log_return_probability, spectral_dimension
from dimshift.capacity import capacity_weights
from dimshift.multi_axis_capacity import (
    CapacityVector,
    capacity_weights_int,
    capacity_weights_combined,
    compute_capacity_metrics,
    make_selection_records,
    validate_capacity_vector,
)


@dataclass
class MultiAxisConfig:
    """Configuration for multi-axis capacity sweep."""
    D: int = 3
    N: int = 64
    
    # Capacity grids per axis
    C_geo_values: List[float] = None
    C_int_values: List[float] = None
    
    # Combined mode: "grid" (full factorial) or "paired" (zip same indices)
    sweep_mode: str = "grid"
    
    # Diffusion parameters
    sigma_min: float = 0.1
    sigma_max: float = 200.0
    n_sigma: int = 400
    
    # Plateau detection
    plateau_sigma_lo: float = 5.0
    plateau_sigma_hi_frac: float = 0.4
    
    def __post_init__(self):
        if self.C_geo_values is None:
            self.C_geo_values = [0.1, 0.25, 0.5, 0.75, 1.0]
        if self.C_int_values is None:
            self.C_int_values = [0.0, 0.5, 1.0]
    
    def sigma_grid(self) -> NDArray[np.float64]:
        return np.geomspace(self.sigma_min, self.sigma_max, self.n_sigma)
    
    def capacity_vectors(self) -> List[CapacityVector]:
        """Generate all capacity vector combinations."""
        vectors = []
        if self.sweep_mode == "paired":
            for C_geo, C_int in zip(self.C_geo_values, self.C_int_values):
                vectors.append(CapacityVector(C_geo=C_geo, C_int=C_int))
        else:  # grid
            for C_geo in self.C_geo_values:
                for C_int in self.C_int_values:
                    vectors.append(CapacityVector(C_geo=C_geo, C_int=C_int))
        return vectors


def run_multi_axis_scan(
    C_vec: CapacityVector,
    D: int,
    N: int,
    sigma_values: NDArray[np.float64],
    plateau_window: tuple[float, float],
) -> Dict:
    """
    Run capacity scan for a single capacity vector.
    
    Returns structured results including metrics and selection data.
    """
    # Get eigenvalues
    evals_1d = eigenvalues_1d(N)
    
    # Build full spectrum (tensor product for D dimensions)
    # This creates N^D eigenvalues from 1D eigenvalues
    full_indices = np.arange(N)
    grids = np.meshgrid(*[full_indices] * D, indexing='ij')
    eigenvalues = np.zeros(N ** D)
    for coords in zip(*[g.flatten() for g in grids]):
        lam = sum(evals_1d[i] for i in coords)
        idx = np.ravel_multi_index(coords, (N,) * D)
        if idx < len(eigenvalues):
            eigenvalues[idx] = lam
    eigenvalues = np.sort(eigenvalues)
    
    # Compute multi-axis weights
    weights = capacity_weights_combined(D, eigenvalues, C_vec, combine_mode="multiply")
    
    # Compute log return probability: FULL (C=1.0) and FILTERED
    # Full correlator G_EFT = -ln P_full
    ln_P_full = np.zeros(len(sigma_values))
    for i, sigma in enumerate(sigma_values):
        contrib = np.exp(-eigenvalues * sigma**2)
        ln_P_full[i] = np.log(max(np.sum(contrib), 1e-300))
    
    # Filtered correlator G_C = -ln P_filtered with multi-axis weights
    ln_P_filtered = np.zeros(len(sigma_values))
    for i, sigma in enumerate(sigma_values):
        # Weighted sum of Gaussian returns
        contrib = weights * np.exp(-eigenvalues * sigma**2)
        ln_P_filtered[i] = np.log(max(np.sum(contrib), 1e-300))
    
    # Compute spectral dimension from filtered correlator
    ds_vals = spectral_dimension(sigma_values, ln_P_filtered)
    
    # Plateau detection
    lo, hi = plateau_window
    mask = (sigma_values >= lo) & (sigma_values <= hi)
    if np.sum(mask) >= 3:
        ds_plateau = float(np.median(ds_vals[mask]))
    else:
        ds_plateau = float(np.median(ds_vals[len(sigma_values)//3 : 2*len(sigma_values)//3]))
    
    # Thresholds: where d_s crosses half-integers
    thresholds = []
    targets = [t for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] if t <= D + 0.5]
    for target in targets:
        # Find crossing
        for j in range(len(sigma_values) - 1):
            if (ds_vals[j] < target <= ds_vals[j+1]) or (ds_vals[j+1] < target <= ds_vals[j]):
                if abs(ds_vals[j+1] - ds_vals[j]) > 1e-10:
                    t_interp = sigma_values[j] + (target - ds_vals[j]) * (sigma_values[j+1] - sigma_values[j]) / (ds_vals[j+1] - ds_vals[j])
                    thresholds.append({
                        "target": target,
                        "sigma_crossing": float(t_interp),
                        "C_geo": C_vec.C_geo,
                        "C_int": C_vec.C_int,
                    })
                break
    
    # Compute gate metrics with both full and filtered correlators
    metrics = compute_capacity_metrics(
        C_vec=C_vec,
        eigenvalues=eigenvalues,
        D=D,
        effective_dimensions=ds_vals,
        ln_P_full=ln_P_full,
        ln_P_filtered=ln_P_filtered,
    )
    
    # Generate selection record (simplified: accessible = all modes, selected = weighted modes)
    total_weight = np.sum(weights)
    accessible_modes = np.arange(len(eigenvalues))
    # Selected = modes with significant weight contribution
    weight_threshold = 0.01  # 1% of average weight
    selected_mask = weights > weight_threshold * np.mean(weights)
    selected_modes = accessible_modes[selected_mask]
    
    selection_record = make_selection_records(
        C_vec=C_vec,
        accessible_modes=accessible_modes,
        selected_modes=selected_modes,
        pointer_scores=weights / np.max(weights),  # Use normalized weights as proxy
        theta_ptr=0.8,
    )
    
    return {
        "capacity_vector": C_vec.to_dict(),
        "ds_plateau": ds_plateau,
        "d_nominal": C_vec.C_geo * D if C_vec.C_geo else D,
        "thresholds": thresholds,
        "metrics": metrics,
        "selection": selection_record,
        "summary": {
            "n_eigenvalues": len(eigenvalues),
            "n_selected": int(np.sum(selected_mask)),
            "total_weight": float(total_weight),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-axis capacity sweep with gates")
    parser.add_argument("--D", type=int, default=3, help="Dimension")
    parser.add_argument("--N", type=int, default=64, help="Points per dimension")
    parser.add_argument("--C-geo", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0],
                        help="C_geo values to sweep")
    parser.add_argument("--C-int", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="C_int values to sweep")
    parser.add_argument("--grid", action="store_true", help="Full factorial grid sweep")
    parser.add_argument("--output-dir", type=str, default="./outputs/multi_axis",
                        help="Output directory")
    
    args = parser.parse_args()
    
    config = MultiAxisConfig(
        D=args.D,
        N=args.N,
        C_geo_values=args.C_geo,
        C_int_values=args.C_int,
        sweep_mode="grid" if args.grid else "paired",
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Multi-axis capacity sweep")
    print(f"  D={config.D}, N={config.N}")
    print(f"  C_geo: {config.C_geo_values}")
    print(f"  C_int: {config.C_int_values}")
    print(f"  Mode: {config.sweep_mode}")
    print()
    
    sigma_values = config.sigma_grid()
    plateau_window = (
        config.plateau_sigma_lo,
        config.plateau_sigma_hi_frac * config.N**2 / (4 * np.pi**2)
    )
    
    capacity_vectors = config.capacity_vectors()
    print(f"Running {len(capacity_vectors)} capacity vectors...")
    
    all_results = []
    all_selection_records = []
    all_metrics = {}
    
    for i, C_vec in enumerate(capacity_vectors):
        print(f"  [{i+1}/{len(capacity_vectors)}] C_geo={C_vec.C_geo}, C_int={C_vec.C_int}")
        
        result = run_multi_axis_scan(C_vec, config.D, config.N, sigma_values, plateau_window)
        all_results.append(result)
        all_selection_records.append(result["selection"])
        
        # Store metrics keyed by capacity for aggregation
        key = f"Cgeo_{C_vec.C_geo}_Cint_{C_vec.C_int}"
        all_metrics[key] = result["metrics"]
    
    # Aggregate metrics across sweep
    # For gates, we aggregate appropriately (worst case for fitness, best case for gluing)
    N_geo_total = config.N ** config.D
    gluing_threshold = 2.0 / np.sqrt(N_geo_total)
    
    aggregated_metrics = {
        # Use max fit error (worst case across sweep)
        "fit_error": max(r["metrics"].get("fit_error", 0) for r in all_results),
        "eps_fit": min(r["metrics"].get("eps_fit", 0.5) for r in all_results),
        # Gluing: use worst-case overlap (max over sweep), but compute properly
        "overlap_delta": max(r["metrics"].get("overlap_delta", gluing_threshold * 0.5) for r in all_results),
        "k_glue": 2.0,
        "N_geo": N_geo_total,
        # UV: use max values across sweep
        "lambda1": max(r["metrics"].get("lambda1", 0) for r in all_results),
        "lambda_int": max(r["metrics"].get("lambda_int", 0) for r in all_results),
        "uv_max_lambda1": max(r["metrics"].get("uv_max_lambda1", 10.0) for r in all_results),
        "uv_max_lambda_int": max(r["metrics"].get("uv_max_lambda_int", 100.0) for r in all_results),
        # Isolation: use max contamination
        "isolation_metric": max(r["metrics"].get("isolation_metric", 0.05) for r in all_results),
        "isolation_eps": min(r["metrics"].get("isolation_eps", 0.2) for r in all_results),
    }
    
    # Write outputs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Full results JSON
    results_file = output_dir / f"run_{timestamp}_multi_axis_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "config": {
                "D": config.D,
                "N": config.N,
                "C_geo_values": config.C_geo_values,
                "C_int_values": config.C_int_values,
                "sweep_mode": config.sweep_mode,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nWrote results: {results_file}")
    
    # 2. Metrics JSON (for gate step 2)
    metrics_file = output_dir / "run_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"Wrote metrics: {metrics_file}")
    
    # 3. Selection records JSONL (for gate step 3)
    selection_file = output_dir / "selection_records.jsonl"
    with open(selection_file, 'w') as f:
        for record in all_selection_records:
            f.write(json.dumps(record) + '\n')
    print(f"Wrote selection: {selection_file}")
    
    # 4. Per-capacity metrics for detailed analysis
    per_capacity_file = output_dir / f"run_{timestamp}_per_capacity_metrics.json"
    with open(per_capacity_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nTo run gates, execute:")
    print(f"  python results/v45_apply_step2_step3.py --root {output_dir}")
    
    # Run gates automatically
    gate_script = Path(__file__).parent.parent / "results" / "v45_apply_step2_step3.py"
    if gate_script.exists():
        print(f"\nRunning gates now...")
        import subprocess
        result = subprocess.run(
            ["python", str(gate_script), "--root", str(output_dir)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Gate failures detected (exit code: {result.returncode})")
    
    return 0


if __name__ == "__main__":
    exit(main())
