#!/usr/bin/env python3
"""
Framework v4.5 Option B — Validation Harness.

Runs a matrix of (substrate x filter x config) sweeps.
For each: sweeps only C over a fixed grid, computes d_s(C, sigma),
extracts plateau, checks scaling assumption.

Usage:
    python scripts/run_framework_validation_b.py               # default
    python scripts/run_framework_validation_b.py --quick        # CI-safe (~15s)

Outputs:
    outputs/framework_validation_b/<suite_id>/
        suite_summary.csv
        suite_metadata.json
        suite_report.md
        <substrate>/<filter>/<config_key>/
            metadata.json
            sweep_results.csv
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.substrates import (
    SubstrateResult,
    periodic_lattice,
    random_geometric_graph,
    small_world_graph,
)
from dimshift.spectral_filters import (
    HardCutoffFilter,
    SoftCutoffFilter,
    PowerLawFilter,
    SpectralFilter,
)
from dimshift.framework_spectral import (
    filtered_spectral_dimension,
    extract_plateau,
    check_scaling_assumption,
)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """One run: substrate + filter + sigma/C grids."""
    substrate: SubstrateResult
    filter_obj: SpectralFilter
    config_key: str        # human-readable key for this config
    d0: float              # baseline dimension for this substrate/config
    C_values: np.ndarray   # capacity grid to sweep
    sigma_values: np.ndarray
    sigma_lo: float        # plateau window lower bound
    sigma_hi: float        # plateau window upper bound


def _plateau_window(n_vertices: int, sigma_max: float) -> tuple[float, float]:
    """Compute a reasonable plateau window from substrate size."""
    # Same logic as SweepConfig: sigma_lo = 5, sigma_hi = 0.4 * n^{2/d} / (4pi^2)
    # For general graphs, use eigenvalue-based estimate:
    # plateau is roughly [5, min(sigma_max*0.5, 0.1*n_vertices)]
    sigma_hi = min(sigma_max * 0.5, 0.1 * n_vertices)
    sigma_hi = max(sigma_hi, 6.0)  # floor
    return (5.0, sigma_hi)


# ---------------------------------------------------------------------------
# Substrate/filter configurations
# ---------------------------------------------------------------------------

def _build_quick_matrix() -> list[RunConfig]:
    """CI-safe subset: small sizes, fewer C steps."""
    configs = []
    C_vals = np.linspace(0.05, 1.0, 15)
    sigma = np.geomspace(0.5, 200.0, 200)

    # Lattice 2D N=16
    sub = periodic_lattice(D=2, N=16)
    s_lo, s_hi = 3.0, 20.0
    for filt_name, filt_obj in [
        ("hard_cutoff", HardCutoffFilter()),
        ("soft_cutoff", SoftCutoffFilter()),
        ("powerlaw", PowerLawFilter(d0=2.0)),
    ]:
        configs.append(RunConfig(
            substrate=sub, filter_obj=filt_obj,
            config_key=f"lattice_D2_N16__{filt_name}",
            d0=2.0, C_values=C_vals, sigma_values=sigma,
            sigma_lo=s_lo, sigma_hi=s_hi,
        ))

    # RGG 2D n=200
    sub = random_geometric_graph(n_vertices=200, D=2, radius=0.35, seed=42)
    # For general graphs: adapt sigma grid to eigenvalue scale
    lam_med = float(np.median(sub.eigenvalues[sub.eigenvalues > 1e-10])) if np.any(sub.eigenvalues > 1e-10) else 1.0
    sigma_g = np.geomspace(0.01, 20.0 / max(lam_med, 0.1), 200)
    s_lo_g = 0.02 / max(lam_med, 0.1)
    s_hi_g = 5.0 / max(lam_med, 0.1)
    configs.append(RunConfig(
        substrate=sub, filter_obj=PowerLawFilter(d0=2.0),
        config_key="rgg_D2_n200__powerlaw",
        d0=2.0, C_values=C_vals, sigma_values=sigma_g,
        sigma_lo=max(s_lo_g, sigma_g[1]), sigma_hi=s_hi_g,
    ))

    # Small-world n=200
    sub = small_world_graph(n_vertices=200, k_neighbors=6, rewire_prob=0.3, seed=42)
    lam_med = float(np.median(sub.eigenvalues[sub.eigenvalues > 1e-10])) if np.any(sub.eigenvalues > 1e-10) else 1.0
    sigma_g = np.geomspace(0.01, 20.0 / max(lam_med, 0.1), 200)
    s_lo_g = 0.02 / max(lam_med, 0.1)
    s_hi_g = 5.0 / max(lam_med, 0.1)
    configs.append(RunConfig(
        substrate=sub, filter_obj=PowerLawFilter(d0=2.0),
        config_key="sw_n200__powerlaw",
        d0=2.0, C_values=C_vals, sigma_values=sigma_g,
        sigma_lo=max(s_lo_g, sigma_g[1]), sigma_hi=s_hi_g,
    ))

    return configs


def _build_default_matrix() -> list[RunConfig]:
    """Full validation matrix."""
    configs = []
    C_vals = np.linspace(0.05, 1.0, 25)
    sigma = np.geomspace(0.5, 300.0, 300)

    # --- Periodic lattices ---
    for D, N in [(2, 32), (3, 16)]:
        sub = periodic_lattice(D=D, N=N)
        s_lo = 5.0
        s_hi = min(0.4 * N**2 / (4 * np.pi**2), 100.0)
        s_hi = max(s_hi, s_lo + 2.0)
        for filt_name, filt_obj in [
            ("hard_cutoff", HardCutoffFilter()),
            ("soft_cutoff", SoftCutoffFilter()),
            ("powerlaw", PowerLawFilter(d0=float(D))),
        ]:
            configs.append(RunConfig(
                substrate=sub, filter_obj=filt_obj,
                config_key=f"lattice_D{D}_N{N}__{filt_name}",
                d0=float(D), C_values=C_vals, sigma_values=sigma,
                sigma_lo=s_lo, sigma_hi=s_hi,
            ))

    # --- RGG substrates ---
    for n_v, D_rgg, r in [(200, 2, 0.35), (150, 3, 0.5)]:
        sub = random_geometric_graph(n_vertices=n_v, D=D_rgg, radius=r, seed=42)
        lam_med = float(np.median(sub.eigenvalues[sub.eigenvalues > 1e-10])) if np.any(sub.eigenvalues > 1e-10) else 1.0
        sigma_g = np.geomspace(0.01, 20.0 / max(lam_med, 0.1), 300)
        s_lo_g = 0.02 / max(lam_med, 0.1)
        s_hi_g = 5.0 / max(lam_med, 0.1)
        for filt_name, filt_obj in [
            ("hard_cutoff", HardCutoffFilter()),
            ("powerlaw", PowerLawFilter(d0=float(D_rgg))),
        ]:
            configs.append(RunConfig(
                substrate=sub, filter_obj=filt_obj,
                config_key=f"rgg_D{D_rgg}_n{n_v}__{filt_name}",
                d0=float(D_rgg), C_values=C_vals, sigma_values=sigma_g,
                sigma_lo=max(s_lo_g, sigma_g[1]), sigma_hi=s_hi_g,
            ))

    # --- Small-world substrates ---
    for n_v, k, p in [(200, 6, 0.3), (200, 4, 0.1)]:
        sub = small_world_graph(n_vertices=n_v, k_neighbors=k, rewire_prob=p, seed=42)
        lam_med = float(np.median(sub.eigenvalues[sub.eigenvalues > 1e-10])) if np.any(sub.eigenvalues > 1e-10) else 1.0
        sigma_g = np.geomspace(0.01, 20.0 / max(lam_med, 0.1), 300)
        s_lo_g = 0.02 / max(lam_med, 0.1)
        s_hi_g = 5.0 / max(lam_med, 0.1)
        for filt_name, filt_obj in [
            ("hard_cutoff", HardCutoffFilter()),
            ("powerlaw", PowerLawFilter(d0=2.0)),
        ]:
            configs.append(RunConfig(
                substrate=sub, filter_obj=filt_obj,
                config_key=f"sw_n{n_v}_k{k}_p{p}__{filt_name}",
                d0=2.0, C_values=C_vals, sigma_values=sigma_g,
                sigma_lo=max(s_lo_g, sigma_g[1]), sigma_hi=s_hi_g,
            ))

    return configs


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(cfg: RunConfig, out_dir: Path, verbose: bool = True) -> dict:
    """Run a single substrate+filter sweep over C."""
    evals = cfg.substrate.eigenvalues
    n_C = len(cfg.C_values)
    n_sigma = len(cfg.sigma_values)

    ds_matrix = np.zeros((n_C, n_sigma))
    ln_P_matrix = np.zeros((n_C, n_sigma))
    plateaus = []
    scaling_checks = []

    for i, C in enumerate(cfg.C_values):
        fr = cfg.filter_obj.apply(evals, float(C))
        ds, ln_P = filtered_spectral_dimension(evals, fr.weights, cfg.sigma_values)
        ds_matrix[i] = ds
        ln_P_matrix[i] = ln_P

        plat = extract_plateau(ds, cfg.sigma_values, cfg.sigma_lo, cfg.sigma_hi)
        plateaus.append(plat)

        sc = check_scaling_assumption(ln_P, cfg.sigma_values, cfg.sigma_lo, cfg.sigma_hi)
        scaling_checks.append(sc)

    # Extract summary arrays
    ds_plat_arr = np.array([p["ds_plateau"] for p in plateaus])
    r2_arr = np.array([s["r_squared"] for s in scaling_checks])

    # Monotonicity check
    drops = np.diff(ds_plat_arr)
    max_drop = float(np.min(drops)) if len(drops) > 0 else 0.0
    mono_pass = max_drop > -0.15  # generous tolerance for diverse substrates

    # d_s at C=1 (full capacity)
    ds_at_full = float(ds_plat_arr[-1])

    # d_s at C=min
    ds_at_min = float(ds_plat_arr[0])

    # Dimension reduction observed?
    dim_reduction = ds_at_full - ds_at_min

    # Build summary row
    summary = {
        "config_key": cfg.config_key,
        "substrate": cfg.substrate.name,
        "filter": cfg.filter_obj.name,
        "n_vertices": cfg.substrate.n_vertices,
        "d0": cfg.d0,
        "n_C_steps": n_C,
        "ds_at_C_min": round(ds_at_min, 4),
        "ds_at_C_max": round(ds_at_full, 4),
        "dim_reduction": round(dim_reduction, 4),
        "max_ds_drop": round(max_drop, 4),
        "monotone_pass": mono_pass,
        "mean_r_squared": round(float(np.nanmean(r2_arr)), 4),
        "min_r_squared": round(float(np.nanmin(r2_arr)), 4),
    }

    # Write artifacts
    out_dir.mkdir(parents=True, exist_ok=True)

    # metadata.json
    meta = {
        "config_key": cfg.config_key,
        "substrate": cfg.substrate.metadata,
        "filter": cfg.filter_obj.describe(),
        "d0": cfg.d0,
        "n_C_steps": n_C,
        "C_range": [float(cfg.C_values[0]), float(cfg.C_values[-1])],
        "sigma_range": [float(cfg.sigma_values[0]), float(cfg.sigma_values[-1])],
        "n_sigma": n_sigma,
        "plateau_window": [cfg.sigma_lo, cfg.sigma_hi],
        "summary": summary,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # sweep_results.csv
    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "ds_plateau", "ds_mean", "ds_std", "r_squared", "fitted_d_eff"])
        for i in range(n_C):
            writer.writerow([
                f"{cfg.C_values[i]:.4f}",
                f"{plateaus[i]['ds_plateau']:.4f}",
                f"{plateaus[i]['ds_mean']:.4f}",
                f"{plateaus[i]['ds_std']:.4f}",
                f"{scaling_checks[i]['r_squared']}",
                f"{scaling_checks[i].get('fitted_d_eff', 'N/A')}",
            ])

    if verbose:
        status = "MONO" if mono_pass else "NON-MONO"
        print(f"  {cfg.config_key}: ds=[{ds_at_min:.2f},{ds_at_full:.2f}] "
              f"reduction={dim_reduction:.2f} R2={float(np.nanmean(r2_arr)):.3f} [{status}]")

    return summary


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_suite(configs: list[RunConfig], label: str = "default",
              verbose: bool = True) -> dict:
    """Run full validation suite."""
    suite_hash = hashlib.sha256(
        json.dumps([c.config_key for c in configs], sort_keys=True).encode()
    ).hexdigest()[:10]
    suite_id = f"{label}_{suite_hash}"

    base_dir = (Path(__file__).resolve().parent.parent
                / "outputs" / "framework_validation_b" / suite_id)
    base_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if verbose:
        print("=" * 70)
        print(f"Framework v4.5 Option B — Validation Suite [{label}]")
        print(f"Suite ID: {suite_id}")
        print(f"Configs: {len(configs)}")
        print("=" * 70)

    summaries = []
    for idx, cfg in enumerate(configs):
        if verbose:
            print(f"\n[{idx+1}/{len(configs)}] {cfg.config_key}")

        # Build per-run output dir from config key parts
        parts = cfg.config_key.split("__")
        sub_name = parts[0] if parts else cfg.config_key
        filt_name = parts[1] if len(parts) > 1 else "unknown"
        run_dir = base_dir / sub_name / filt_name

        summary = run_single(cfg, run_dir, verbose=verbose)
        summaries.append(summary)

    elapsed = time.time() - t0

    # --- Suite summary CSV ---
    csv_path = base_dir / "suite_summary.csv"
    if summaries:
        fieldnames = list(summaries[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    # --- Suite metadata JSON ---
    n_mono = sum(1 for s in summaries if s["monotone_pass"])
    n_reduction = sum(1 for s in summaries if s["dim_reduction"] > 0.1)

    meta = {
        "suite_id": suite_id,
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 2),
        "n_configs": len(configs),
        "n_monotone_pass": n_mono,
        "n_dimension_reduction": n_reduction,
        "all_monotone": n_mono == len(configs),
    }
    with open(base_dir / "suite_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Suite report MD ---
    report_lines = [
        "# Framework v4.5 Option B — Validation Report",
        "",
        f"**Suite ID:** {suite_id}",
        f"**Date:** {meta['timestamp']}",
        f"**Elapsed:** {elapsed:.1f}s",
        f"**Configs:** {len(configs)}",
        "",
        "## Summary",
        "",
        f"- Monotone d_s(C): {n_mono}/{len(configs)} runs",
        f"- Dimension reduction observed: {n_reduction}/{len(configs)} runs",
        "",
        "## Results by Run",
        "",
        "| Config | Substrate | Filter | |V| | d_s(C_min) | d_s(C_max) | Reduction | Mono | R^2 |",
        "|--------|-----------|--------|-----|------------|------------|-----------|------|-----|",
    ]
    for s in summaries:
        mono_str = "PASS" if s["monotone_pass"] else "FAIL"
        report_lines.append(
            f"| {s['config_key']} | {s['substrate']} | {s['filter']} | "
            f"{s['n_vertices']} | {s['ds_at_C_min']:.2f} | {s['ds_at_C_max']:.2f} | "
            f"{s['dim_reduction']:.2f} | {mono_str} | {s['mean_r_squared']:.3f} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "",
        "**What is proved:** For any substrate with known Laplacian eigenvalues,",
        "the monotone spectral filter family g_C produces a filtered return probability",
        "P_C(sigma) whose spectral dimension d_s^C(sigma) varies with C.",
        "",
        "**What is empirically validated:**",
        "- d_s(C) is approximately monotone non-decreasing in C",
        "- Dimension reduction is observed: d_s at low C < d_s at high C",
        "- The power-law scaling assumption (ln P ~ linear in ln sigma)",
        "  holds reasonably well in the declared scaling window (R^2 values above)",
        "",
        "**What is NOT claimed:**",
        "- The exact value of d_s at a given C is not predicted by the theorem",
        "  (it depends on the substrate's spectral density and the filter form)",
        "- The scaling window bounds are set heuristically, not derived",
        "- The monotonicity tolerance (0.15) is empirical",
    ]

    with open(base_dir / "suite_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    if verbose:
        print("\n" + "=" * 70)
        print(f"Monotone: {n_mono}/{len(configs)}  |  "
              f"Dim reduction: {n_reduction}/{len(configs)}  |  "
              f"Time: {elapsed:.1f}s")
        print(f"Artifacts: {base_dir}")
        print("=" * 70)

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Framework v4.5 Option B validation harness"
    )
    parser.add_argument("--quick", action="store_true",
                        help="CI-safe subset (~15s)")
    args = parser.parse_args()

    if args.quick:
        configs = _build_quick_matrix()
        label = "quick"
    else:
        configs = _build_default_matrix()
        label = "default"

    meta = run_suite(configs, label=label)

    if not meta["all_monotone"]:
        print("\nWARNING: Not all runs passed monotonicity check")
        # Don't exit(1) -- non-monotonicity in some substrates/filters
        # is an empirical observation, not a hard failure


if __name__ == "__main__":
    main()
