#!/usr/bin/env python3
"""
Canonical experiment entrypoint: Framework v4.5 Capacity → Dimension Shift.

Demonstrates that the same fixed substrate yields different effective geometry
(spectral dimension) when only the observational capacity C_geo changes.

Usage:
    python scripts/run_capacity_dimshift.py                  # default (3D, N=64)
    python scripts/run_capacity_dimshift.py --preset small   # quick test
    python scripts/run_capacity_dimshift.py --preset large   # high-res
    python scripts/run_capacity_dimshift.py --D 4 --N 16     # custom

Outputs are written to:
    outputs/capacity_dimshift/<run_id>/
        metadata.json          — all fixed parameters and run info
        sweep_results.csv      — C_geo, d_eff_nominal, ds_plateau, weights
        ds_matrix.json         — full d_s(C_geo, σ) matrix
        thresholds.json        — detected integer/half-integer thresholds
        heatmap.png            — d_s heatmap (money plot)
        representative_curves.png — d_s(σ) at key C_geo values
        phase_diagram.png      — d_eff vs C_geo
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure dimshift is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift import SweepConfig, run_capacity_sweep
from dimshift.sweep import write_artifacts
from dimshift.plotting import save_all_figures


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "small": SweepConfig(D=3, N=32, C_geo_steps=15, n_sigma=200),
    "medium": SweepConfig(D=3, N=64, C_geo_steps=30, n_sigma=400),
    "large": SweepConfig(D=3, N=128, C_geo_steps=50, n_sigma=600),
    "4d": SweepConfig(D=4, N=16, C_geo_steps=40, n_sigma=400),
}


def main():
    parser = argparse.ArgumentParser(
        description="Framework v4.5 Capacity → Dimension Shift experiment"
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        help="Use a named preset configuration")
    parser.add_argument("--D", type=int, help="Lattice dimension")
    parser.add_argument("--N", type=int, help="Lattice side length")
    parser.add_argument("--C-steps", type=int, dest="C_steps",
                        help="Number of C_geo steps")
    parser.add_argument("--n-sigma", type=int, dest="n_sigma",
                        help="Number of sigma points")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Build config
    if args.preset:
        config = PRESETS[args.preset]
        print(f"Using preset: {args.preset}")
    else:
        kwargs = {}
        if args.D is not None:
            kwargs["D"] = args.D
        if args.N is not None:
            kwargs["N"] = args.N
        if args.C_steps is not None:
            kwargs["C_geo_steps"] = args.C_steps
        if args.n_sigma is not None:
            kwargs["n_sigma"] = args.n_sigma
        config = SweepConfig(**kwargs)

    # Print config
    print("=" * 60)
    print("Framework v4.5 — Capacity → Dimension Shift Experiment")
    print("=" * 60)
    print(f"  Lattice:      {config.D}D periodic cubic, N={config.N}")
    print(f"  Total sites:  {config.N**config.D:,}")
    print(f"  C_geo grid:   [{config.C_geo_min}, {config.C_geo_max}] × {config.C_geo_steps} steps")
    print(f"  σ grid:       [{config.sigma_min}, {config.sigma_max}] × {config.n_sigma} points (log-spaced)")
    lo, hi = config.plateau_window()
    print(f"  Plateau:      σ ∈ [{lo:.1f}, {hi:.1f}]")
    print(f"  Fixed:        a=1, continuous-time heat kernel, central FD (np.gradient; fwd/bwd at endpoints)")
    print()

    # Run sweep
    print("Running capacity sweep...", flush=True)
    result = run_capacity_sweep(config)
    print(f"  Done in {result.elapsed_s:.2f}s  (run_id={result.run_id})")
    print()

    # Print summary
    print("Detected thresholds:")
    for t in result.thresholds:
        print(f"  d_s = {t['target_dimension']:.1f}  at  C_geo = {t['C_geo_threshold']:.4f}"
              f"  (bracket [{t['bracket'][0]:.3f}, {t['bracket'][1]:.3f}])")
    print()

    # Validation check — use interpolation to get d_s at exactly C_geo=1.0
    if config.C_geo_max >= 1.0:
        ds_full = result.plateau_at(1.0)
    else:
        ds_full = result.ds_plateau[-1]
    error_pct = abs(ds_full - config.D) / config.D * 100
    c_label = "1" if config.C_geo_max >= 1.0 else f"{config.C_geo_max}"
    print(f"Full-capacity check: d_s(C_geo={c_label}) = {ds_full:.4f}  "
          f"(expect {config.D}.0, error = {error_pct:.2f}%)")
    if error_pct > 1.0:
        print("  WARNING: error exceeds 1% — consider increasing N or adjusting σ range")
    else:
        print("  PASS: within 1% tolerance")
    print()

    # Write artifacts
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%dT%H%M%S")
        out_dir = Path(__file__).resolve().parent.parent / "outputs" / "capacity_dimshift" / f"{ts}_{result.run_id}"
    print(f"Writing artifacts to: {out_dir}")

    artifact_paths = write_artifacts(result, out_dir)
    for name, p in artifact_paths.items():
        print(f"  {name}: {p}")

    # Generate plots
    try:
        fig_paths = save_all_figures(result, out_dir)
        for name, p in fig_paths.items():
            print(f"  {name}: {p}")
    except ImportError:
        print("  (matplotlib not installed — skipping plots)")

    print()
    print("Experiment complete.")
    return result


if __name__ == "__main__":
    main()
