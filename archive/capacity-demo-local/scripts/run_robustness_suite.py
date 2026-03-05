#!/usr/bin/env python3
"""
Robustness suite: run canonical capacity sweeps across multiple (D, N) configs.

Each individual run sweeps ONLY C_geo with everything else fixed.
Between runs, D and N vary to test that the capacity→dimension-shift
effect is robust across lattice sizes and dimensions.

Usage:
    python scripts/run_robustness_suite.py              # default suite
    python scripts/run_robustness_suite.py --quick       # fast CI-safe subset

Outputs:
    outputs/capacity_dimshift_suite/<suite_id>/
        <run_id>/                     — per-run artifacts (same as canonical)
        suite_summary.csv             — one row per run
        suite_metadata.json           — full suite config
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift import SweepConfig, run_capacity_sweep
from dimshift.sweep import write_artifacts
from dimshift.plotting import save_all_figures


# ---------------------------------------------------------------------------
# Suite configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuiteEntry:
    """One run in the suite: a fixed (D, N) pair."""
    D: int
    N: int


# Default suite: 6 runs across D ∈ {2,3,4}, N ∈ {32,64,128}
# Skip D=4 N≥64 (too many modes for a quick suite) and D=3/4 N=128
DEFAULT_SUITE = [
    SuiteEntry(D=2, N=32),
    SuiteEntry(D=2, N=64),
    SuiteEntry(D=2, N=128),
    SuiteEntry(D=3, N=32),
    SuiteEntry(D=3, N=64),
    SuiteEntry(D=4, N=16),
]

# Quick suite: 4 runs, CI-safe (<5s total)
QUICK_SUITE = [
    SuiteEntry(D=2, N=32),
    SuiteEntry(D=3, N=32),
    SuiteEntry(D=3, N=64),
    SuiteEntry(D=4, N=16),
]

# Shared sweep parameters (identical across all runs)
SHARED_C_GEO_STEPS = 30
SHARED_N_SIGMA = 300


def _find_threshold(thresholds: list[dict], target: float) -> float:
    """Extract C_geo threshold for a given target d_s, or NaN if not found."""
    for t in thresholds:
        if t["target_dimension"] == target:
            return t["C_geo_threshold"]
    return float("nan")


def run_suite(entries: list[SuiteEntry], label: str, output_base: Path) -> Path:
    """Run the full suite and write all artifacts."""

    suite_id = hashlib.sha256(
        f"{label}:{[(e.D, e.N) for e in entries]}".encode()
    ).hexdigest()[:10]
    suite_dir = output_base / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Framework v4.5 — Robustness Suite  [{label}]")
    print(f"Suite ID: {suite_id}")
    print(f"Runs: {len(entries)}  |  C_geo steps: {SHARED_C_GEO_STEPS}  |  σ points: {SHARED_N_SIGMA}")
    print("=" * 65)

    summary_rows: list[dict] = []
    suite_t0 = time.time()

    for idx, entry in enumerate(entries):
        config = SweepConfig(
            D=entry.D,
            N=entry.N,
            C_geo_steps=SHARED_C_GEO_STEPS,
            n_sigma=SHARED_N_SIGMA,
        )

        print(f"\n[{idx+1}/{len(entries)}] D={entry.D}, N={entry.N} "
              f"({entry.N**entry.D:,} sites) ...", end=" ", flush=True)

        result = run_capacity_sweep(config)
        print(f"done in {result.elapsed_s:.2f}s  (run_id={result.run_id})")

        # Write per-run artifacts
        run_dir = suite_dir / result.run_id
        write_artifacts(result, run_dir)
        try:
            save_all_figures(result, run_dir)
        except ImportError:
            pass

        # Compute summary stats
        ds_plat = result.ds_plateau
        d_nom = result.d_eff_nominal
        rmse = float(np.sqrt(np.mean((ds_plat - d_nom) ** 2)))
        lo, hi = config.plateau_window()

        row = {
            "suite_id": suite_id,
            "run_id": result.run_id,
            "D": entry.D,
            "N": entry.N,
            "total_sites": entry.N ** entry.D,
            "C_steps": SHARED_C_GEO_STEPS,
            "sigma_lo": round(lo, 2),
            "sigma_hi": round(hi, 2),
            "ds_plateau_min": round(float(ds_plat.min()), 4),
            "ds_plateau_max": round(float(ds_plat.max()), 4),
            "plateau_d_eff_mean": round(float(ds_plat.mean()), 4),
            "plateau_d_eff_rmse_to_nominal": round(rmse, 4),
            "threshold_C_at_ds_1p5": round(_find_threshold(result.thresholds, 1.5), 4),
            "threshold_C_at_ds_2p5": round(_find_threshold(result.thresholds, 2.5), 4),
            "ds_at_full_capacity": round(float(ds_plat[-1]), 4),
            "full_capacity_error_pct": round(abs(float(ds_plat[-1]) - entry.D) / entry.D * 100, 2),
            "runtime_s": result.elapsed_s,
        }
        summary_rows.append(row)

        # Print thresholds
        for t in result.thresholds:
            if t["target_dimension"] in (1.5, 2.5, 3.5):
                print(f"    d_s={t['target_dimension']:.1f} at C_geo={t['C_geo_threshold']:.4f}")

    suite_elapsed = round(time.time() - suite_t0, 2)

    # Write suite_summary.csv
    csv_path = suite_dir / "suite_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write suite_metadata.json
    meta_path = suite_dir / "suite_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "suite_id": suite_id,
            "label": label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_elapsed_s": suite_elapsed,
            "n_runs": len(entries),
            "shared_C_geo_steps": SHARED_C_GEO_STEPS,
            "shared_n_sigma": SHARED_N_SIGMA,
            "entries": [asdict(e) for e in entries],
            "summary": summary_rows,
        }, f, indent=2)

    # Print final summary
    print("\n" + "=" * 65)
    print("Suite Summary")
    print("=" * 65)
    print(f"{'D':>3} {'N':>5} {'ds_min':>7} {'ds_max':>7} {'RMSE':>7} {'ds(C=1)':>8} {'err%':>6} {'time':>6}")
    print("-" * 55)
    for r in summary_rows:
        print(f"{r['D']:3d} {r['N']:5d} {r['ds_plateau_min']:7.3f} {r['ds_plateau_max']:7.3f} "
              f"{r['plateau_d_eff_rmse_to_nominal']:7.4f} {r['ds_at_full_capacity']:8.4f} "
              f"{r['full_capacity_error_pct']:5.2f}% {r['runtime_s']:5.2f}s")

    print(f"\nTotal time: {suite_elapsed}s")
    print(f"Artifacts:  {suite_dir}")
    print(f"Summary:    {csv_path}")
    return suite_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Framework v4.5 Robustness Suite — multi-(D,N) capacity sweeps"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run quick CI-safe subset (4 runs)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output base directory")
    args = parser.parse_args()

    if args.quick:
        entries, label = QUICK_SUITE, "quick"
    else:
        entries, label = DEFAULT_SUITE, "default"

    output_base = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent.parent / "outputs" / "capacity_dimshift_suite"
    )

    run_suite(entries, label, output_base)


if __name__ == "__main__":
    main()
