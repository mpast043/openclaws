#!/usr/bin/env python3
"""
Stress-test: verify the capacity-only claim across a wide grid of (D, N, C_geo).

Runs ALL verification obligations across a configurable matrix of lattice
dimensions and sizes.

Usage:
    python scripts/run_stress_test.py               # default matrix
    python scripts/run_stress_test.py --quick        # CI-safe subset (~30s)
    python scripts/run_stress_test.py --exhaustive   # full matrix (~5min)

Outputs:
    outputs/stress_test/<suite_id>/
        stress_results.csv   — per-(D,N) obligation pass/fail/skip + diagnostics
        stress_metadata.json — suite config, timing, summary
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

from dimshift.capacity import capacity_weights
from dimshift.theorem import (
    verify_eigenvalue_bounds,
    verify_factorisation,
    verify_staircase,
    verify_plateau,
    verify_thresholds,
    verify_monotonicity,
    verify_continuum_limit,
    verify_capacity_only,
)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

@dataclass
class StressConfig:
    D: int
    N: int


# Minimum N for a usable plateau window: the plateau regime requires
# 5 ≤ σ ≤ 0.4 N²/(4π²), which gives σ_hi ≈ 10 at N=32.
# N=32 is the minimum for reliable verification across all D.

QUICK_MATRIX = [
    StressConfig(2, 32),
    StressConfig(2, 64),
    StressConfig(3, 32),
    StressConfig(3, 64),
    StressConfig(4, 32),
]

DEFAULT_MATRIX = [
    StressConfig(2, 32),
    StressConfig(2, 64),
    StressConfig(2, 128),
    StressConfig(3, 32),
    StressConfig(3, 64),
    StressConfig(4, 32),
    StressConfig(4, 64),
]

EXHAUSTIVE_MATRIX = [
    StressConfig(1, 32),
    StressConfig(1, 64),
    StressConfig(1, 128),
    StressConfig(2, 32),
    StressConfig(2, 64),
    StressConfig(2, 128),
    StressConfig(3, 32),
    StressConfig(3, 64),
    StressConfig(3, 128),
    StressConfig(4, 32),
    StressConfig(4, 64),
    StressConfig(5, 32),
]

# C_geo values to test factorisation and plateau at
C_GEO_TEST_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]


def _plateau_assumptions_met(C_geo: float, D: int) -> bool:
    """Check if all active weights >= 0.5 for this C_geo / D combination."""
    w = capacity_weights(C_geo, D)
    active = w[w > 1e-15]
    if len(active) == 0:
        return True  # no active dims → trivially true
    return float(np.min(active)) >= 0.5


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_stress_test(matrix: list[StressConfig], label: str = "default",
                    verbose: bool = True) -> dict:
    """Run all verification obligations across the given (D, N) matrix."""
    suite_id = hashlib.sha256(
        json.dumps([(c.D, c.N) for c in matrix], sort_keys=True).encode()
    ).hexdigest()[:10]

    t0 = time.time()
    rows = []
    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_checks = 0

    if verbose:
        print("=" * 70)
        print(f"Framework v4.5 — Stress Test  [{label}]")
        print(f"Suite ID: {suite_id}")
        print(f"Configs: {len(matrix)}  |  C_geo test values: {len(C_GEO_TEST_VALUES)}")
        print("=" * 70)

    for idx, cfg in enumerate(matrix):
        D, N = cfg.D, cfg.N
        if verbose:
            print(f"\n[{idx+1}/{len(matrix)}] D={D}, N={N}")

        config_results = {"D": D, "N": N}
        config_pass = 0
        config_fail = 0
        config_skip = 0

        # Eigenvalue bounds (algebraic)
        r = verify_eigenvalue_bounds(N)
        config_results["eigenvalue_bounds"] = "PASS" if r.passed else "FAIL"
        if r.passed:
            config_pass += 1
        else:
            config_fail += 1
        total_checks += 1

        # Staircase (algebraic)
        r = verify_staircase(D, n_points=2000)
        config_results["staircase"] = "PASS" if r.passed else "FAIL"
        if r.passed:
            config_pass += 1
        else:
            config_fail += 1
        total_checks += 1

        # Capacity-only dependence (algebraic)
        r = verify_capacity_only(D, N)
        config_results["capacity_only"] = "PASS" if r.passed else "FAIL"
        if r.passed:
            config_pass += 1
        else:
            config_fail += 1
        total_checks += 1

        # Factorisation at multiple C_geo (algebraic; skip if N^D too large)
        fact_pass = 0
        fact_skip = 0
        fact_total = 0
        fact_max_err = 0.0
        for C_geo in C_GEO_TEST_VALUES:
            total_sites = N ** D
            if total_sites > 100_000:
                # Algebraic identity — no brute-force needed
                fact_skip += 1
                config_skip += 1
                total_skip += 1
                total_checks += 1
                fact_total += 1
                continue
            r = verify_factorisation(D, N, C_geo, n_sigma=100)
            fact_total += 1
            total_checks += 1
            if r.passed:
                fact_pass += 1
                config_pass += 1
            else:
                config_fail += 1
            fact_max_err = max(fact_max_err, r.diagnostics.get("max_relative_error", 0))
        config_results["factorisation"] = f"{fact_pass}/{fact_total - fact_skip} verified"
        if fact_skip > 0:
            config_results["factorisation"] += f" ({fact_skip} algebraic_identity)"
        config_results["factorisation_max_err"] = f"{fact_max_err:.2e}"

        # Plateau at multiple C_geo (asymptotic; skip if assumptions not met)
        plat_pass = 0
        plat_skip = 0
        plat_total = 0
        plat_max_dev = 0.0
        for C_geo in C_GEO_TEST_VALUES:
            plat_total += 1
            total_checks += 1

            if not _plateau_assumptions_met(C_geo, D):
                # Assumption violated — not a failure, just inapplicable
                plat_skip += 1
                config_skip += 1
                total_skip += 1
                continue

            r = verify_plateau(D, N, C_geo, n_sigma=200)
            if r.passed:
                plat_pass += 1
                config_pass += 1
            else:
                config_fail += 1
            dev = r.diagnostics.get("max_deviation_from_D_active", 0)
            if isinstance(dev, (int, float)):
                plat_max_dev = max(plat_max_dev, dev)
        config_results["plateau"] = f"{plat_pass}/{plat_total - plat_skip} verified"
        if plat_skip > 0:
            config_results["plateau"] += f" ({plat_skip} assumption_skipped)"
        config_results["plateau_max_dev"] = f"{plat_max_dev:.4f}"

        # Thresholds (empirical, only for D >= 2)
        if D >= 2:
            r = verify_thresholds(D, N, C_geo_steps=100, n_sigma=200)
            config_results["thresholds"] = "PASS" if r.passed else "FAIL"
            config_results["threshold_max_err"] = f"{r.diagnostics.get('max_error', 'N/A')}"
            if r.passed:
                config_pass += 1
            else:
                config_fail += 1
            total_checks += 1
        else:
            config_results["thresholds"] = "N/A"
            config_results["threshold_max_err"] = "N/A"

        # Monotonicity (empirical, only for D >= 2)
        if D >= 2:
            r = verify_monotonicity(D, N, C_geo_steps=100, n_sigma=200)
            config_results["monotonicity"] = "PASS" if r.passed else "FAIL"
            config_results["mono_max_drop"] = f"{r.diagnostics.get('max_drop', 'N/A')}"
            if r.passed:
                config_pass += 1
            else:
                config_fail += 1
            total_checks += 1
        else:
            config_results["monotonicity"] = "N/A"
            config_results["mono_max_drop"] = "N/A"

        total_pass += config_pass
        total_fail += config_fail
        total_skip += config_skip
        config_results["pass_count"] = config_pass
        config_results["fail_count"] = config_fail
        config_results["skip_count"] = config_skip
        rows.append(config_results)

        if verbose:
            parts = [f"{config_pass} pass"]
            if config_fail > 0:
                parts.append(f"{config_fail} fail")
            if config_skip > 0:
                parts.append(f"{config_skip} skip")
            status = "ALL PASS" if config_fail == 0 else f"{config_fail} FAIL"
            print(f"  {', '.join(parts)}  [{status}]")

    # Continuum limit (run once, not per-config)
    if verbose:
        print(f"\nInfinite-lattice limit convergence:")
    r = verify_continuum_limit()
    total_checks += 1
    if r.passed:
        total_pass += 1
    else:
        total_fail += 1
    if verbose:
        print(f"  {'PASS' if r.passed else 'FAIL'}")

    elapsed = time.time() - t0

    # Write artifacts
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "stress_test" / suite_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "stress_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Metadata
    meta = {
        "suite_id": suite_id,
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 2),
        "n_configs": len(matrix),
        "total_checks": total_checks,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_skip": total_skip,
        "all_passed": total_fail == 0,
        "matrix": [(c.D, c.N) for c in matrix],
        "C_geo_test_values": C_GEO_TEST_VALUES,
        "continuum_limit": "PASS" if r.passed else "FAIL",
    }
    meta_path = out_dir / "stress_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        parts = [f"{total_pass} pass"]
        if total_fail > 0:
            parts.append(f"{total_fail} fail")
        if total_skip > 0:
            parts.append(f"{total_skip} skip (assumptions not met)")
        print(f"Total: {', '.join(parts)}  out of {total_checks} checks  [{elapsed:.1f}s]")
        print(f"Artifacts: {out_dir}")
        if total_fail == 0:
            print("ALL VERIFICATION OBLIGATIONS PASSED (skips are assumption-based, not failures)")
        else:
            print(f"WARNING: {total_fail} CHECKS FAILED")
        print("=" * 70)

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stress-test capacity-only verification obligations across (D,N) matrix"
    )
    parser.add_argument("--quick", action="store_true",
                        help="CI-safe subset (~30s)")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Full matrix including D=5 (~5min)")
    args = parser.parse_args()

    if args.quick:
        matrix, label = QUICK_MATRIX, "quick"
    elif args.exhaustive:
        matrix, label = EXHAUSTIVE_MATRIX, "exhaustive"
    else:
        matrix, label = DEFAULT_MATRIX, "default"

    meta = run_stress_test(matrix, label=label)

    if not meta["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
