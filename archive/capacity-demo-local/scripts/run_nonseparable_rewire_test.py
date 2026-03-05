#!/usr/bin/env python3
"""
CLI entrypoint for the non-separable Laplacian rewire test.

Tests whether the capacity-controlled dimension staircase survives
when the graph Laplacian is made non-separable via edge rewiring.

Usage:
    python -m scripts.run_nonseparable_rewire_test --preset small
    python -m scripts.run_nonseparable_rewire_test --preset medium
    python -m scripts.run_nonseparable_rewire_test --D 3 --N 64 --rates 0.03
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent to path so we can import dimshift
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.nonseparable import (
    run_nonseparable_sweep,
    evaluate_criteria,
    write_artifacts,
)


PRESETS = {
    "small": {
        "configs": [
            {"D": 3, "N": 16, "rates": [0.01, 0.03, 0.05]},
        ],
        "n_sigma": 200, "n_probes": 20, "m_lanczos": 50,
    },
    "medium": {
        "configs": [
            {"D": 3, "N": 32, "rates": [0.01, 0.03, 0.05]},
        ],
        "n_sigma": 300, "n_probes": 25, "m_lanczos": 60,
    },
    "large": {
        "configs": [
            {"D": 3, "N": 32, "rates": [0.01, 0.03, 0.05]},
            {"D": 3, "N": 64, "rates": [0.01, 0.03]},
        ],
        "n_sigma": 400, "n_probes": 30, "m_lanczos": 80,
    },
}


def run_single(
    D: int, N: int, rate: float, seed: int,
    n_sigma: int, n_probes: int, m_lanczos: int,
    C_geo_steps: int = 30, verbose: bool = True,
) -> dict:
    """Run a single configuration and return criteria dict."""
    print(f"\n{'='*70}")
    print(f"  D={D}  N={N}  r={rate}  seed={seed}")
    print(f"  n_sigma={n_sigma}  n_probes={n_probes}  m_lanczos={m_lanczos}")
    print(f"{'='*70}")

    result = run_nonseparable_sweep(
        D=D, N=N, rewire_rate=rate, seed=seed,
        C_geo_steps=C_geo_steps, n_sigma=n_sigma,
        n_probes=n_probes, m_lanczos=m_lanczos,
        verbose=verbose,
    )

    criteria = evaluate_criteria(result)

    # Write artifacts
    run_id = f"D{D}_N{N}_r{rate}_s{seed}"
    out_dir = Path("outputs") / "nonseparable_rewire" / run_id
    paths = write_artifacts(result, out_dir)

    # Print results
    print(f"\n--- Results: D={D} N={N} r={rate} ---")
    print(f"  Elapsed: {result.elapsed_s:.1f}s  Method: {result.method}")
    print(f"  ds_plateau range: [{result.ds_plateau.min():.3f}, {result.ds_plateau.max():.3f}]")

    for key in ["A", "B", "C"]:
        c = criteria[key]
        status = "PASS" if c.get("pass") else "FAIL"
        print(f"  Criterion {key} ({c['name']}): {status}")
        if key == "A":
            print(f"    max_drop={c.get('max_drop', 'N/A')}  tol={c.get('tolerance', 'N/A')}")
        elif key == "B" and "jumps" in c:
            for j in c["jumps"]:
                print(f"    Jump {j['rank']}: c_mid={j['c_mid']} pred={j['predicted_c']} "
                      f"err={j['error']} mag={j['magnitude']}")
        elif key == "C":
            if "segment_means" in c:
                print(f"    means={c.get('segment_means')}  stds={c.get('segment_stds')}")
                print(f"    gaps={c.get('gaps')}  gap_ratio={c.get('gap_ratio', 'N/A')}")

    overall = "PASS" if criteria["overall_pass"] else "FAIL"
    print(f"  Overall: {overall}")
    print(f"  Artifacts: {out_dir}")

    return criteria


def run_preset(preset_name: str, seed: int = 42, verbose: bool = True):
    """Run a named preset configuration."""
    if preset_name not in PRESETS:
        print(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
        sys.exit(1)

    preset = PRESETS[preset_name]
    n_sigma = preset["n_sigma"]
    n_probes = preset["n_probes"]
    m_lanczos = preset["m_lanczos"]

    all_results = []
    t0 = time.time()

    for cfg in preset["configs"]:
        D, N = cfg["D"], cfg["N"]
        for rate in cfg["rates"]:
            criteria = run_single(
                D, N, rate, seed,
                n_sigma=n_sigma, n_probes=n_probes, m_lanczos=m_lanczos,
                verbose=verbose,
            )
            all_results.append({
                "D": D, "N": N, "rate": rate,
                "pass": criteria["overall_pass"],
            })

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Preset '{preset_name}' complete in {elapsed:.1f}s")
    print(f"{'='*70}")
    for r in all_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  D={r['D']} N={r['N']} r={r['rate']}: {status}")

    n_pass = sum(1 for r in all_results if r["pass"])
    print(f"\n  {n_pass}/{len(all_results)} configurations passed")


def run_custom(
    D: int, N: int, rates: list[float], seed: int,
    n_sigma: int, n_probes: int, m_lanczos: int,
    C_geo_steps: int = 30, verbose: bool = True,
):
    """Run a custom configuration."""
    all_results = []
    for rate in rates:
        criteria = run_single(
            D, N, rate, seed,
            n_sigma=n_sigma, n_probes=n_probes, m_lanczos=m_lanczos,
            C_geo_steps=C_geo_steps, verbose=verbose,
        )
        all_results.append({"rate": rate, "pass": criteria["overall_pass"]})

    n_pass = sum(1 for r in all_results if r["pass"])
    print(f"\n{n_pass}/{len(all_results)} configurations passed")


def main():
    parser = argparse.ArgumentParser(
        description="Non-separable Laplacian rewire test"
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        help="Run a named preset")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.03])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sigma", type=int, default=300)
    parser.add_argument("--n-probes", type=int, default=25)
    parser.add_argument("--m-lanczos", type=int, default=60)
    parser.add_argument("--C-geo-steps", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.preset:
        run_preset(args.preset, seed=args.seed, verbose=not args.quiet)
    else:
        run_custom(
            D=args.D, N=args.N, rates=args.rates, seed=args.seed,
            n_sigma=args.n_sigma, n_probes=args.n_probes,
            m_lanczos=args.m_lanczos, C_geo_steps=args.C_geo_steps,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
