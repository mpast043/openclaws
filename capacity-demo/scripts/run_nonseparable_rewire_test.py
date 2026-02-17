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
import math
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent to path so we can import dimshift
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.nonseparable import (
    run_nonseparable_sweep,
    evaluate_criteria,
    write_artifacts,
)


DEFAULT_REFINE_BANDS = [
    (0.27, 0.45, 24),
    (0.60, 0.78, 24),
]


PRESETS = {
    "small": {
        "configs": [
            {"D": 3, "N": 16, "rates": [0.01, 0.03, 0.05]},
        ],
        "n_sigma": 200,
        "C_geo_steps": 30,
    },
    "medium": {
        "configs": [
            {"D": 3, "N": 32, "rates": [0.01, 0.03, 0.05]},
        ],
        "n_sigma": 300,
        "C_geo_steps": 30,
    },
    "large": {
        "configs": [
            {"D": 3, "N": 32, "rates": [0.01, 0.03, 0.05]},
            {"D": 3, "N": 64, "rates": [0.01, 0.03]},
        ],
        "n_sigma": 400,
        "C_geo_steps": 30,
    },
}


def _auto_probe_budget(N: int, rewire_rate: float) -> tuple[int, int]:
    """Return (n_probes, m_lanczos) tuned for lattice size and rewire rate."""
    if N <= 16:
        n_probes = 80
        m_lanczos = 120
    elif N <= 32:
        n_probes = 55
        m_lanczos = 95
    else:
        n_probes = 35
        m_lanczos = 80

    if rewire_rate >= 0.08:
        boost = 1.35
    elif rewire_rate >= 0.05:
        boost = 1.2
    else:
        boost = 1.0

    n_probes = max(20, int(math.ceil(n_probes * boost)))
    lanczos_boost = 1.0 if boost == 1.0 else 1.0 + (boost - 1.0) * 0.6
    m_lanczos = max(60, int(math.ceil(m_lanczos * lanczos_boost)))
    return n_probes, m_lanczos


def run_single(
    D: int, N: int, rate: float, seed: int,
    n_sigma: int, n_probes: Optional[int], m_lanczos: Optional[int],
    C_geo_steps: int = 30, verbose: bool = True,
    refine_bands: Optional[list[tuple[float, float, int]]] = None,
    family: str = "lattice", family_kwargs: Optional[dict] = None,
) -> dict:
    """Run a single configuration and return criteria dict."""
    print(f"\n{'='*70}")
    print(f"  D={D}  N={N}  r={rate}  seed={seed}  family={family}")
    family_kwargs = family_kwargs or {}
    auto_probes, auto_lanczos = _auto_probe_budget(N, rate)
    eff_n_probes = n_probes if n_probes is not None else auto_probes
    eff_m_lanczos = m_lanczos if m_lanczos is not None else auto_lanczos
    probe_label = "auto" if n_probes is None else "manual"
    lanczos_label = "auto" if m_lanczos is None else "manual"
    print(f"  n_sigma={n_sigma}  n_probes={eff_n_probes} ({probe_label})  m_lanczos={eff_m_lanczos} ({lanczos_label})")
    print(f"{'='*70}")

    result = run_nonseparable_sweep(
        D=D, N=N, rewire_rate=rate, seed=seed,
        C_geo_steps=C_geo_steps, n_sigma=n_sigma,
        n_probes=eff_n_probes, m_lanczos=eff_m_lanczos,
        verbose=verbose, refine_bands=refine_bands,
        family=family, family_kwargs=family_kwargs,
    )

    criteria = evaluate_criteria(result)

    # Write artifacts
    run_id = f"D{D}_N{N}_r{rate}_s{seed}"
    out_dir = Path("outputs") / "nonseparable_rewire" / run_id
    paths = write_artifacts(result, out_dir)

    # Print results
    print(f"\n--- Results: D={D} N={N} r={rate} family={family} ---")
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
    graph_meta = result.graph_metadata or {}
    if graph_meta:
        extra = []
        if 'n_rewired' in graph_meta:
            extra.append(f"rewired={graph_meta['n_rewired']}")
        if 'n_local_edges' in graph_meta:
            extra.append(f"local_edges={graph_meta['n_local_edges']}")
        if 'n_edges' in graph_meta:
            extra.append(f"edges={graph_meta['n_edges']}")
        if 'radius' in graph_meta:
            extra.append(f"radius={graph_meta['radius']:.4f}")
        if extra:
            print(f"  Graph metadata: {'; '.join(extra)}")
    print(f"  Artifacts: {out_dir}")

    return criteria


def run_preset(
    preset_name: str,
    seed: int = 42,
    n_sigma_override: Optional[int] = None,
    n_probes_override: Optional[int] = None,
    m_lanczos_override: Optional[int] = None,
    C_geo_steps_override: Optional[int] = None,
    refine_bands: Optional[list[tuple[float, float, int]]] = None,
    family: str = "lattice", family_kwargs: Optional[dict] = None,
    verbose: bool = True,
):
    """Run a named preset configuration."""
    if preset_name not in PRESETS:
        print(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
        sys.exit(1)

    preset = PRESETS[preset_name]
    n_sigma = n_sigma_override if n_sigma_override is not None else preset.get("n_sigma", 300)
    C_geo_steps = C_geo_steps_override if C_geo_steps_override is not None else preset.get("C_geo_steps", 30)
    n_probes = n_probes_override if n_probes_override is not None else preset.get("n_probes")
    m_lanczos = m_lanczos_override if m_lanczos_override is not None else preset.get("m_lanczos")

    all_results = []
    t0 = time.time()

    for cfg in preset["configs"]:
        D, N = cfg["D"], cfg["N"]
        for rate in cfg["rates"]:
            criteria = run_single(
                D, N, rate, seed,
                n_sigma=n_sigma, n_probes=n_probes, m_lanczos=m_lanczos,
                C_geo_steps=C_geo_steps, verbose=verbose,
                refine_bands=refine_bands, family=family, family_kwargs=family_kwargs,
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
    n_sigma: int, n_probes: Optional[int], m_lanczos: Optional[int],
    C_geo_steps: int = 30, verbose: bool = True,
    refine_bands: Optional[list[tuple[float, float, int]]] = None,
    family: str = "lattice", family_kwargs: Optional[dict] = None,
):
    """Run a custom configuration."""
    all_results = []
    for rate in rates:
        criteria = run_single(
            D, N, rate, seed,
            n_sigma=n_sigma, n_probes=n_probes, m_lanczos=m_lanczos,
            C_geo_steps=C_geo_steps, verbose=verbose,
            refine_bands=refine_bands, family=family, family_kwargs=family_kwargs,
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
    parser.add_argument("--n-sigma", type=int, default=None,
                        help="Override sigma grid (default: preset value or 300 for custom runs)")
    parser.add_argument("--n-probes", type=int, default=None,
                        help="Override probe count (default: auto N/r-dependent budget)")
    parser.add_argument("--m-lanczos", type=int, default=None,
                        help="Override Lanczos depth (default: auto N/r-dependent budget)")
    parser.add_argument("--C-geo-steps", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-refine", action="store_true", help="Disable additional C_geo refinement bands")
    parser.add_argument("--refine-band", action="append", nargs=3, metavar=("LO", "HI", "STEPS"),
                        help="Add a custom refinement band (lo hi steps)")
    parser.add_argument("--family", choices=["lattice", "rgg"], default="lattice",
                        help="Graph family (default: lattice rewiring)")
    parser.add_argument("--rgg-radius-scale", type=float, default=4.0,
                        help="Scale factor mapping rewire rate to RGG radius (radius = scale * rate)")
    parser.add_argument("--rgg-radius-min", type=float, default=0.01,
                        help="Minimum RGG radius to avoid disconnected graphs")

    args = parser.parse_args()

    if args.no_refine:
        refine_bands = None
    elif args.refine_band:
        refine_bands = []
        for lo, hi, steps in args.refine_band:
            try:
                steps_int = int(float(steps))
            except ValueError:
                steps_int = 0
            refine_bands.append((float(lo), float(hi), steps_int))
    else:
        refine_bands = [tuple(band) for band in DEFAULT_REFINE_BANDS]

    if args.family == "rgg":
        family_kwargs = {
            "radius_scale": args.rgg_radius_scale,
            "radius_min": args.rgg_radius_min,
        }
    else:
        family_kwargs = None

    if args.preset:
        run_preset(
            args.preset,
            seed=args.seed,
            n_sigma_override=args.n_sigma,
            n_probes_override=args.n_probes,
            m_lanczos_override=args.m_lanczos,
            C_geo_steps_override=args.C_geo_steps,
            refine_bands=refine_bands,
            family=args.family, family_kwargs=family_kwargs,
            verbose=not args.quiet,
        )
    else:
        n_sigma = args.n_sigma if args.n_sigma is not None else 300
        run_custom(
            D=args.D, N=args.N, rates=args.rates, seed=args.seed,
            n_sigma=n_sigma, n_probes=args.n_probes,
            m_lanczos=args.m_lanczos, C_geo_steps=args.C_geo_steps,
            refine_bands=refine_bands, family=args.family, family_kwargs=family_kwargs,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
