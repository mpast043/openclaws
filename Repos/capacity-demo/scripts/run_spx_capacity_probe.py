#!/usr/bin/env python3
"""
Capacity-only sweep on SPX correlation Laplacian eigenvalues.

Reads eigenvalue dumps from Market Program Work (JSON), applies a
capacity-only filter (hard cutoff by eigenvalue rank), and runs the
Framework spectral pipeline (ln P, d_s(sigma), plateau extraction) to
check for staircase behaviour as capacity increases.

Usage:
    python scripts/run_spx_capacity_probe.py \
        --eigs-json outputs/spx_capacity_probe/spx_laplacian_eigenvalues.json \
        --filter hard --C-steps 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimshift.sweep import SweepConfig, _detect_thresholds
from dimshift.framework_spectral import (
    filtered_spectral_dimension,
    extract_plateau,
)
from dimshift.spectral_filters import (
    HardCutoffFilter,
    SoftCutoffFilter,
    PowerLawFilter,
    SpectralFilter,
)

FILTERS = {
    "hard": HardCutoffFilter,
    "soft": SoftCutoffFilter,
    "powerlaw": PowerLawFilter,
}

SOURCE_INFO = {
    "dataset": "SPX",
    "price_file": str(REPO_ROOT / "Market Program Work" / "data" / "sp500_prices_lcc.csv"),
}


def build_filter(name: str, **kwargs) -> SpectralFilter:
    if name not in FILTERS:
        raise ValueError(f"Unknown filter '{name}'. Options: {list(FILTERS)}")
    return FILTERS[name](**kwargs)


def run_capacity_probe(
    eigenvalues: np.ndarray,
    config: SweepConfig,
    spectral_filter: SpectralFilter,
    D_targets: int = 4,
) -> Dict:
    """Run capacity sweep on a single eigenvalue set."""
    sigma_values = config.sigma_grid()
    sigma_lo, sigma_hi = config.plateau_window()
    C_geo_values = config.C_geo_grid()
    ln_sigma = np.log(sigma_values)

    ds_matrix = np.zeros((len(C_geo_values), len(sigma_values)))
    ds_plateau = np.zeros(len(C_geo_values))

    for idx, C in enumerate(C_geo_values):
        filt = spectral_filter.apply(eigenvalues, float(C))
        ds_curve, _ = filtered_spectral_dimension(
            eigenvalues, filt.weights, sigma_values
        )
        ds_matrix[idx] = ds_curve
        plateau_stats = extract_plateau(ds_curve, sigma_values, sigma_lo, sigma_hi)
        ds_plateau[idx] = plateau_stats["ds_plateau"]

    thresholds = _detect_thresholds(C_geo_values, ds_plateau, D=D_targets)

    return {
        "C_geo_values": C_geo_values.tolist(),
        "sigma_values": sigma_values.tolist(),
        "ds_matrix": ds_matrix.tolist(),
        "ds_plateau": ds_plateau.tolist(),
        "thresholds": thresholds,
        "sigma_window": [sigma_lo, sigma_hi],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run capacity-only sweeps on SPX Laplacian eigenvalues"
    )
    parser.add_argument(
        "--eigs-json",
        type=str,
        default="outputs/spx_capacity_probe/spx_laplacian_eigenvalues.json",
        help="Path to JSON file with eigenvalues keyed by date",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=list(FILTERS.keys()),
        default="hard",
        help="Spectral filter to use for capacity weighting",
    )
    parser.add_argument(
        "--floor-fraction",
        type=float,
        default=0.05,
        help="Floor fraction for hard cutoff filter",
    )
    parser.add_argument(
        "--steepness-frac",
        type=float,
        default=0.05,
        help="Steepness fraction for soft cutoff filter",
    )
    parser.add_argument(
        "--base-fraction",
        type=float,
        default=0.05,
        help="Base fraction for soft cutoff filter",
    )
    parser.add_argument(
        "--power-d0",
        type=float,
        default=3.0,
        help="d0 prior for power-law filter",
    )
    parser.add_argument(
        "--power-gamma",
        type=float,
        default=1.0,
        help="Gamma exponent for power-law filter",
    )
    parser.add_argument(
        "--C-steps",
        type=int,
        default=30,
        help="Number of capacity steps (linspace in [0.05, 1])",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=200.0,
        help="Max sigma for the log-spaced grid",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/spx_capacity_probe/results",
        help="Directory to store per-date results",
    )
    args = parser.parse_args()

    eigs_path = Path(args.eigs_json)
    if not eigs_path.exists():
        raise FileNotFoundError(f"Eigenvalue file not found: {eigs_path}")

    with open(eigs_path) as f:
        eigenvalue_dict = json.load(f)

    # Build config (reuse SweepConfig for sigma grid / plateau window)
    config = SweepConfig(
        C_geo_steps=args.C_steps,
        sigma_max=args.sigma_max,
    )

    # Build filter
    if args.filter == "hard":
        spectral_filter = build_filter("hard", floor_fraction=args.floor_fraction)
    elif args.filter == "soft":
        spectral_filter = build_filter(
            "soft",
            steepness_frac=args.steepness_frac,
            base_fraction=args.base_fraction,
        )
    else:
        spectral_filter = build_filter(
            "powerlaw",
            d0=args.power_d0,
            gamma=args.power_gamma,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[str] = []

    for date_str, payload in eigenvalue_dict.items():
        eigenvalues = np.array(payload["eigenvalues"], dtype=np.float64)
        result = run_capacity_probe(eigenvalues, config, spectral_filter)

        out_path = out_dir / f"spx_capacity_{date_str}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "date": date_str,
                    "n_vertices": payload.get("n_vertices"),
                    "alpha": payload.get("alpha"),
                    "window": payload.get("window"),
                    "use_abs": payload.get("use_abs"),
                    "filter": spectral_filter.describe(),
                    "config": config.to_dict(),
                    "source": {
                        **SOURCE_INFO,
                        "eigenvalue_file": str(eigs_path),
                    },
                    "result": result,
                },
                f,
                indent=2,
            )

        thresholds = result["thresholds"]
        plateau = result["ds_plateau"]
        if thresholds:
            thresh_summary = ", ".join(
                [
                    f"d_s={t['target_dimension']:.1f} at C={t['C_geo_threshold']:.3f}"
                    for t in thresholds
                ]
            )
        else:
            thresh_summary = "no crossings"
        summaries.append(
            f"{date_str}: ds_plateau range [{min(plateau):.2f}, {max(plateau):.2f}] — {thresh_summary}"
        )

    report_path = out_dir / "summary.txt"
    header = (
        "Capacity probe summaries "
        f"(source={SOURCE_INFO['dataset']}, price_file={SOURCE_INFO['price_file']}):\n"
    )
    with open(report_path, "w") as f:
        f.write(header)
        for line in summaries:
            f.write(line + "\n")

    print("\n".join(summaries))
    print(f"Written detailed results to {out_dir}")


if __name__ == "__main__":
    main()
