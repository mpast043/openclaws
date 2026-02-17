#!/usr/bin/env python3
"""Aggregate boundary diagnostics across nonseparable runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _parse_seed(run_id: str) -> str:
    for part in run_id.split("_"):
        if part.startswith("s") and len(part) > 1:
            return part[1:]
    return ""


def _load_boundaries(run_dir: Path) -> list[dict[str, Any]]:
    boundaries_path = run_dir / "split_boundaries.json"
    if not boundaries_path.exists():
        return []
    with open(boundaries_path, "r") as f:
        data = json.load(f)
    return data.get("boundaries", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="outputs/nonseparable_rewire",
        help="Directory containing run subfolders",
    )
    parser.add_argument(
        "--table",
        default="boundary_table.csv",
        help="Filename for per-run boundary table (saved under root)",
    )
    parser.add_argument(
        "--stats",
        default="boundary_stats_by_N_r.csv",
        help="Filename for grouped stats (saved under root)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    table_rows: list[dict[str, Any]] = []
    grouped: defaultdict[tuple[int, float, int], list[dict[str, Any]]] = defaultdict(list)
    runs_by_rate: defaultdict[tuple[int, float], list[list[dict[str, Any]]]] = defaultdict(list)

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r") as f:
            summary = json.load(f)
        result = summary.get("result")
        if result is None:
            continue
        rewire_rate = summary.get("rewire_rate")
        N = summary.get("N")
        run_id = run_dir.name
        seed = _parse_seed(run_id)
        boundaries = _load_boundaries(run_dir)
        for boundary in boundaries:
            row = {
                "run_id": run_id,
                "rewire_rate": rewire_rate,
                "seed": seed,
                "N": N,
                "rank": boundary.get("rank"),
                "from_N": boundary.get("from_N"),
                "to_N": boundary.get("to_N"),
                "delta_N": boundary.get("delta_N"),
                "C_lo": boundary.get("C_lo"),
                "C_hi": boundary.get("C_hi"),
                "C_mid": boundary.get("C_mid"),
                "peak_C": boundary.get("peak_C"),
                "peak_R_gamma": boundary.get("peak_R_gamma"),
                "signature": boundary.get("signature", ""),
            }
            table_rows.append(row)
            key = (N, int(round(rewire_rate * 1000)) if isinstance(rewire_rate, (int, float)) else rewire_rate, boundary.get("rank", 0))
            grouped[key].append(row)
        runs_by_rate[(N, rewire_rate)].append(boundaries)

    table_path = root / args.table
    header = [
        "run_id",
        "rewire_rate",
        "seed",
        "N",
        "rank",
        "from_N",
        "to_N",
        "delta_N",
        "C_lo",
        "C_hi",
        "C_mid",
        "peak_C",
        "peak_R_gamma",
        "signature",
    ]
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    stats_path = root / args.stats
    stats_header = [
        "N",
        "rewire_rate",
        "rank",
        "n_runs",
        "C_mid_mean",
        "C_mid_std",
        "delta_N_mean",
        "delta_N_std",
        "peak_R_gamma_mean",
        "peak_R_gamma_std",
    ]
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_header)
        writer.writeheader()
        for key in sorted(grouped.keys()):
            rows = grouped[key]
            N, rate_key, rank = key
            rate = rate_key / 1000.0 if isinstance(rate_key, int) else rate_key
            c_mids = [row["C_mid"] for row in rows if row["C_mid"] is not None]
            deltas = [row["delta_N"] for row in rows if row["delta_N"] is not None]
            peaks = [row["peak_R_gamma"] for row in rows if row["peak_R_gamma"] is not None]
            def _mean(values):
                return mean(values) if values else None
            def _std(values):
                return pstdev(values) if len(values) > 1 else 0.0 if values else None
            writer.writerow({
                "N": N,
                "rewire_rate": rate,
                "rank": rank,
                "n_runs": len(rows),
                "C_mid_mean": _mean(c_mids),
                "C_mid_std": _std(c_mids),
                "delta_N_mean": _mean(deltas),
                "delta_N_std": _std(deltas),
                "peak_R_gamma_mean": _mean(peaks),
                "peak_R_gamma_std": _std(peaks),
            })

    print(f"Wrote {table_path}")
    print(f"Wrote {stats_path}")

    presence_path = root / "boundary_presence_by_rate.csv"
    presence_header = [
        "N",
        "rewire_rate",
        "n_runs",
        "mean_boundaries",
        "frac_rank1",
        "frac_rank2",
        "frac_rank3",
        "rank1_delta_mean",
        "rank2_delta_mean",
        "rank3_delta_mean",
    ]
    with open(presence_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=presence_header)
        writer.writeheader()
        for key in sorted(runs_by_rate.keys()):
            N, rate = key
            boundary_lists = runs_by_rate[key]
            n_runs = len(boundary_lists)
            mean_boundaries = mean(len(b) for b in boundary_lists) if boundary_lists else 0.0
            def _frac_and_mean(rank_idx: int):
                rows = [b[rank_idx] for b in boundary_lists if len(b) > rank_idx]
                frac = len(rows) / n_runs if n_runs else 0.0
                delta_vals = [row.get("delta_N") for row in rows if row.get("delta_N") is not None]
                delta_mean = mean(delta_vals) if delta_vals else None
                return frac, delta_mean
            frac1, delta1 = _frac_and_mean(0)
            frac2, delta2 = _frac_and_mean(1)
            frac3, delta3 = _frac_and_mean(2)
            writer.writerow({
                "N": N,
                "rewire_rate": rate,
                "n_runs": n_runs,
                "mean_boundaries": mean_boundaries,
                "frac_rank1": frac1,
                "frac_rank2": frac2,
                "frac_rank3": frac3,
                "rank1_delta_mean": delta1,
                "rank2_delta_mean": delta2,
                "rank3_delta_mean": delta3,
            })
    print(f"Wrote {presence_path}")

if __name__ == "__main__":
    main()
