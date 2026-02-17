#!/usr/bin/env python3
"""Post-process a capacity sweep to extract class-splitting diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    try:
        C_geo = np.asarray(data["C_geo_values"], dtype=float)
    except KeyError as exc:  # pragma: no cover
        raise SystemExit(f"Missing key in {path}: {exc}")

    ds_raw = data.get("ds_plateau")
    if ds_raw is None:
        raise SystemExit(
            "This artifact does not contain ds_plateau; rerun the sweep with the updated code."
        )
    ds_plateau = np.asarray(ds_raw, dtype=float)
    if ds_plateau.size == 0:
        raise SystemExit("ds_plateau array is empty; rerun sweep with updated artifacts.")
    return C_geo, ds_plateau


def _equivalence_count(ds_plateau: np.ndarray) -> np.ndarray:
    """Estimate N(C) as the number of distinct plateau levels resolved so far."""
    # Treat each additional +1 plateau as a new class; enforce monotonicity explicitly.
    running = np.maximum.accumulate(ds_plateau)
    counts = np.maximum(1, np.floor(running + 1e-9)).astype(float)
    return counts


def _split_rate(log_N: np.ndarray, log_C: np.ndarray) -> np.ndarray:
    """Compute d log N / d log ||C|| using central differences."""
    # Guard against duplicate log_C entries (happens when C starts at the same value).
    _, unique_idx = np.unique(log_C, return_index=True)
    if unique_idx.size < log_C.size:
        log_N = log_N[unique_idx]
        log_C = log_C[unique_idx]
    rate = np.gradient(log_N, log_C, edge_order=2)
    return rate


def _split_signature(counts: np.ndarray, idx: int) -> str:
    """Return a class-split signature like '2→4' for the idx-th sample."""
    curr = int(round(counts[idx]))
    prev = int(round(counts[idx - 1])) if idx > 0 else curr
    nxt = int(round(counts[idx + 1])) if idx + 1 < len(counts) else curr

    if curr > prev:
        return f"{prev}\u2192{curr}"
    if nxt > curr:
        return f"{curr}\u2192{nxt}"
    return f"{curr}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to ds_matrix.json emitted by a sweep")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path (default: alongside input as class_splitting.csv)",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    C_geo, ds_plateau = _load_arrays(input_path)
    counts = _equivalence_count(ds_plateau)
    log_C = np.log(np.clip(C_geo, 1e-9, None))
    log_N = np.log(np.clip(counts, 1e-9, None))
    split_rate = _split_rate(log_N, log_C)

    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name("class_splitting.csv")
    )

    rows: list[dict[str, float]] = []
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "C_geo",
            "ds_plateau",
            "N_equiv",
            "log_C",
            "log_N",
            "R_gamma",
        ])
        for i in range(len(C_geo)):
            rgamma = float(split_rate[min(i, len(split_rate) - 1)])
            row = {
                "index": i,
                "C_geo": float(C_geo[i]),
                "ds_plateau": float(ds_plateau[i]),
                "N_equiv": float(counts[i]),
                "log_C": float(log_C[i]),
                "log_N": float(log_N[i]),
                "R_gamma": rgamma,
            }
            rows.append(row)
            writer.writerow([
                f"{row['C_geo']:.8f}",
                f"{row['ds_plateau']:.6f}",
                f"{row['N_equiv']:.6f}",
                f"{row['log_C']:.6f}",
                f"{row['log_N']:.6f}",
                f"{rgamma:.6f}",
            ])

    # Human-readable summary
    idx_max = int(np.nanargmax(split_rate))
    print(f"Wrote class-splitting diagnostics to {output_path}")
    print(
        "Peak splitting rate at C_geo={:.4f}: R_gamma={:.3f}".format(
            C_geo[idx_max], split_rate[idx_max]
        )
    )

    # Capture the top two peaks for downstream plotting
    sorted_rows = sorted(rows, key=lambda r: r["R_gamma"], reverse=True)
    peak_rows = [r for r in sorted_rows if np.isfinite(r["R_gamma"])][:2]
    peaks_output = []
    for rank, row in enumerate(peak_rows, start=1):
        signature = _split_signature(counts, int(row["index"]))
        peaks_output.append({
            "rank": rank,
            "C_geo": row["C_geo"],
            "R_gamma": row["R_gamma"],
            "ds_plateau": row["ds_plateau"],
            "N_equiv": row["N_equiv"],
            "split_signature": signature,
        })
        print(
            f"Peak {rank}: C_geo={row['C_geo']:.4f}, R_gamma={row['R_gamma']:.3f}, split={signature}"
        )

    if peaks_output:
        peaks_path = output_path.with_name("class_splitting_peaks.json")
        with open(peaks_path, "w") as pf:
            json.dump(peaks_output, pf, indent=2)
        print(f"Saved peak summary to {peaks_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
