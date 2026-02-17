#!/usr/bin/env python3
"""Post-process nonseparable rewire outputs: reconstruct class-splitting diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HEADER = [
    "run_id",
    "run_path",
    "summary_path",
    "class_splitting_path",
    "D",
    "N",
    "rewire_rate",
    "seed",
    "C_geo",
    "ds_plateau",
    "N_equiv",
    "log_C",
    "log_N",
    "R_gamma",
]


def _repo_relative(path: Path) -> str:
    """Return a repo-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _parse_seed_from_name(name: str) -> str | None:
    for part in name.split("_"):
        if part.startswith("s") and len(part) > 1:
            suffix = part[1:]
            if suffix.replace(".", "", 1).isdigit():
                return suffix
            return suffix
    return None


def _load_seed(run_dir: Path) -> str | None:
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for key in ("seed",):
                if key in meta:
                    return str(meta[key])
            config = meta.get("config")
            if isinstance(config, dict):
                seed = config.get("seed")
                if seed is not None:
                    return str(seed)
        except (json.JSONDecodeError, OSError):  # pragma: no cover - best effort only
            pass
    return _parse_seed_from_name(run_dir.name)


def _update_index(
    index_path: Path,
    run_dir: Path,
    summary_path: Path,
    csv_path: Path,
    summary_data: dict,
    rows: list[dict[str, str]],
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    run_path = _repo_relative(run_dir)
    summary_rel = _repo_relative(summary_path)
    csv_rel = _repo_relative(csv_path)

    D = summary_data.get("D")
    N = summary_data.get("N")
    rewire_rate = summary_data.get("rewire_rate")
    seed = _load_seed(run_dir)

    new_rows = []
    for row in rows:
        new_rows.append({
            "run_id": run_id,
            "run_path": run_path,
            "summary_path": summary_rel,
            "class_splitting_path": csv_rel,
            "D": str(D) if D is not None else "",
            "N": str(N) if N is not None else "",
            "rewire_rate": str(rewire_rate) if rewire_rate is not None else "",
            "seed": seed or "",
            "C_geo": row["C_geo"],
            "ds_plateau": row["ds_plateau"],
            "N_equiv": row["N_equiv"],
            "log_C": row["log_C"],
            "log_N": row["log_N"],
            "R_gamma": row["R_gamma"],
        })

    existing_rows: list[dict[str, str]] = []
    if index_path.exists():
        with open(index_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if existing.get("run_id") != run_id:
                    existing_rows.append(existing)

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_HEADER)
        writer.writeheader()
        writer.writerows(existing_rows + new_rows)



def _load_summary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)

    config = data.get("config")
    if config is None:
        raise SystemExit("Summary JSON missing embedded result/config data")

    result = config.get("result") if isinstance(config, dict) else data
    if "result" in data:
        result = data["result"]

    C_geo = np.asarray(result["C_geo_values"], dtype=float)
    ds_plateau = np.asarray(result["ds_plateau"], dtype=float)
    return C_geo, ds_plateau







def _boundary_records(
    C_geo: np.ndarray,
    counts: np.ndarray,
    split_rate: np.ndarray,
) -> list[dict]:
    boundaries: list[dict] = []
    last_index = len(split_rate) - 1
    for i in range(len(counts) - 1):
        delta = counts[i + 1] - counts[i]
        if delta <= 0:
            continue
        c_lo = float(C_geo[i])
        c_hi = float(C_geo[i + 1])
        c_mid = 0.5 * (c_lo + c_hi)
        from_n = int(round(counts[i]))
        to_n = int(round(counts[i + 1]))
        idx_candidates = [i]
        if i + 1 <= last_index:
            idx_candidates.append(i + 1)
        best_idx = max(idx_candidates, key=lambda idx: split_rate[idx])
        boundaries.append({
            "index_lo": i,
            "index_hi": i + 1,
            "C_lo": c_lo,
            "C_hi": c_hi,
            "C_mid": c_mid,
            "from_N": from_n,
            "to_N": to_n,
            "delta_N": float(delta),
            "peak_index": int(best_idx),
            "peak_C": float(C_geo[best_idx]),
            "peak_R_gamma": float(split_rate[best_idx]),
        })
    boundaries.sort(key=lambda b: b["C_mid"])
    for rank, boundary in enumerate(boundaries, start=1):
        boundary["rank"] = rank
    return boundaries


def _format_signature(boundary: dict | None) -> str:
    if not boundary:
        return ""
    return "{}→{} [{:.4f},{:.4f}]".format(
        boundary["from_N"],
        boundary["to_N"],
        boundary["C_lo"],
        boundary["C_hi"],
    )



def _class_splitting(C_geo: np.ndarray, ds_plateau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    running = np.maximum.accumulate(ds_plateau)
    counts = np.maximum(1, np.floor(running + 1e-9)).astype(float)
    log_C = np.log(np.clip(C_geo, 1e-9, None))
    log_N = np.log(np.clip(counts, 1e-9, None))
    _, unique_idx = np.unique(log_C, return_index=True)
    log_C = log_C[unique_idx]
    log_N = log_N[unique_idx]
    counts = counts[unique_idx]
    split_rate = np.gradient(log_N, log_C, edge_order=2)
    return counts, log_C, log_N, split_rate


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to JSON summary with embedded result data")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path (default: class_splitting.csv next to summary)")
    parser.add_argument("--index", type=str, default=None, help="Optional global index CSV path (default: outputs/nonseparable_rewire/class_splitting_index.csv)")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}")

    run_dir = summary_path.parent
    index_path = Path(args.index).expanduser() if args.index else None
    if index_path is None and run_dir.parent != run_dir:
        index_path = run_dir.parent / "class_splitting_index.csv"

    with open(summary_path, "r") as f:
        data = json.load(f)

    result = data.get("result")
    if result is None:
        raise SystemExit("Summary file lacks 'result' section (rerun nonseparable test with latest code)")

    C_geo = np.asarray(result["C_geo_values"], dtype=float)
    ds_plateau = np.asarray(result["ds_plateau"], dtype=float)

    counts, log_C, log_N, split_rate = _class_splitting(C_geo, ds_plateau)
    boundaries = _boundary_records(C_geo, counts, split_rate)

    output_path = (
        Path(args.output) if args.output else summary_path.with_name("class_splitting.csv")
    )

    rows: list[dict[str, str]] = []
    for i in range(len(log_C)):
        rows.append({
            "C_geo": f"{C_geo[i]:.8f}",
            "ds_plateau": f"{ds_plateau[i]:.6f}",
            "N_equiv": f"{counts[i]:.6f}",
            "log_C": f"{log_C[i]:.6f}",
            "log_N": f"{log_N[i]:.6f}",
            "R_gamma": f"{split_rate[min(i, len(split_rate)-1)]:.6f}",
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["C_geo", "ds_plateau", "N_equiv", "log_C", "log_N", "R_gamma"])
        for row in rows:
            writer.writerow([
                row["C_geo"],
                row["ds_plateau"],
                row["N_equiv"],
                row["log_C"],
                row["log_N"],
                row["R_gamma"],
            ])

    if index_path is not None:
        _update_index(index_path, run_dir, summary_path, output_path, data, rows)

    boundary_records = []
    for boundary in boundaries:
        record = dict(boundary)
        record["signature"] = _format_signature(boundary)
        boundary_records.append(record)

    boundaries_path = output_path.with_name("split_boundaries.json")
    with open(boundaries_path, "w") as f:
        json.dump({
            "run_id": run_dir.name,
            "boundaries": boundary_records,
        }, f, indent=2)

    print(f"Wrote class-splitting diagnostics to {output_path}")
    if boundary_records:
        summary_parts = []
        for boundary in boundary_records:
            summary_parts.append(
                f"rank {boundary['rank']}: C_mid={boundary['C_mid']:.4f} ΔN={boundary['delta_N']:.1f} peakRγ={boundary['peak_R_gamma']:.3f} {boundary['signature']}"
            )
        print("Boundaries -> " + "; ".join(summary_parts))
    else:
        print("Boundaries -> none detected")


if __name__ == "__main__":
    main()
