#!/usr/bin/env python3
"""Run Step 3 selection gates using truth infrastructure.

Inputs:
- claim list (txt/json)
- selection results directory (must contain selection_results/<claim_id>.json)
- repo root (optional)

Outputs:
- ledger.jsonl
- step3_metrics.json
- step3_verdict.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from ..truth.truth_generator import parse_claim_list, repo_root_from_this_file
from ..selection.selection_engine import FilesystemSelectionEngine, run_all
from .verification_engine import load_truth_set, VerificationEngine, step3_gate_metrics, step3_gate_pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="Repo root (defaults to inferred)")
    p.add_argument("--claims", required=True, help="Path to claim list (txt or json)")
    p.add_argument("--selection-dir", required=True, help="Directory containing selection_results/")
    p.add_argument("--out", required=True, help="Output directory (e.g., <run_dir>/results/selection)")
    p.add_argument("--truth-threshold", type=float, default=0.95)
    p.add_argument("--high-conf-threshold", type=float, default=0.90)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve() if args.root else repo_root_from_this_file()

    claim_ids = parse_claim_list(Path(args.claims))
    if not claim_ids:
        raise SystemExit("No claims found")

    truth = load_truth_set(root, claim_ids)

    sel_engine = FilesystemSelectionEngine(Path(args.selection_dir).resolve())
    selections = run_all(sel_engine, claim_ids)

    verifier = VerificationEngine(high_conf_threshold=args.high_conf_threshold)
    ledger = verifier.verify(truth, selections)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = out_dir / "ledger.jsonl"
    ledger.save_jsonl(ledger_path)

    metrics = step3_gate_metrics(ledger, claim_ids, high_conf_threshold=args.high_conf_threshold)
    metrics_path = out_dir / "step3_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    passed, reasons = step3_gate_pass(
        metrics,
        truth_threshold=args.truth_threshold,
        high_conf_threshold=args.high_conf_threshold,
    )

    verdict = {
        "step3_selection": "PASS" if passed else "FAIL",
        "reasons": reasons,
        "params": {
            "truth_threshold": args.truth_threshold,
            "high_conf_threshold": args.high_conf_threshold,
        },
    }

    verdict_path = out_dir / "step3_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
