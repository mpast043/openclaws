#!/usr/bin/env python3
"""Truth label generator for Step 3 selection gates.

This module supports two modes:
1) "generate": produce (or refresh) truth labels into experiments/truth/truth_labels.
2) "validate": verify stored truth labels and referenced artifact hashes.

Important separation:
- Truth generation should be explicit and reproducible.
- Verification should never silently generate missing truth.

Recommended workflow:
- Commit truth_labels/*.json plus any referenced artifacts under ed_reference/.
- Treat the committed truth set as append-only and versioned.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json

from .schema import TruthLabel, TruthArtifactRef, save_truth_label, load_truth_label, sha256_file


SCHEMA_VERSION = "truth.v1"


def repo_root_from_this_file() -> Path:
    # Assumes this file lives at experiments/truth/truth_generator.py
    return Path(__file__).resolve().parents[2]


def truth_dir(root: Path) -> Path:
    return root / "experiments" / "truth"


def labels_dir(root: Path) -> Path:
    return truth_dir(root) / "truth_labels"


def artifacts_dir(root: Path) -> Path:
    return truth_dir(root) / "ed_reference"


def parse_claim_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Claim list not found: {path}")
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if path.suffix.lower() in {".json"}:
        return list(json.loads(txt))
    # default: one per line
    return [line.strip() for line in txt.splitlines() if line.strip() and not line.strip().startswith("#")]


class TruthLabelGenerator:
    """Plugin-based truth label generator.

    Plugins live under experiments/truth/plugins and expose:
        generate_truth(claim_id: str, root: Path, **kwargs) -> TruthLabel

    If a claim has no plugin, you can still supply a manual truth label JSON
    (committed under truth_labels/) and skip generation.
    """

    def __init__(self, root: Path):
        self.root = root

    def generate(self, claim_id: str, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> TruthLabel:
        # Map claim families to plugin modules. Adjust as your taxonomy stabilizes.
        # Suggested convention:
        #   Wxx = observer/workflow claims
        #   Pxx = physics claims
        #   Vxx = validation gate claims
        if claim_id.startswith("P01"):
            from .plugins.spectral_dimension import generate_truth
            return generate_truth(claim_id, self.root, substrate_id=substrate_id, seed=seed)

        if claim_id.startswith("P02"):
            from .plugins.ed_reference import generate_truth
            return generate_truth(claim_id, self.root, substrate_id=substrate_id, seed=seed)

        if claim_id.startswith("W02"):
            from .plugins.observer_poset import generate_truth
            return generate_truth(claim_id, self.root, substrate_id=substrate_id, seed=seed)

        # Default: require manual truth label file
        raise KeyError(
            f"No truth plugin registered for {claim_id}. "
            f"Provide a manual truth label JSON under {labels_dir(self.root)} or add a plugin."
        )


def cmd_generate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve() if args.root else repo_root_from_this_file()
    out_dir = labels_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    claim_ids = parse_claim_list(Path(args.claims)) if args.claims else args.claim_id
    if isinstance(claim_ids, str):
        claim_ids = [claim_ids]

    gen = TruthLabelGenerator(root)

    for cid in claim_ids:
        label = gen.generate(cid, substrate_id=args.substrate_id, seed=args.seed)
        # enforce schema version
        if label.schema_version != SCHEMA_VERSION:
            label = TruthLabel(
                schema_version=SCHEMA_VERSION,
                claim_id=label.claim_id,
                truth_type=label.truth_type,
                expected=label.expected,
                tolerance=label.tolerance,
                source=label.source,
                confidence=label.confidence,
                substrate_id=label.substrate_id,
                seed=label.seed,
                artifacts=label.artifacts,
                evidence_sha256=label.evidence_sha256,
            )

        save_truth_label(out_dir / f"{cid}.json", label)
        print(f"Wrote {cid}.json")


def cmd_validate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve() if args.root else repo_root_from_this_file()
    tdir = labels_dir(root)
    if not tdir.exists():
        raise FileNotFoundError(f"truth_labels dir missing: {tdir}")

    ok = True
    for path in sorted(tdir.glob("*.json")):
        label = load_truth_label(path)
        if label.schema_version != SCHEMA_VERSION:
            print(f"FAIL schema_version {path.name}: {label.schema_version} != {SCHEMA_VERSION}")
            ok = False

        if label.artifacts:
            for name, ref in label.artifacts.items():
                apath = artifacts_dir(root) / ref.relpath
                if not apath.exists():
                    print(f"FAIL missing artifact for {path.name}: {name} -> {apath}")
                    ok = False
                    continue
                got = sha256_file(apath)
                if got != ref.sha256:
                    print(f"FAIL artifact hash mismatch for {path.name}: {name}")
                    print(f"  expected {ref.sha256}")
                    print(f"  got      {got}")
                    ok = False

    if not ok:
        raise SystemExit(2)

    print("Truth labels validated")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="Repo root (defaults to inferred)")

    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--claim-id", default=None, help="Single claim id")
    g.add_argument("--claims", default=None, help="Path to claim list (txt or json)")
    g.add_argument("--substrate-id", default=None)
    g.add_argument("--seed", type=int, default=None)

    v = sub.add_parser("validate")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "generate":
        if not args.claim_id and not args.claims:
            raise SystemExit("Provide --claim-id or --claims")
        cmd_generate(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
