"""Truth plugin for an observer poset infimum claim (example: W02).

This is a placeholder that encodes the theorem-level truth as boolean.
If your W02 truth depends on a specific construction variant, use substrate_id
(and possibly a frozen witness) to scope the truth label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schema import TruthLabel


SCHEMA_VERSION = "truth.v1"


def generate_truth(claim_id: str, root: Path, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> TruthLabel:
    return TruthLabel(
        schema_version=SCHEMA_VERSION,
        claim_id=claim_id,
        truth_type="boolean",
        expected={"should_pass": True},
        tolerance={},
        source="analytical_theorem_or_spec",
        confidence=1.0,
        substrate_id=substrate_id,
        seed=seed,
    )
