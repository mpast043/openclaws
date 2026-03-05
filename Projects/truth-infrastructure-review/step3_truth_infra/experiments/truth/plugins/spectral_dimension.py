"""Truth plugin for a spectral-dimension claim (example: P01).

This is intentionally conservative: it encodes a reference expected d_s
for known substrates.

You should extend substrate_id parsing to match your repo's naming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schema import TruthLabel


SCHEMA_VERSION = "truth.v1"


def generate_truth(claim_id: str, root: Path, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> TruthLabel:
    sid = (substrate_id or "").lower()

    # Default reference: Sierpinski gasket spectral dimension
    # d_s = 2 * log(3) / log(5) ≈ 1.3652
    expected_ds = 1.3652
    tol = 0.15
    source = "analytical"

    # If you're targeting a D-dimensional periodic lattice, you can set expected_ds = D.
    # Example substrate_id conventions you might adopt:
    #   lattice_d2_n64
    #   lattice_d3_n64
    if "lattice_d1" in sid:
        expected_ds = 1.0
        tol = 0.15
    elif "lattice_d2" in sid:
        expected_ds = 2.0
        tol = 0.20
    elif "lattice_d3" in sid:
        expected_ds = 3.0
        tol = 0.25

    return TruthLabel(
        schema_version=SCHEMA_VERSION,
        claim_id=claim_id,
        truth_type="scalar",
        expected={"d_s": expected_ds},
        tolerance={"d_s": tol},
        source=source,
        confidence=1.0,
        substrate_id=substrate_id,
        seed=seed,
    )
