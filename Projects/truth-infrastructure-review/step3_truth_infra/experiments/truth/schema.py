"""Truth schema for Step 3 selection gates.

Design goals:
- Versioned, hashable, append-only truth records.
- Truth can be boolean (PASS/FAIL), scalar, or structured dict.
- Heavy artifacts (vectors, matrices) should be stored as files and referenced
  by path + sha256 rather than embedded.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Literal
import hashlib
import json


TruthType = Literal["boolean", "scalar", "dict"]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TruthArtifactRef:
    """Reference to a heavy artifact stored on disk."""

    relpath: str
    sha256: str


@dataclass(frozen=True)
class TruthLabel:
    """A single truth label for a claim id under a declared substrate/config."""

    schema_version: str
    claim_id: str
    truth_type: TruthType

    # Truth content
    expected: Any

    # Comparison tolerance
    # - For boolean: ignored
    # - For scalar: float
    # - For dict: {key: tol}
    tolerance: Any

    # Provenance
    source: str
    confidence: float

    # Optional metadata
    substrate_id: Optional[str] = None
    seed: Optional[int] = None
    artifacts: Optional[Dict[str, TruthArtifactRef]] = None

    # Integrity
    evidence_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.artifacts:
            d["artifacts"] = {
                k: asdict(v) for k, v in self.artifacts.items()
            }
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TruthLabel":
        artifacts = d.get("artifacts")
        if artifacts:
            artifacts = {k: TruthArtifactRef(**v) for k, v in artifacts.items()}
        return TruthLabel(
            schema_version=d["schema_version"],
            claim_id=d["claim_id"],
            truth_type=d["truth_type"],
            expected=d["expected"],
            tolerance=d["tolerance"],
            source=d["source"],
            confidence=float(d["confidence"]),
            substrate_id=d.get("substrate_id"),
            seed=d.get("seed"),
            artifacts=artifacts,
            evidence_sha256=d.get("evidence_sha256"),
        )


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def truth_label_digest(label: TruthLabel) -> str:
    """Deterministic digest of the label content."""
    return sha256_bytes(canonical_json(label.to_dict()))


def save_truth_label(path: Path, label: TruthLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_truth_label(path: Path) -> TruthLabel:
    return TruthLabel.from_dict(json.loads(path.read_text(encoding="utf-8")))
