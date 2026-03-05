"""Selection schema used by the verification engine."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass(frozen=True)
class SelectionResult:
    schema_version: str
    claim_id: str

    predicted: Dict[str, Any]
    confidence: float

    witness_path: Optional[str] = None
    substrate_id: Optional[str] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SelectionResult":
        return SelectionResult(
            schema_version=d["schema_version"],
            claim_id=d["claim_id"],
            predicted=dict(d["predicted"]),
            confidence=float(d["confidence"]),
            witness_path=d.get("witness_path"),
            substrate_id=d.get("substrate_id"),
            seed=d.get("seed"),
        )


def save_selection_result(path: Path, res: SelectionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selection_result(path: Path) -> SelectionResult:
    return SelectionResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
