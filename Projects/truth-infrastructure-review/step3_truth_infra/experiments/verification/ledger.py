"""Verification ledger for Step 3."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class LedgerEntry:
    claim_id: str
    status: str  # VERIFIED | FALSIFIED | INCONCLUSIVE | MISSING
    match: bool

    predicted: Optional[Dict[str, Any]] = None
    truth: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

    danger_gap: bool = False
    witness: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VerificationLedger:
    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []

    def add(self, e: LedgerEntry) -> None:
        self.entries.append(e)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")

    def summary(self) -> Dict[str, Any]:
        n = len(self.entries)
        verified = sum(1 for e in self.entries if e.status == "VERIFIED")
        falsified = sum(1 for e in self.entries if e.status == "FALSIFIED")
        inconc = sum(1 for e in self.entries if e.status == "INCONCLUSIVE")
        missing = sum(1 for e in self.entries if e.status == "MISSING")
        danger = sum(1 for e in self.entries if e.danger_gap)

        return {
            "n": n,
            "verified": verified,
            "falsified": falsified,
            "inconclusive": inconc,
            "missing": missing,
            "danger_gaps": danger,
            "accuracy_over_present": (verified / max(1, (n - missing))),
        }

    def save_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
