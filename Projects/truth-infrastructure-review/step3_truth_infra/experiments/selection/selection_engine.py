"""Selection engines.

Step 3 expects selections in a standard format so verification can be uniform.

If you already have a selection workflow that writes claim verdict JSON, the
fastest path is to add a thin adapter that emits SelectionResult objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from .schema import SelectionResult, load_selection_result


SCHEMA_VERSION = "selection.v1"


class SelectionEngine(ABC):
    @abstractmethod
    def run_selection(self, claim_id: str, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> SelectionResult:
        raise NotImplementedError


class FilesystemSelectionEngine(SelectionEngine):
    """Loads selections produced elsewhere.

    Expected layout (configurable):
      <base_dir>/selection_results/<claim_id>.json

    Each file must match experiments/selection/schema.py.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def run_selection(self, claim_id: str, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> SelectionResult:
        path = self.base_dir / "selection_results" / f"{claim_id}.json"
        return load_selection_result(path)


def run_all(engine: SelectionEngine, claim_ids: list[str], substrate_id: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, SelectionResult]:
    out: Dict[str, SelectionResult] = {}
    for cid in claim_ids:
        out[cid] = engine.run_selection(cid, substrate_id=substrate_id, seed=seed)
    return out
