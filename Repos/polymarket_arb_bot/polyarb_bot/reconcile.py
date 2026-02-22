from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReconcileSummary:
    rows: int
    by_status: dict[str, int]
    expected_locked_pnl: float
    gross_cost: float
    fees: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "by_status": dict(self.by_status),
            "expected_locked_pnl": round(self.expected_locked_pnl, 4),
            "gross_cost": round(self.gross_cost, 4),
            "fees": round(self.fees, 4),
        }


def summarize_trade_log(csv_path: str) -> ReconcileSummary:
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        return ReconcileSummary(rows=0, by_status={}, expected_locked_pnl=0.0, gross_cost=0.0, fees=0.0)

    rows = 0
    by_status: dict[str, int] = defaultdict(int)
    pnl = 0.0
    gross = 0.0
    fees = 0.0

    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            status = (row.get("status") or "").strip() or "UNKNOWN"
            by_status[status] += 1
            try:
                pnl += float(row.get("est_profit") or 0.0)
            except Exception:
                pass
            try:
                gross += float(row.get("gross_cost") or 0.0)
            except Exception:
                pass
            try:
                fees += float(row.get("fees") or 0.0)
            except Exception:
                pass

    return ReconcileSummary(rows=rows, by_status=by_status, expected_locked_pnl=pnl, gross_cost=gross, fees=fees)
