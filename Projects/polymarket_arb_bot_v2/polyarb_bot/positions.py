"""
Position tracking for swing trading capability.
Tracks open positions and monitors for early exit opportunities.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("polyarb")


@dataclass(slots=True)
class Position:
    """An open arbitrage position (both legs filled)."""
    position_id: str
    bundle_id: str
    mode: str  # paper | live

    # Leg A details
    token_a: str
    side_a: str  # YES or NO
    shares_a: float
    fill_price_a: float
    cost_a: float
    fee_a: float

    # Leg B details
    token_b: str
    side_b: str  # YES or NO
    shares_b: float
    fill_price_b: float
    cost_b: float
    fee_b: float

    # Position metrics
    total_cost: float
    total_fees: float
    entry_edge: float  # Expected profit at entry
    entry_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Current state
    current_bid_a: float = 0.0
    current_bid_b: float = 0.0
    unrealized_pnl: float = 0.0

    # Status
    status: str = "OPEN"  # OPEN, CLOSED_A, CLOSED_B, EXITED
    exit_time: str | None = None
    exit_profit: float = 0.0

    @property
    def net_pnl(self) -> float:
        """Realized PnL if exited, unrealized if open."""
        return self.exit_profit if self.status == "EXITED" else self.unrealized_pnl

    @property
    def total_shares(self) -> float:
        """Both legs same shares."""
        return self.shares_a

    @property
    def entry_sum(self) -> float:
        """Sum of fill prices at entry."""
        return self.fill_price_a + self.fill_price_b

    @property
    def current_sum(self) -> float:
        """Current sum of bid prices (estimated exit)."""
        return self.current_bid_a + self.current_bid_b

    @property
    def captured_edge_pct(self) -> float:
        """Percentage of original edge currently captured."""
        if self.entry_edge <= 0:
            return 0.0
        # Original sum = 1.0 - entry_edge
        # Current sum = bid_a + bid_b
        # Profit captured = 1.0 - current_sum - fees
        original_sum = 1.0 - self.entry_edge
        current_profit = 1.0 - self.current_sum
        return max(0.0, min(1.0, current_profit / self.entry_edge))

    @property
    def can_exit_profitably(self) -> bool:
        """Check if current bids allow profitable early exit."""
        # Can exit if sum of current bids > entry cost + fees
        min_exit_sum = (self.total_cost + self.total_fees) / self.total_shares
        return self.current_sum >= min_exit_sum

    def update_prices(self, bid_a: float, bid_b: float):
        """Update current prices and recalc unrealized PnL."""
        self.current_bid_a = bid_a
        self.current_bid_b = bid_b
        # Unrealized = current exit value - cost - fees
        exit_value = (bid_a + bid_b) * self.total_shares
        self.unrealized_pnl = exit_value - self.total_cost - self.total_fees

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "bundle_id": self.bundle_id,
            "mode": self.mode,
            "token_a": self.token_a,
            "side_a": self.side_a,
            "shares_a": self.shares_a,
            "fill_price_a": self.fill_price_a,
            "token_b": self.token_b,
            "side_b": self.side_b,
            "shares_b": self.shares_b,
            "fill_price_b": self.fill_price_b,
            "total_cost": self.total_cost,
            "entry_edge": self.entry_edge,
            "entry_time": self.entry_time,
            "status": self.status,
            "current_bid_a": self.current_bid_a,
            "current_bid_b": self.current_bid_b,
            "captured_edge_pct": round(self.captured_edge_pct, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
        }


class PositionTracker:
    """
    Tracks open positions across scans.
    Supports swing trading by monitoring for early exit triggers.
    """

    def __init__(self, storage_path: str = "./positions.json"):
        self.path = Path(storage_path).expanduser().resolve()
        self.positions: dict[str, Position] = {}  # position_id -> Position
        self._open_by_token: dict[str, str] = {}  # token_id -> position_id
        self._load()

    def _load(self) -> None:
        """Load positions from disk."""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            for pos_data in data.get("positions", []):
                pos = Position(**{k: v for k, v in pos_data.items()
                               if k in Position.__dataclass_fields__})
                self.positions[pos.position_id] = pos
                self._open_by_token[pos.token_a] = pos.position_id
                self._open_by_token[pos.token_b] = pos.position_id
            LOG.info("Loaded %s positions from %s", len(self.positions), self.path)
        except Exception as exc:
            LOG.warning("Failed to load positions: %s", exc)

    def _save(self) -> None:
        """Persist positions to disk."""
        try:
            data = {
                "updated": datetime.now(timezone.utc).isoformat(),
                "positions": [p.to_dict() for p in self.positions.values()]
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            LOG.warning("Failed to save positions: %s", exc)

    def add_position(self, position: Position) -> None:
        """Track a new open position."""
        self.positions[position.position_id] = position
        self._open_by_token[position.token_a] = position.position_id
        self._open_by_token[position.token_b] = position.position_id
        self._save()
        LOG.info("Added position %s: %s shares, edge=%.4f",
                position.position_id, position.total_shares, position.entry_edge)

    def get_open_positions(self) -> list[Position]:
        """Get all currently open positions."""
        return [p for p in self.positions.values() if p.status == "OPEN"]

    def get_position_by_token(self, token_id: str) -> Position | None:
        """Find open position containing this token."""
        pos_id = self._open_by_token.get(token_id)
        if pos_id:
            pos = self.positions.get(pos_id)
            if pos and pos.status == "OPEN":
                return pos
        return None

    def update_position_prices(self, position_id: str, bid_a: float, bid_b: float) -> Position | None:
        """Update prices and check if swing exit is triggered."""
        pos = self.positions.get(position_id)
        if not pos or pos.status != "OPEN":
            return None
        pos.update_prices(bid_a, bid_b)
        self._save()
        return pos

    def mark_exited(self, position_id: str, exit_profit: float) -> None:
        """Mark position as exited with realized profit."""
        pos = self.positions.get(position_id)
        if pos:
            pos.status = "EXITED"
            pos.exit_time = datetime.now(timezone.utc).isoformat()
            pos.exit_profit = exit_profit
            # Remove from token index
            self._open_by_token.pop(pos.token_a, None)
            self._open_by_token.pop(pos.token_b, None)
            self._save()
            LOG.info("Position %s exited with profit %.4f", position_id, exit_profit)

    def get_stats(self) -> dict[str, Any]:
        """Get current position stats."""
        open_pos = self.get_open_positions()
        total_cost = sum(p.total_cost for p in open_pos)
        total_unrealized = sum(p.unrealized_pnl for p in open_pos)
        return {
            "open_count": len(open_pos),
            "open_cost": round(total_cost, 4),
            "open_unrealized_pnl": round(total_unrealized, 4),
            "total_positions_tracked": len(self.positions),
        }
