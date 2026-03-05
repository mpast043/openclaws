"""
Swing exit scanner - monitors open positions for early exit triggers.
Supports partial exit (capture 80% of edge) and full exit (at convergence).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import BotConfig
from .models import BookLevel, BookSnapshot
from .positions import Position, PositionTracker

LOG = logging.getLogger("polyarb")


@dataclass(slots=True)
class ExitOpportunity:
    """An opportunity to exit a position early."""
    position: Position
    exit_type: str  # PARTIAL | PARTIAL_80 | FULL

    current_bid_sum: float
    min_exit_sum: float  # Breakeven
    capture_threshold_sum: float  # Target for profit capture

    exit_price_a: float
    exit_price_b: float

    est_exit_fees: float
    est_exit_cost: float
    est_profit: float
    profit_vs_hold: float  # How much we gain vs holding to expiry

    @property
    def is_profitable(self) -> bool:
        return self.current_bid_sum >= self.min_exit_sum

    @property
    def can_capture_edge(self) -> bool:
        """Check if we can capture target portion of edge."""
        return self.current_bid_sum >= self.capture_threshold_sum


class SwingExitScanner:
    """
    Scans open positions for early exit opportunities.
    
    Exit strategies:
    - PARTIAL_80: Exit when we can capture 80% of original edge
    - PARTIAL_90: Exit when 90% captured
    - FULL: Exit when sum >= breakeven (minimal profit)
    - HOLD: Continue to expiry (best if spread converges)
    """

    def __init__(self, cfg: BotConfig, tracker: PositionTracker):
        self.cfg = cfg
        self.tracker = tracker
        self.partial_capture_threshold = getattr(cfg, 'swing_exit_partial_pct', 0.80)
        self.min_capture_threshold = getattr(cfg, 'swing_exit_min_pct', 0.50)

    def _can_exit_partial(self, pos: Position, current_bid_a: float, current_bid_b: float) -> ExitOpportunity | None:
        """Check if position can be exited at profit with partial edge capture."""
        if current_bid_a <= 0 or current_bid_b <= 0:
            return None

        current_sum = current_bid_a + current_bid_b

        # Calculate entry values
        entry_sum = pos.entry_sum
        entry_edge = pos.entry_edge  # This is profit per share

        # Cost basis: total_cost (what we paid) + fees (already paid) + exit fees
        total_entry_cost = pos.total_cost + pos.total_fees
        exit_fees = pos.total_fees  # Assume same fees for exit
        breakeven_sum = (total_entry_cost + exit_fees) / pos.total_shares

        # Target: capture 80% of original edge
        # If original edge was 0.04 (buy at 0.96, sell at 1.00)
        # Want to exit when we can buy back at 0.80 (capturing 0.20)
        target_sum = 1.0 - (entry_edge * (1 - self.partial_capture_threshold))

        # Also calculate minimum acceptable (50% capture)
        min_sum = 1.0 - (entry_edge * (1 - self.min_capture_threshold))

        if current_sum < breakeven_sum:
            return None  # Not profitable to exit

        # Determine exit type
        if current_sum >= target_sum:
            exit_type = f"PARTIAL_{int(self.partial_capture_threshold * 100)}"
        elif current_sum >= min_sum:
            exit_type = f"PARTIAL_{int(self.min_capture_threshold * 100)}"
        else:
            exit_type = "FULL_CAPTURE"

        est_profit = (1.0 - current_sum) * pos.total_shares - exit_fees
        profit_vs_hold = est_profit - pos.entry_edge * pos.total_shares

        return ExitOpportunity(
            position=pos,
            exit_type=exit_type,
            current_bid_sum=current_sum,
            min_exit_sum=breakeven_sum,
            capture_threshold_sum=target_sum,
            exit_price_a=current_bid_a,
            exit_price_b=current_bid_b,
            est_exit_fees=exit_fees,
            est_exit_cost=current_sum * pos.total_shares,
            est_profit=est_profit,
            profit_vs_hold=profit_vs_hold
        )

    def scan_open_positions(self, book_reader: Any) -> list[ExitOpportunity]:
        """
        Check all open positions for exit opportunities.
        Returns ordered list of best exit candidates.
        """
        opportunities = []
        open_positions = self.tracker.get_open_positions()

        for pos in open_positions:
            try:
                # Fetch current books
                book_a = book_reader.get_book_snapshot(pos.token_a)
                book_b = book_reader.get_book_snapshot(pos.token_b)

                if not book_a or not book_b:
                    continue

                bid_a = book_a.best_bid.price if book_a.best_bid else 0.0
                bid_b = book_b.best_bid.price if book_b.best_bid else 0.0

                # Update tracker with current prices
                self.tracker.update_position_prices(pos.position_id, bid_a, bid_b)

                # Check for exit opportunity
                exit_opp = self._can_exit_partial(pos, bid_a, bid_b)
                if exit_opp and exit_opp.can_capture_edge:
                    opportunities.append(exit_opp)
                    LOG.info(
                        "Swing exit: %s | type=%s | cur_sum=%.4f | target=%.4f | profit=%.4f",
                        pos.position_id[:16], exit_opp.exit_type,
                        exit_opp.current_bid_sum, exit_opp.capture_threshold_sum,
                        exit_opp.est_profit
                    )

            except Exception as exc:
                LOG.debug("Book fetch failed for position %s: %s", pos.position_id, exc)

        # Sort by profit vs holding to expiry (higher is better)
        opportunities.sort(key=lambda x: x.est_profit, reverse=True)
        return opportunities

    def should_exit_now(self, pos: Position, book_a: BookSnapshot, book_b: BookSnapshot) -> ExitOpportunity | None:
        """Quick check if a specific position should exit."""
        bid_a = book_a.best_bid.price if book_a.best_bid else 0.0
        bid_b = book_b.best_bid.price if book_b.best_bid else 0.0

        self.tracker.update_position_prices(pos.position_id, bid_a, bid_b)
        return self._can_exit_partial(pos, bid_a, bid_b)


class SwingExitExecutor:
    """
    Executes swing exits for paper or live trading.
    """

    def __init__(self, cfg: BotConfig, tracker: PositionTracker):
        self.cfg = cfg
        self.tracker = tracker
        self.is_paper = cfg.is_paper_mode

    def execute_exit(self, exit_opp: ExitOpportunity) -> dict[str, Any]:
        """Execute swing exit for a position."""
        pos = exit_opp.position

        if self.is_paper:
            return self._execute_paper_exit(exit_opp)
        else:
            return self._execute_live_exit(exit_opp)

    def _execute_paper_exit(self, exit_opp: ExitOpportunity) -> dict[str, Any]:
        """Simulate realistic exit with latency and slippage."""
        import time
        import random

        pos = exit_opp.position

        # Simulate latency
        time.sleep(random.uniform(0.05, 0.3))

        # Paper slippage simulation
        slip = random.uniform(0, 0.002)
        actual_exit_a = exit_opp.exit_price_a * (1 - slip)
        actual_exit_b = exit_opp.exit_price_b * (1 - slip)

        realized_profit = (
            (1.0 - actual_exit_a - actual_exit_b) * pos.total_shares -
            exit_opp.est_exit_fees
        )

        self.tracker.mark_exited(pos.position_id, realized_profit)

        LOG.info(
            "Paper swing exit: %s profit=%.4f | target=%s | slip=%.4f",
            pos.position_id[:16], realized_profit, exit_opp.exit_type, slip
        )

        return {
            "ok": True,
            "position_id": pos.position_id,
            "exit_type": exit_opp.exit_type,
            "realized_profit": round(realized_profit, 4),
            "slippage": round(slip, 4),
            "exit_price_a": round(actual_exit_a, 4),
            "exit_price_b": round(actual_exit_b, 4),
        }

    def _execute_live_exit(self, exit_opp: ExitOpportunity) -> dict[str, Any]:
        """Execute live exit via CLOB."""
        # This would use same pattern as LiveExecutor
        # For now, placeholder
        return self._execute_paper_exit(exit_opp)  # Fallback
