from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class BookLevel:
    price: float
    size: float


@dataclass(slots=True)
class BookSnapshot:
    token_id: str
    bids: list[BookLevel]
    asks: list[BookLevel]
    asof_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def total_ask_size(self) -> float:
        return sum(x.size for x in self.asks)

    @property
    def total_bid_size(self) -> float:
        return sum(x.size for x in self.bids)


@dataclass(slots=True)
class DepthFill:
    shares: float
    vwap: float
    worst_price: float
    total_cost: float
    levels_used: int


@dataclass(slots=True)
class LegMarket:
    token_id: str
    market_id: str
    event_id: str
    event_title: str
    market_question: str
    side_label: str
    category: str
    subcategory: str
    start_time_utc: str = ""
    neg_risk: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PairCandidate:
    pair_kind: str  # YES_PAIR | NO_PAIR | OUTCOME_PAIR
    pair_label: str
    leg_a: LegMarket
    leg_b: LegMarket
    validation_flags: tuple[str, ...] = ()
    rule_hash: str = ""


@dataclass(slots=True)
class Opportunity:
    candidate: PairCandidate
    book_a: BookSnapshot
    book_b: BookSnapshot
    fill_a: DepthFill
    fill_b: DepthFill
    fee_bps_a: float
    fee_bps_b: float
    fee_total_a: float
    fee_total_b: float
    gross_cost_total: float
    net_cost_total: float
    edge_total: float
    edge_per_share: float
    target_shares: float
    slippage_reserve_total: float
    ops_cost_total: float
    lower_bound: bool = True

    @property
    def est_profit(self) -> float:
        return self.edge_total

    @property
    def gross_cost_per_share(self) -> float:
        return self.gross_cost_total / self.target_shares if self.target_shares > 0 else 0.0

    @property
    def net_cost_per_share(self) -> float:
        return self.net_cost_total / self.target_shares if self.target_shares > 0 else 0.0

    @property
    def ask_a(self) -> BookLevel:
        return self.book_a.best_ask or BookLevel(0.0, 0.0)

    @property
    def ask_b(self) -> BookLevel:
        return self.book_b.best_ask or BookLevel(0.0, 0.0)

    @property
    def worst_limit_a(self) -> float:
        return self.fill_a.worst_price

    @property
    def worst_limit_b(self) -> float:
        return self.fill_b.worst_price
