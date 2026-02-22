from __future__ import annotations

import logging
from dataclasses import dataclass

from .clob_reader import ClobBookReader
from .config import BotConfig
from .fees import FeeRateClient
from .gamma import GammaClient, extract_pair_candidates
from .models import BookSnapshot, Opportunity, PairCandidate
from .ws_quotes import QuoteCache

LOG = logging.getLogger("polyarb")


@dataclass(slots=True)
class ScanResult:
    opportunities: list[Opportunity]
    candidate_count: int
    event_count: int
    skipped_count: int = 0


class ArbScanner:
    def __init__(self, cfg: BotConfig, quote_cache: QuoteCache | None = None):
        self.cfg = cfg
        self.gamma = GammaClient()
        self.books = ClobBookReader(cfg.poly_host)
        self.fees = FeeRateClient(cfg.poly_host)
        self.quote_cache = quote_cache

    def scan_once(self) -> ScanResult:
        events = list(self.gamma.iter_active_events(page_size=self.cfg.events_page_size, max_pages=self.cfg.max_event_pages))
        candidates = extract_pair_candidates(
            events,
            only_sports=self.cfg.only_sports,
            skip_neg_risk=self.cfg.skip_neg_risk,
            skip_red_flag_rules=self.cfg.skip_red_flag_rules,
        )

        opps: list[Opportunity] = []
        skipped = 0
        for c in candidates:
            try:
                opp = self._score_candidate(c)
            except Exception as exc:
                skipped += 1
                LOG.debug("score failed for %s: %s", c.pair_label, exc)
                continue
            if opp is None:
                skipped += 1
                continue
            if opp.edge_per_share >= self.cfg.min_edge_usdc_per_share and opp.edge_total >= self.cfg.min_edge_total_usdc and opp.target_shares >= self.cfg.min_shares:
                opps.append(opp)

        opps.sort(key=lambda o: (o.edge_total, o.edge_per_share), reverse=True)
        return ScanResult(opportunities=opps, candidate_count=len(candidates), event_count=len(events), skipped_count=skipped)

    def _get_book(self, token_id: str) -> BookSnapshot:
        if self.quote_cache:
            cached = self.quote_cache.get(token_id)
            if cached and cached.asks and cached.bids:
                return cached
        return self.books.get_book_snapshot(token_id)

    def _score_candidate(self, c: PairCandidate) -> Opportunity | None:
        book_a = self._get_book(c.leg_a.token_id)
        book_b = self._get_book(c.leg_b.token_id)
        if not book_a.asks or not book_b.asks:
            return None

        # Quick top-of-book reject
        top_sum = (book_a.best_ask.price if book_a.best_ask else 1) + (book_b.best_ask.price if book_b.best_ask else 1)
        if top_sum >= self.cfg.max_acceptable_combined_cost:
            return None

        fee_bps_a = self.fees.get_fee_bps(c.leg_a.token_id)
        fee_bps_b = self.fees.get_fee_bps(c.leg_b.token_id)

        cum_a = self.books.cumulative_sizes(book_a.asks, max_levels=self.cfg.depth_levels_limit)
        cum_b = self.books.cumulative_sizes(book_b.asks, max_levels=self.cfg.depth_levels_limit)
        if not cum_a or not cum_b:
            return None

        q_max_liq = min(cum_a[-1], cum_b[-1], self.cfg.max_shares_per_trade)
        if q_max_liq < self.cfg.min_shares:
            return None

        # Conservative budget upper bound from top-of-book
        top_cash_per_share = top_sum + 2 * self.cfg.latency_slippage_buffer_per_leg
        q_max_budget = self.cfg.capital_per_trade_usdc / max(top_cash_per_share, 1e-9)
        q_cap = min(q_max_liq, q_max_budget)
        if q_cap < self.cfg.min_shares:
            return None

        qty_candidates = {round(self.cfg.min_shares, 6), round(q_cap, 6)}
        for q in cum_a + cum_b:
            if self.cfg.min_shares <= q <= q_cap:
                qty_candidates.add(round(q, 6))
        # Add a few interior points to reduce chance of missing a better q between breakpoints
        for frac in (0.25, 0.5, 0.75):
            q = q_cap * frac
            if q >= self.cfg.min_shares:
                qty_candidates.add(round(q, 6))

        best: Opportunity | None = None
        for q in sorted(qty_candidates):
            fill_a = self.books.simulate_take(book_a.asks, q)
            fill_b = self.books.simulate_take(book_b.asks, q)
            if not fill_a or not fill_b:
                continue

            # Lower-bound fees/slippage use worst fill price on each leg
            fee_a = self.fees.estimate_buy_fee_usdc(q, fill_a.worst_price, fee_bps_a)
            fee_b = self.fees.estimate_buy_fee_usdc(q, fill_b.worst_price, fee_bps_b)
            gross = fill_a.total_cost + fill_b.total_cost
            slip = q * (2.0 * self.cfg.latency_slippage_buffer_per_leg)
            ops = self.cfg.ops_cost_per_bundle
            net = gross + fee_a + fee_b + slip + ops
            edge_total = q - net
            if edge_total <= 0:
                continue

            # Exact budget check with lower-bound costs
            if net > self.cfg.capital_per_trade_usdc:
                continue

            edge_per = edge_total / q
            opp = Opportunity(
                candidate=c,
                book_a=book_a,
                book_b=book_b,
                fill_a=fill_a,
                fill_b=fill_b,
                fee_bps_a=fee_bps_a,
                fee_bps_b=fee_bps_b,
                fee_total_a=fee_a,
                fee_total_b=fee_b,
                gross_cost_total=gross,
                net_cost_total=net,
                edge_total=edge_total,
                edge_per_share=edge_per,
                target_shares=q,
                slippage_reserve_total=slip,
                ops_cost_total=ops,
                lower_bound=True,
            )
            # Mathematically conservative sizing: choose largest q with positive lower bound.
            if best is None or q > best.target_shares or (abs(q - best.target_shares) < 1e-9 and edge_total > best.edge_total):
                best = opp

        return best
