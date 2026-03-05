from __future__ import annotations

import logging
import time
from typing import Any

from .config import BotConfig
from .executor import BundleLedger, LiveExecutor, PaperExecutor, TradeLogger
from .positions import Position, PositionTracker
from .scanner import ArbScanner
from .swing import SwingExitScanner, SwingExitExecutor
from .ws_quotes import PolymarketBookStream, QuoteCache

LOG = logging.getLogger("polyarb")


class PolymarketArbBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.quote_cache = QuoteCache() if cfg.enable_websocket_quotes else None
        self.ws_stream = PolymarketBookStream(cfg.ws_url, self.quote_cache) if self.quote_cache else None
        self.scanner = ArbScanner(cfg, quote_cache=self.quote_cache)
        self.trade_logger = TradeLogger(cfg.csv_log_path)
        self.bundle_ledger = BundleLedger(cfg.bundle_log_path)
        
        # Position tracking for swing trading
        self.position_tracker = PositionTracker(cfg.positions_path)
        
        self.executor = None
        if cfg.is_paper_mode:
            self.executor = PaperExecutor(cfg, self.trade_logger, self.bundle_ledger, book_reader=self.scanner.books)
        elif cfg.is_live_mode:
            self.executor = LiveExecutor(cfg, self.trade_logger, self.bundle_ledger)
        
        # Swing exit scanner/executor
        self.swing_scanner = SwingExitScanner(cfg, self.position_tracker)
        self.swing_executor = SwingExitExecutor(cfg, self.position_tracker)

    def _prime_ws_subscriptions(self) -> None:
        if not self.ws_stream:
            return
        # subscribe to a slice of current candidate universe
        try:
            events = list(self.scanner.gamma.iter_active_events(page_size=min(100, self.cfg.events_page_size), max_pages=1))
            candidates = self.scanner.gamma and []  # no-op placeholder for readability
            from .gamma import extract_pair_candidates
            pairs = extract_pair_candidates(
                events,
                only_sports=self.cfg.only_sports,
                skip_neg_risk=self.cfg.skip_neg_risk,
                skip_red_flag_rules=self.cfg.skip_red_flag_rules,
            )
            token_ids = []
            for p in pairs[:250]:
                token_ids.extend([p.leg_a.token_id, p.leg_b.token_id])
            self.ws_stream.set_tokens(list(dict.fromkeys(token_ids)))
            self.ws_stream.start()
        except Exception as exc:
            LOG.debug("websocket prime failed: %s", exc)

    def run_once(self) -> dict[str, Any]:
        if self.ws_stream:
            self._prime_ws_subscriptions()

        # First: Check swing exits for open positions
        if self.cfg.enable_swing_exits:
            try:
                exit_opps = self.swing_scanner.scan_open_positions(self.scanner.books)
                for exit_opp in exit_opps[:self.cfg.top_n_opps]:
                    LOG.info("Swing exit triggered for %s", exit_opp.position.position_id[:16])
                    result = self.swing_executor.execute_exit(exit_opp)
                    LOG.info("Swing exit result: %s", result)
            except Exception as exc:
                LOG.debug("Swing scan error: %s", exc)

        # Show position stats
        stats = self.position_tracker.get_stats()
        if stats["open_count"] > 0:
            LOG.info("Open positions: %s cost=%s pnl=%s", 
                    stats["open_count"], stats["open_cost"], stats["open_unrealized_pnl"])

        # Check position cap
        if len(self.position_tracker.get_open_positions()) >= self.cfg.max_concurrent_positions:
            LOG.info("Position cap reached (%s), skipping new entries", self.cfg.max_concurrent_positions)
            return {"ok": True, "opportunities": 0, "cap_reached": True}

        res = self.scanner.scan_once()
        LOG.info(
            "Scanned %s events, %s candidates, %s opportunities (%s skipped)",
            res.event_count, res.candidate_count, len(res.opportunities), res.skipped_count
        )

        if not res.opportunities:
            return {"ok": True, "opportunities": 0}

        top = res.opportunities[: self.cfg.top_n_opps]
        executed = 0
        for i, opp in enumerate(top, start=1):
            LOG.info(
                "[%s] q=%.2f edge=%.4f/shr pnl=%.2f gross=%.4f net=%.4f | %s",
                i,
                opp.target_shares,
                opp.edge_per_share,
                opp.edge_total,
                opp.gross_cost_per_share,
                opp.net_cost_per_share,
                opp.candidate.pair_label,
            )
            if self.cfg.print_books:
                LOG.info(
                    "    A ask %.4f bid %.4f depthVWAP %.4f worst %.4f | B ask %.4f bid %.4f depthVWAP %.4f worst %.4f",
                    opp.ask_a.price,
                    (opp.book_a.best_bid.price if opp.book_a.best_bid else 0.0),
                    opp.fill_a.vwap,
                    opp.fill_a.worst_price,
                    opp.ask_b.price,
                    (opp.book_b.best_bid.price if opp.book_b.best_bid else 0.0),
                    opp.fill_b.vwap,
                    opp.fill_b.worst_price,
                )
            if self.executor:
                exec_res = self.executor.execute(opp)
                LOG.info("    execution: %s", exec_res)
                
                # Record position if filled
                if exec_res.get("ok") and exec_res.get("status") in ("FILLED", "OPEN_EXPECTED"):
                    self._record_filled_position(opp, exec_res)
                    executed += 1

        return {"ok": True, "opportunities": len(res.opportunities), "executed": executed}
    
    def _record_filled_position(self, opp, exec_res: dict[str, Any]) -> None:
        """Record a filled position to the tracker."""
        import uuid
        from datetime import datetime, timezone
        
        pos = Position(
            position_id=str(uuid.uuid4())[:16],
            bundle_id=exec_res.get("bundle_id", ""),
            mode="paper" if self.cfg.is_paper_mode else "live",
            token_a=opp.candidate.leg_a.token_id,
            side_a=opp.candidate.leg_a.side_label,
            shares_a=opp.target_shares,
            fill_price_a=opp.fill_a.vwap,
            cost_a=opp.fill_a.total_cost,
            fee_a=opp.fee_total_a,
            token_b=opp.candidate.leg_b.token_id,
            side_b=opp.candidate.leg_b.side_label,
            shares_b=opp.target_shares,
            fill_price_b=opp.fill_b.vwap,
            cost_b=opp.fill_b.total_cost,
            fee_b=opp.fee_total_b,
            total_cost=opp.net_cost_total,
            total_fees=opp.fee_total_a + opp.fee_total_b,
            entry_edge=opp.edge_per_share,
            current_bid_a=opp.book_a.best_bid.price if opp.book_a.best_bid else 0,
            current_bid_b=opp.book_b.best_bid.price if opp.book_b.best_bid else 0,
            unrealized_pnl=opp.edge_total,
        )
        self.position_tracker.add_position(pos)
        LOG.info("Recorded position %s: %s shares, edge=%.4f", 
                pos.position_id, pos.shares_a, pos.entry_edge)

    def run_forever(self) -> None:
        LOG.info("Starting bot loop: mode=%s interval=%ss", self.cfg.bot_mode, self.cfg.scan_interval_seconds)
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                LOG.exception("Loop error: %s", exc)
            time.sleep(self.cfg.scan_interval_seconds)
