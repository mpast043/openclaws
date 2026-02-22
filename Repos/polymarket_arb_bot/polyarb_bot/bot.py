from __future__ import annotations

import logging
import time
from typing import Any

from .config import BotConfig
from .executor import BundleLedger, LiveExecutor, PaperExecutor, TradeLogger
from .scanner import ArbScanner
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
        self.executor = None
        if cfg.is_paper_mode:
            self.executor = PaperExecutor(cfg, self.trade_logger, self.bundle_ledger)
        elif cfg.is_live_mode:
            self.executor = LiveExecutor(cfg, self.trade_logger, self.bundle_ledger)

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

        res = self.scanner.scan_once()
        LOG.info(
            "Scanned %s events, %s candidates, %s opportunities (%s skipped)",
            res.event_count, res.candidate_count, len(res.opportunities), res.skipped_count
        )

        if not res.opportunities:
            return {"ok": True, "opportunities": 0}

        top = res.opportunities[: self.cfg.top_n_opps]
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

        return {"ok": True, "opportunities": len(res.opportunities)}

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
