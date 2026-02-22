from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BotConfig
from .models import BookLevel, Opportunity

LOG = logging.getLogger("polyarb")


class TradeLogger:
    _FIELDS = [
        "ts_utc","mode","status","pair_kind","pair_label","event_id","event_title","rule_hash",
        "token_a","token_b","leg_a_label","leg_b_label",
        "best_ask_a","best_ask_b","best_bid_a","best_bid_b",
        "vwap_a","vwap_b","worst_a","worst_b","levels_used_a","levels_used_b",
        "fee_bps_a","fee_bps_b","fee_total_a","fee_total_b",
        "edge_per_share","target_shares","est_profit","gross_cost","net_cost","fees",
        "slippage_reserve","ops_cost","resp_a","resp_b","resp_hedge","reason","note"
    ]

    def __init__(self, csv_path: str):
        self.path = Path(csv_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, opp: Opportunity, mode: str, status: str, extra: dict[str, Any] | None = None) -> None:
        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "status": status,
            "pair_kind": opp.candidate.pair_kind,
            "pair_label": opp.candidate.pair_label,
            "event_id": opp.candidate.leg_a.event_id,
            "event_title": opp.candidate.leg_a.event_title,
            "rule_hash": opp.candidate.rule_hash,
            "token_a": opp.candidate.leg_a.token_id,
            "token_b": opp.candidate.leg_b.token_id,
            "leg_a_label": opp.candidate.leg_a.side_label,
            "leg_b_label": opp.candidate.leg_b.side_label,
            "best_ask_a": opp.ask_a.price,
            "best_ask_b": opp.ask_b.price,
            "best_bid_a": (opp.book_a.best_bid.price if opp.book_a.best_bid else 0.0),
            "best_bid_b": (opp.book_b.best_bid.price if opp.book_b.best_bid else 0.0),
            "vwap_a": opp.fill_a.vwap,
            "vwap_b": opp.fill_b.vwap,
            "worst_a": opp.fill_a.worst_price,
            "worst_b": opp.fill_b.worst_price,
            "levels_used_a": opp.fill_a.levels_used,
            "levels_used_b": opp.fill_b.levels_used,
            "fee_bps_a": opp.fee_bps_a,
            "fee_bps_b": opp.fee_bps_b,
            "fee_total_a": opp.fee_total_a,
            "fee_total_b": opp.fee_total_b,
            "edge_per_share": opp.edge_per_share,
            "target_shares": opp.target_shares,
            "est_profit": opp.edge_total,
            "gross_cost": opp.gross_cost_total,
            "net_cost": opp.net_cost_total,
            "fees": (opp.fee_total_a + opp.fee_total_b),
            "slippage_reserve": opp.slippage_reserve_total,
            "ops_cost": opp.ops_cost_total,
        }
        if extra:
            row.update(extra)

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class BundleLedger:
    def __init__(self, csv_path: str):
        self.path = Path(csv_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0
        self._fields = [
            "ts_utc","mode","bundle_id","status","event_id","pair_kind","pair_label","rule_hash",
            "token_a","token_b","shares","gross_cost","fee_total","net_cost","expected_payout","locked_pnl",
            "note",
        ]

    def record(self, opp: Opportunity, mode: str, status: str, note: str = "") -> str:
        bundle_id = f"{opp.candidate.leg_a.event_id}:{opp.candidate.rule_hash}:{datetime.now(timezone.utc).timestamp():.6f}"
        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "bundle_id": bundle_id,
            "status": status,
            "event_id": opp.candidate.leg_a.event_id,
            "pair_kind": opp.candidate.pair_kind,
            "pair_label": opp.candidate.pair_label,
            "rule_hash": opp.candidate.rule_hash,
            "token_a": opp.candidate.leg_a.token_id,
            "token_b": opp.candidate.leg_b.token_id,
            "shares": opp.target_shares,
            "gross_cost": opp.gross_cost_total,
            "fee_total": opp.fee_total_a + opp.fee_total_b,
            "net_cost": opp.net_cost_total,
            "expected_payout": opp.target_shares,
            "locked_pnl": opp.edge_total,
            "note": note,
        }
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerow(row)
        return bundle_id


class PaperExecutor:
    def __init__(self, cfg: BotConfig, trade_logger: TradeLogger, bundle_ledger: BundleLedger):
        self.cfg = cfg
        self.log = trade_logger
        self.ledger = bundle_ledger

    def execute(self, opp: Opportunity) -> dict[str, Any]:
        shares = opp.target_shares
        if shares < self.cfg.min_shares:
            result = {"ok": False, "reason": "shares_below_min"}
            self.log.log(opp, mode="paper", status="SKIP", extra=result)
            return result

        result = {
            "ok": True,
            "shares": round(shares, 4),
            "gross_cost": round(opp.gross_cost_total, 4),
            "fees": round(opp.fee_total_a + opp.fee_total_b, 4),
            "est_profit": round(opp.edge_total, 4),
            "note": "paper_depth_vwap_fill",
        }
        bundle_id = self.ledger.record(opp, mode="paper", status="OPEN_EXPECTED", note="paper")
        result["note"] += f";bundle={bundle_id}"
        self.log.log(opp, mode="paper", status="FILLED", extra=result)
        return result


class LiveExecutor:
    """
    Live execution via py-clob-client with:
    - depth-aware sizing from scanner
    - FOK per-leg
    - second-leg failure hedge fallback (best-effort)
    """

    def __init__(self, cfg: BotConfig, trade_logger: TradeLogger, bundle_ledger: BundleLedger):
        self.cfg = cfg
        self.log = trade_logger
        self.ledger = bundle_ledger
        try:
            from py_clob_client.client import ClobClient  # type: ignore
            from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore
            from py_clob_client.order_builder.constants import BUY  # type: ignore
            try:
                from py_clob_client.order_builder.constants import SELL  # type: ignore
            except Exception:
                SELL = "SELL"
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("py-clob-client is required for live mode") from exc

        self._OrderArgs = OrderArgs
        self._OrderType = OrderType
        self._BUY = BUY
        self._SELL = SELL

        kwargs: dict[str, Any] = {
            "key": cfg.poly_private_key,
            "chain_id": cfg.poly_chain_id,
            "signature_type": cfg.poly_signature_type,
        }
        if cfg.poly_funder:
            kwargs["funder"] = cfg.poly_funder
        self.client = ClobClient(cfg.poly_host, **kwargs)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def _post_fok_limit(self, token_id: str, price: float, size: float, side: Any) -> Any:
        order = self._OrderArgs(token_id=token_id, price=price, size=size, side=side)
        signed = self.client.create_order(order)
        return self.client.post_order(signed, self._OrderType.FOK)

    def _get_best_bid(self, token_id: str) -> BookLevel | None:
        try:
            book = self.client.get_order_book(token_id)
            side = getattr(book, "bids", None)
            if side is None and isinstance(book, dict):
                side = book.get("bids")
            bids = []
            if isinstance(side, dict):
                side = side.get("levels") or side.get("orders") or []
            for lvl in side or []:
                if isinstance(lvl, dict):
                    p = float(lvl.get("price")); s = float(lvl.get("size") or lvl.get("quantity"))
                else:
                    p = float(getattr(lvl, "price", 0)); s = float(getattr(lvl, "size", getattr(lvl, "quantity", 0)))
                if p > 0 and s > 0:
                    bids.append(BookLevel(price=p, size=s))
            bids.sort(key=lambda x: x.price, reverse=True)
            return bids[0] if bids else None
        except Exception:
            return None

    @staticmethod
    def _resp_ok(resp: Any) -> bool:
        if resp is None:
            return False
        if isinstance(resp, dict):
            if resp.get("error"):
                return False
            status = str(resp.get("status") or resp.get("state") or "").lower()
            if status in {"filled", "matched", "live", "accepted", "success", "ok"}:
                return True
            success = resp.get("success")
            if isinstance(success, bool):
                return success
            return bool(resp.get("orderID") or resp.get("orderId") or resp.get("id"))
        return bool(resp)

    def _hedge_leg_a(self, opp: Opportunity, shares: float) -> tuple[bool, str]:
        if not self.cfg.hedge_on_leg_fail:
            return False, "hedge_disabled"
        bid = self._get_best_bid(opp.candidate.leg_a.token_id)
        if not bid:
            return False, "no_bid_for_hedge"
        hedge_px = max(0.001, round(bid.price - self.cfg.hedge_price_buffer, 4))
        try:
            resp = self._post_fok_limit(opp.candidate.leg_a.token_id, hedge_px, shares, self._SELL)
            return self._resp_ok(resp), str(resp)[:500]
        except Exception as exc:
            return False, f"hedge_exception:{exc}"

    def execute(self, opp: Opportunity) -> dict[str, Any]:
        shares = round(opp.target_shares, 4)
        if shares < self.cfg.min_shares:
            result = {"ok": False, "reason": "shares_below_min"}
            self.log.log(opp, mode="live", status="SKIP", extra=result)
            return result

        if not self.cfg.execution_enabled:
            result = {"ok": False, "reason": "execution_disabled"}
            self.log.log(opp, mode="live", status="DRY_SKIP", extra=result)
            return result

        px_a = round(min(0.999, opp.worst_limit_a + self.cfg.latency_slippage_buffer_per_leg), 4)
        px_b = round(min(0.999, opp.worst_limit_b + self.cfg.latency_slippage_buffer_per_leg), 4)

        try:
            resp_a = self._post_fok_limit(opp.candidate.leg_a.token_id, px_a, shares, self._BUY)
            if not self._resp_ok(resp_a):
                result = {"ok": False, "reason": "leg_a_failed", "resp_a": str(resp_a)[:500]}
                self.log.log(opp, mode="live", status="FAIL", extra=result)
                return result

            resp_b = self._post_fok_limit(opp.candidate.leg_b.token_id, px_b, shares, self._BUY)
            if not self._resp_ok(resp_b):
                hedge_ok, hedge_resp = self._hedge_leg_a(opp, shares)
                status = "HEDGED_FAILSAFE" if hedge_ok else "UNHEDGED_ALERT"
                result = {
                    "ok": False,
                    "reason": "leg_b_failed_after_leg_a",
                    "resp_a": str(resp_a)[:500],
                    "resp_b": str(resp_b)[:500],
                    "resp_hedge": hedge_resp,
                    "note": "attempted_hedge" if self.cfg.hedge_on_leg_fail else "no_hedge",
                }
                self.log.log(opp, mode="live", status=status, extra=result)
                return result

            bundle_id = self.ledger.record(opp, mode="live", status="OPEN_EXPECTED", note="live_pair_filled")
            result = {
                "ok": True,
                "shares": shares,
                "gross_cost": round(opp.gross_cost_total, 4),
                "fees": round(opp.fee_total_a + opp.fee_total_b, 4),
                "est_profit": round(opp.edge_total, 4),
                "resp_a": str(resp_a)[:500],
                "resp_b": str(resp_b)[:500],
                "note": f"bundle={bundle_id}",
            }
            self.log.log(opp, mode="live", status="FILLED", extra=result)
            return result
        except Exception as exc:
            result = {"ok": False, "reason": "exception", "note": str(exc)}
            self.log.log(opp, mode="live", status="EXCEPTION", extra=result)
            return result
