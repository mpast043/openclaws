from __future__ import annotations

import logging
from typing import Any

from .models import BookLevel, BookSnapshot, DepthFill
from .utils import safe_float

LOG = logging.getLogger("polyarb")


class ClobBookReader:
    def __init__(self, host: str = "https://clob.polymarket.com"):
        self.host = host
        try:
            from py_clob_client.client import ClobClient  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("py-clob-client is required. Install with `pip install py-clob-client`.") from exc
        self._client = ClobClient(host)

    @staticmethod
    def _levels_from_side(side_obj: Any) -> list[BookLevel]:
        levels: list[BookLevel] = []
        if side_obj is None:
            return levels
        if isinstance(side_obj, dict):
            side_iter = side_obj.get("levels") or side_obj.get("orders") or []
        else:
            side_iter = side_obj
        for lvl in side_iter:
            if isinstance(lvl, dict):
                price = safe_float(lvl.get("price"))
                size = safe_float(lvl.get("size") or lvl.get("quantity"))
            else:
                price = safe_float(getattr(lvl, "price", 0.0))
                size = safe_float(getattr(lvl, "size", getattr(lvl, "quantity", 0.0)))
            if price > 0 and size > 0:
                levels.append(BookLevel(price=price, size=size))
        return levels

    def get_book_snapshot(self, token_id: str) -> BookSnapshot:
        book = self._client.get_order_book(token_id)
        asks = self._levels_from_side(getattr(book, "asks", None))
        bids = self._levels_from_side(getattr(book, "bids", None))
        if isinstance(book, dict):
            if not asks:
                asks = self._levels_from_side(book.get("asks"))
            if not bids:
                bids = self._levels_from_side(book.get("bids"))
        asks.sort(key=lambda x: x.price)
        bids.sort(key=lambda x: x.price, reverse=True)
        return BookSnapshot(token_id=str(token_id), asks=asks, bids=bids)

    def get_best_ask(self, token_id: str) -> BookLevel | None:
        return self.get_book_snapshot(token_id).best_ask

    def get_best_bid(self, token_id: str) -> BookLevel | None:
        return self.get_book_snapshot(token_id).best_bid

    @staticmethod
    def simulate_take(levels: list[BookLevel], shares: float) -> DepthFill | None:
        if shares <= 0:
            return None
        rem = shares
        total = 0.0
        worst = 0.0
        used = 0
        for lvl in levels:
            if rem <= 1e-12:
                break
            take = min(rem, lvl.size)
            total += take * lvl.price
            rem -= take
            worst = lvl.price
            used += 1
        if rem > 1e-9:
            return None
        return DepthFill(
            shares=shares,
            vwap=(total / shares) if shares > 0 else 0.0,
            worst_price=worst,
            total_cost=total,
            levels_used=used,
        )

    @staticmethod
    def cumulative_sizes(levels: list[BookLevel], max_levels: int = 25) -> list[float]:
        out = []
        s = 0.0
        for i, lvl in enumerate(levels):
            if i >= max_levels:
                break
            s += lvl.size
            out.append(s)
        return out
