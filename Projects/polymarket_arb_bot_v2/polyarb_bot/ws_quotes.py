from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .models import BookLevel, BookSnapshot
from .utils import safe_float

LOG = logging.getLogger("polyarb")


@dataclass
class QuoteCache:
    books: dict[str, BookSnapshot] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, token_id: str, bids: list[BookLevel], asks: list[BookLevel]) -> None:
        with self.lock:
            bids = sorted(bids, key=lambda x: x.price, reverse=True)
            asks = sorted(asks, key=lambda x: x.price)
            self.books[token_id] = BookSnapshot(token_id=token_id, bids=bids, asks=asks)

    def get(self, token_id: str) -> BookSnapshot | None:
        with self.lock:
            return self.books.get(token_id)


class PolymarketBookStream:
    """
    Optional websocket quote feed.
    This is a best-effort helper and is intentionally not required for the bot to run.
    If the websocket schema changes, polling remains the fallback.
    """

    def __init__(self, ws_url: str, quote_cache: QuoteCache):
        self.ws_url = ws_url
        self.quote_cache = quote_cache
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tokens: set[str] = set()

    def set_tokens(self, token_ids: list[str]) -> None:
        self._tokens = set(token_ids)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception:
            LOG.warning("websocket-client not installed; skipping websocket feed")
            return

        def on_open(ws):
            payload = {
                "type": "subscribe",
                "channels": ["market"],
                "assets_ids": list(self._tokens)[:500],
            }
            try:
                ws.send(json.dumps(payload))
            except Exception:
                pass

        def on_message(ws, message: str):
            try:
                msg = json.loads(message)
            except Exception:
                return
            self._handle_message(msg)

        while not self._stop.is_set():
            try:
                app = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=on_open,
                    on_message=on_message,
                )
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                LOG.debug("websocket loop error: %s", exc)
            time.sleep(2)

    def _handle_message(self, msg: Any) -> None:
        payloads = msg if isinstance(msg, list) else [msg]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            token = str(item.get("asset_id") or item.get("token_id") or item.get("tokenId") or "")
            if not token:
                continue
            bids_raw = item.get("bids") or item.get("bid") or []
            asks_raw = item.get("asks") or item.get("ask") or []
            bids = self._parse_levels(bids_raw)
            asks = self._parse_levels(asks_raw)
            if bids or asks:
                self.quote_cache.put(token, bids, asks)

    @staticmethod
    def _parse_levels(raw: Any) -> list[BookLevel]:
        out: list[BookLevel] = []
        if isinstance(raw, dict):
            raw = raw.get("levels") or raw.get("orders") or []
        for x in raw or []:
            if isinstance(x, dict):
                p = safe_float(x.get("price"))
                s = safe_float(x.get("size") or x.get("quantity"))
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                p = safe_float(x[0]); s = safe_float(x[1])
            else:
                continue
            if p > 0 and s > 0:
                out.append(BookLevel(price=p, size=s))
        return out
