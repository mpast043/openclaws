from __future__ import annotations

import logging
from typing import Any

import requests

from .utils import safe_float

LOG = logging.getLogger("polyarb")


class FeeRateClient:
    def __init__(self, host: str = "https://clob.polymarket.com", session: requests.Session | None = None, timeout: int = 10):
        self.host = host.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self._cache: dict[str, float] = {}

    def get_fee_bps(self, token_id: str) -> float:
        if token_id in self._cache:
            return self._cache[token_id]
        paths = [
            f"{self.host}/fee-rate",
            f"{self.host}/fee_rate",
            f"{self.host}/fee-rate/{token_id}",
        ]
        fee_bps = 0.0
        for path in paths:
            try:
                if path.endswith(token_id):
                    r = self.session.get(path, timeout=self.timeout)
                else:
                    r = self.session.get(path, params={"token_id": token_id}, timeout=self.timeout)
                if r.status_code >= 400:
                    continue
                data = r.json()
                fee_bps = self._parse_fee_bps(data)
                break
            except Exception:
                continue
        self._cache[token_id] = fee_bps
        return fee_bps

    @staticmethod
    def _parse_fee_bps(data: Any) -> float:
        if isinstance(data, dict):
            for k in ("fee_rate_bps", "feeRateBps", "fee_rate", "feeRate", "rateBps"):
                if k in data:
                    v = safe_float(data.get(k), 0.0)
                    if 0 < v < 1:
                        return v * 10000.0
                    return v
            for v in data.values():
                if isinstance(v, dict):
                    nested = FeeRateClient._parse_fee_bps(v)
                    if nested:
                        return nested
        return 0.0

    @staticmethod
    def estimate_buy_fee_usdc(shares: float, price: float, fee_bps: float, exponent: int = 1) -> float:
        if shares <= 0 or price <= 0 or fee_bps <= 0:
            return 0.0
        rate = fee_bps / 10000.0
        if exponent == 1:
            return shares * rate * price * (1.0 - price)
        return shares * rate * price * ((1.0 - price) ** exponent)
