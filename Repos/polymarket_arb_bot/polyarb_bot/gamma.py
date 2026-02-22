from __future__ import annotations

import logging
from typing import Any, Iterable

import requests

from .models import LegMarket, PairCandidate
from .utils import (
    contains_red_flag_terms,
    is_yes_no_outcomes,
    normalize_outcomes_tokens,
    outcome_token_map,
    parse_dt,
    stable_hash,
)

LOG = logging.getLogger("polyarb")

GAMMA_BASE = "https://gamma-api.polymarket.com"


class GammaClient:
    def __init__(self, session: requests.Session | None = None, timeout: int = 15):
        self.session = session or requests.Session()
        self.timeout = timeout

    def list_events(self, *, active: bool = True, closed: bool = False, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        r = self.session.get(f"{GAMMA_BASE}/events", params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def iter_active_events(self, page_size: int = 100, max_pages: int = 3) -> Iterable[dict[str, Any]]:
        for page in range(max_pages):
            offset = page * page_size
            events = self.list_events(limit=page_size, offset=offset)
            if not events:
                break
            yield from events
            if len(events) < page_size:
                break


def _sports_like(event: dict[str, Any]) -> bool:
    fields = [
        str(event.get("category") or ""),
        str(event.get("subcategory") or ""),
        str(event.get("sportsMarketType") or ""),
        str(event.get("title") or ""),
    ]
    s = " | ".join(fields).lower()
    needles = (
        "sport", "nba", "wnba", "nfl", "nhl", "mlb", "soccer", "premier league", "serie a",
        "ncaab", "ncaa", "tennis", "ufc", "mma", "boxing", "hockey", "basketball", "baseball", "football",
    )
    return any(n in s for n in needles)


def _market_is_tradable(m: dict[str, Any]) -> bool:
    return (
        bool(m.get("active", True))
        and not bool(m.get("closed", False))
        and bool(m.get("enableOrderBook", True))
        and bool(m.get("acceptingOrders", True))
    )


def _leg_from_market(event: dict[str, Any], market: dict[str, Any], token_id: str, side_label: str) -> LegMarket:
    start = (
        event.get("startDate")
        or event.get("start_time")
        or event.get("startTime")
        or market.get("startDate")
        or market.get("start_time")
    )
    return LegMarket(
        token_id=str(token_id),
        market_id=str(market.get("id", "")),
        event_id=str(event.get("id", "")),
        event_title=str(event.get("title") or event.get("slug") or ""),
        market_question=str(market.get("question") or market.get("title") or market.get("slug") or ""),
        side_label=side_label,
        category=str(event.get("category") or ""),
        subcategory=str(event.get("subcategory") or ""),
        start_time_utc=parse_dt(start),
        neg_risk=bool(market.get("negRisk") or event.get("negRisk")),
        raw=market,
    )


def _pair_rule_hash(a: LegMarket, b: LegMarket) -> str:
    return stable_hash(
        [
            a.event_id,
            a.start_time_utc,
            a.market_question,
            b.market_question,
            str(a.raw.get("resolutionSource") or ""),
            str(b.raw.get("resolutionSource") or ""),
        ]
    )


def _pair_red_flags(a: LegMarket, b: LegMarket) -> tuple[str, ...]:
    flags = []
    flags.extend(contains_red_flag_terms(a.market_question))
    flags.extend(contains_red_flag_terms(b.market_question))
    return tuple(sorted(set(flags)))


def extract_pair_candidates(
    events: Iterable[dict[str, Any]],
    *,
    only_sports: bool = True,
    skip_neg_risk: bool = True,
    skip_red_flag_rules: bool = True,
) -> list[PairCandidate]:
    out: list[PairCandidate] = []

    for event in events:
        if only_sports and not _sports_like(event):
            continue
        markets = event.get("markets") or []
        if not isinstance(markets, list) or not markets:
            continue

        tradable = [m for m in markets if isinstance(m, dict) and _market_is_tradable(m)]
        if skip_neg_risk:
            tradable = [m for m in tradable if not bool(m.get("negRisk") or event.get("negRisk"))]
        if not tradable:
            continue

        yesno_markets: list[tuple[dict[str, Any], dict[str, str]]] = []
        for m in tradable:
            outcomes, tokens = normalize_outcomes_tokens(m)
            if len(outcomes) == 2 and len(tokens) == 2 and is_yes_no_outcomes(outcomes):
                yesno_markets.append((m, outcome_token_map(m)))

        if len(yesno_markets) == 2:
            (m1, map1), (m2, map2) = yesno_markets
            for label, pair_kind in (("YES", "YES_PAIR"), ("NO", "NO_PAIR")):
                t1 = map1.get(label) or map1.get(label.lower())
                t2 = map2.get(label) or map2.get(label.lower())
                if not (t1 and t2):
                    continue
                a = _leg_from_market(event, m1, t1, label)
                b = _leg_from_market(event, m2, t2, label)
                flags = _pair_red_flags(a, b)
                if skip_red_flag_rules and flags:
                    continue
                out.append(
                    PairCandidate(
                        pair_kind=pair_kind,
                        pair_label=f"{a.market_question} [{label}] + {b.market_question} [{label}]",
                        leg_a=a,
                        leg_b=b,
                        validation_flags=flags,
                        rule_hash=_pair_rule_hash(a, b),
                    )
                )
            continue

        for m in tradable:
            outcomes, tokens = normalize_outcomes_tokens(m)
            if len(outcomes) != 2 or len(tokens) != 2:
                continue
            if is_yes_no_outcomes(outcomes):
                continue
            a = _leg_from_market(event, m, tokens[0], str(outcomes[0]))
            b = _leg_from_market(event, m, tokens[1], str(outcomes[1]))
            flags = _pair_red_flags(a, b)
            if skip_red_flag_rules and flags:
                continue
            out.append(
                PairCandidate(
                    pair_kind="OUTCOME_PAIR",
                    pair_label=f"{a.market_question}: {a.side_label} + {b.side_label}",
                    leg_a=a,
                    leg_b=b,
                    validation_flags=flags,
                    rule_hash=_pair_rule_hash(a, b),
                )
            )

    seen: set[tuple[str, str, str]] = set()
    deduped: list[PairCandidate] = []
    for c in out:
        key = tuple(sorted([c.leg_a.token_id, c.leg_b.token_id])) + (c.pair_kind,)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped
