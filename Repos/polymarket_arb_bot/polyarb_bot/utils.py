from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any


LOG = logging.getLogger("polyarb")


def parse_maybe_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            if "|" in s:
                return [x.strip() for x in s.split("|") if x.strip()]
            if "," in s:
                return [x.strip() for x in s.split(",") if x.strip()]
            return [s]
    return []


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_yes_no_outcomes(outcomes: list[str]) -> bool:
    norm = [o.strip().lower() for o in outcomes]
    return len(norm) == 2 and set(norm) == {"yes", "no"}


def normalize_outcomes_tokens(market: dict[str, Any]) -> tuple[list[str], list[str]]:
    outcomes = [str(x) for x in parse_maybe_json_list(market.get("outcomes"))]
    tokens = [str(x) for x in parse_maybe_json_list(market.get("clobTokenIds"))]
    return outcomes, tokens


def outcome_token_map(market: dict[str, Any]) -> dict[str, str]:
    outcomes, tokens = normalize_outcomes_tokens(market)
    if len(outcomes) != len(tokens):
        return {}
    return {str(o).strip(): str(t) for o, t in zip(outcomes, tokens)}


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def parse_dt(val: Any) -> str:
    if not val:
        return ""
    s = str(val)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).isoformat()
    except Exception:
        return str(val)


def stable_hash(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def contains_red_flag_terms(text: str) -> list[str]:
    s = normalize_text(text)
    flags = []
    for term in ("draw", "tie", "overtime excluded", "regulation only", "void", "canceled", "cancelled", "postponed"):
        if term in s:
            flags.append(term)
    return flags
