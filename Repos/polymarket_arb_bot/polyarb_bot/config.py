from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw != "" else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw != "" else default


@dataclass(slots=True)
class BotConfig:
    bot_mode: str = "scan"
    scan_interval_seconds: int = 45
    max_event_pages: int = 3
    events_page_size: int = 100
    top_n_opps: int = 5

    # Risk / sizing
    capital_per_trade_usdc: float = 7000.0
    max_shares_per_trade: float = 20000.0
    min_edge_usdc_per_share: float = 0.003
    min_edge_total_usdc: float = 5.0
    min_shares: float = 5.0
    latency_slippage_buffer_per_leg: float = 0.0015
    ops_cost_per_bundle: float = 0.00
    max_acceptable_combined_cost: float = 0.999
    depth_levels_limit: int = 25

    # Universe / validation
    skip_neg_risk: bool = True
    only_sports: bool = True
    skip_red_flag_rules: bool = True

    # Execution
    execution_enabled: bool = False
    print_books: bool = True
    hedge_on_leg_fail: bool = True
    hedge_price_buffer: float = 0.0020

    # Logs
    csv_log_path: str = "./trades.csv"
    bundle_log_path: str = "./bundles.csv"

    # Connectivity
    poly_host: str = "https://clob.polymarket.com"
    poly_chain_id: int = 137
    poly_private_key: str = ""
    poly_funder: str = ""
    poly_signature_type: int = 0

    # Optional quote stream (off by default)
    enable_websocket_quotes: bool = False
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    max_quote_age_seconds: int = 3

    @property
    def csv_log_path_abs(self) -> Path:
        return Path(self.csv_log_path).expanduser().resolve()

    @property
    def bundle_log_path_abs(self) -> Path:
        return Path(self.bundle_log_path).expanduser().resolve()

    @property
    def is_live_mode(self) -> bool:
        return self.bot_mode.lower() == "live"

    @property
    def is_paper_mode(self) -> bool:
        return self.bot_mode.lower() == "paper"

    @property
    def is_scan_mode(self) -> bool:
        return self.bot_mode.lower() == "scan"


def load_config(dotenv_path: str | None = None) -> BotConfig:
    load_dotenv(dotenv_path)
    return BotConfig(
        bot_mode=os.getenv("BOT_MODE", "scan").strip().lower(),
        scan_interval_seconds=_get_int("SCAN_INTERVAL_SECONDS", 45),
        max_event_pages=_get_int("MAX_EVENT_PAGES", 3),
        events_page_size=_get_int("EVENTS_PAGE_SIZE", 100),
        top_n_opps=_get_int("TOP_N_OPPS", 5),
        capital_per_trade_usdc=_get_float("CAPITAL_PER_TRADE_USDC", 7000.0),
        max_shares_per_trade=_get_float("MAX_SHARES_PER_TRADE", 20000.0),
        min_edge_usdc_per_share=_get_float("MIN_EDGE_USDC_PER_SHARE", 0.003),
        min_edge_total_usdc=_get_float("MIN_EDGE_TOTAL_USDC", 5.0),
        min_shares=_get_float("MIN_SHARES", 5.0),
        latency_slippage_buffer_per_leg=_get_float("LATENCY_SLIPPAGE_BUFFER_PER_LEG", 0.0015),
        ops_cost_per_bundle=_get_float("OPS_COST_PER_BUNDLE", 0.0),
        max_acceptable_combined_cost=_get_float("MAX_ACCEPTABLE_COMBINED_COST", 0.999),
        depth_levels_limit=_get_int("DEPTH_LEVELS_LIMIT", 25),
        skip_neg_risk=_get_bool("SKIP_NEG_RISK", True),
        only_sports=_get_bool("ONLY_SPORTS", True),
        skip_red_flag_rules=_get_bool("SKIP_RED_FLAG_RULES", True),
        execution_enabled=_get_bool("EXECUTION_ENABLED", False),
        print_books=_get_bool("PRINT_BOOKS", True),
        hedge_on_leg_fail=_get_bool("HEDGE_ON_LEG_FAIL", True),
        hedge_price_buffer=_get_float("HEDGE_PRICE_BUFFER", 0.002),
        csv_log_path=os.getenv("CSV_LOG_PATH", "./trades.csv"),
        bundle_log_path=os.getenv("BUNDLE_LOG_PATH", "./bundles.csv"),
        poly_host=os.getenv("POLY_HOST", "https://clob.polymarket.com"),
        poly_chain_id=_get_int("POLY_CHAIN_ID", 137),
        poly_private_key=os.getenv("POLY_PRIVATE_KEY", ""),
        poly_funder=os.getenv("POLY_FUNDER", ""),
        poly_signature_type=_get_int("POLY_SIGNATURE_TYPE", 0),
        enable_websocket_quotes=_get_bool("ENABLE_WEBSOCKET_QUOTES", False),
        ws_url=os.getenv("WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"),
        max_quote_age_seconds=_get_int("MAX_QUOTE_AGE_SECONDS", 3),
    )
