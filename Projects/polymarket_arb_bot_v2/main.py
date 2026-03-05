from __future__ import annotations

import argparse
import json
import logging

from polyarb_bot.bot import PolymarketArbBot
from polyarb_bot.config import load_config
from polyarb_bot.reconcile import summarize_trade_log


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket complementary-outcome arb bot (V2)")
    p.add_argument("command", choices=["scan", "paper", "live", "loop", "reconcile"], nargs="?", default="scan")
    p.add_argument("--env", dest="env_path", default=None, help="Path to .env file")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--csv", dest="csv_path", default=None, help="Trade log CSV path for reconcile")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.env_path)

    if args.command in {"scan", "paper", "live"}:
        cfg.bot_mode = args.command

    if args.command == "reconcile":
        summary = summarize_trade_log(args.csv_path or cfg.csv_log_path)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    bot = PolymarketArbBot(cfg)
    if args.command == "loop":
        bot.run_forever()
        return 0
    bot.run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
