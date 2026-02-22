#!/usr/bin/env python3
"""Export per-alert audit rows (hits, leads, context) for adaptive q-stop regimes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover
    yf = None


def load_series(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is required for this audit export")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        ser = data.xs("Close", axis=1, level=0)
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
    else:
        close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        ser = data[close_col]
    ser.name = ticker
    return ser


def build_targets(vix: pd.Series, summary_path: Path, vix_threshold: float, enforce_precondition: bool = False) -> dict[str, tuple[pd.DatetimeIndex, int, str]]:
    if enforce_precondition:
        shifted = vix.shift(1)
        mask = vix >= vix_threshold
        if bool(mask.any()):
            mask = mask & (shifted < vix_threshold)
        vix_dates = vix[mask].index
        vix_label = f"vix_cross{int(vix_threshold)}"
    else:
        vix_dates = vix[vix >= vix_threshold].index
        vix_label = f"vix_ge{int(vix_threshold)}"
    summary = pd.read_csv(summary_path, parse_dates=["drawdown_date"])
    dd10 = summary[summary["drawdown_threshold_pct"] <= -10]["drawdown_date"].dropna()
    dd20 = summary[summary["drawdown_threshold_pct"] <= -20]["drawdown_date"].dropna()
    return {
        vix_label: (pd.DatetimeIndex(vix_dates), 30, "vix"),
        "dd10": (pd.DatetimeIndex(dd10), 60, "drawdown"),
        "dd20": (pd.DatetimeIndex(dd20), 120, "drawdown"),
    }


def find_alerts(qstop: pd.DataFrame, tiers: Iterable[str]) -> list[dict]:
    qstop = qstop.sort_values("date").copy()
    qstop["prev_regime"] = qstop["regime"].shift(1)
    mask = qstop["regime"] != qstop["prev_regime"]
    alerts = []
    for _, row in qstop[mask].iterrows():
        regime = str(row["regime"])
        if regime not in tiers:
            continue
        alerts.append({
            "date": pd.Timestamp(row["date"]),
            "tier": regime.lower(),
            "regime": regime,
            "q_stop": row.get("q_stop"),
            "stop_reason": row.get("stop_reason", row.get("reason", "")),
        })
    return alerts


def find_event(alert_date: pd.Timestamp, target_dates: pd.DatetimeIndex, window_days: int) -> tuple[int, pd.Timestamp | pd.NaT, float | float("nan")]:
    start = alert_date + pd.Timedelta(days=1)
    end = alert_date + pd.Timedelta(days=window_days)
    mask = (target_dates >= start) & (target_dates <= end)
    if mask.any():
        event_date = target_dates[mask].min()
        lead = float((event_date - alert_date).days)
        return 1, event_date, lead
    return 0, pd.NaT, float("nan")


def forward_stats(alert_date: pd.Timestamp, prices: pd.Series, returns: pd.Series, window_days: int) -> tuple[float | float("nan"), float | float("nan")]:
    if alert_date not in prices.index:
        return float("nan"), float("nan")
    window_end = alert_date + pd.Timedelta(days=window_days)
    window_idx = prices.index[(prices.index > alert_date) & (prices.index <= window_end)]
    if len(window_idx) == 0:
        return float("nan"), float("nan")
    price0 = float(prices.loc[alert_date])
    future_prices = prices.loc[window_idx]
    min_price = float(future_prices.min())
    max_drawdown = (min_price - price0) / price0
    min_return = float(returns.reindex(window_idx).min())
    return max_drawdown, min_return


def main():
    parser = argparse.ArgumentParser(description="Export alert-by-alert audit rows for adaptive q-stop signals")
    parser.add_argument("--qstop", type=Path, required=True, help="Path to q_stop history CSV")
    parser.add_argument("--event-summary", type=Path, default=Path("data/event_timing_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/alert_audit_matches.csv"))
    parser.add_argument("--index-ticker", type=str, default="^GSPC")
    parser.add_argument("--vix-ticker", type=str, default="^VIX")
    parser.add_argument("--vix-threshold", type=float, default=30.0)
    parser.add_argument("--tiers", type=str, default="Critical,Rigid", help="Comma-separated regimes to treat as alert tiers")
    parser.add_argument("--vix-precondition", action="store_true", help="Require VIX to be below threshold before counting a crossing")
    args = parser.parse_args()

    qstop_df = pd.read_csv(args.qstop, parse_dates=["date"])
    alerts = find_alerts(qstop_df, [t.strip() for t in args.tiers.split(",") if t.strip()])
    if not alerts:
        raise RuntimeError("No alerts found for specified tiers")

    start = qstop_df["date"].min().strftime("%Y-%m-%d")
    end = qstop_df["date"].max().strftime("%Y-%m-%d")

    vix_series = load_series(args.vix_ticker, start, end)
    index_series = load_series(args.index_ticker, start, end)

    full_index = pd.date_range(qstop_df["date"].min(), qstop_df["date"].max(), freq="D")
    vix_series = vix_series.reindex(full_index).ffill()
    index_series = index_series.reindex(full_index).ffill()
    index_returns = index_series.pct_change()

    targets = build_targets(vix_series, args.event_summary, args.vix_threshold, enforce_precondition=args.vix_precondition)

    rows = []
    for alert in alerts:
        alert_date = alert["date"]
        vix_value = float(vix_series.loc[alert_date]) if alert_date in vix_series.index else float("nan")
        for target_name, (target_dates, window_days, target_kind) in targets.items():
            hit, event_date, lead = find_event(alert_date, target_dates, window_days)
            max_dd, min_ret = forward_stats(alert_date, index_series, index_returns, window_days)
            rows.append({
                "alert_date": alert_date,
                "tier": alert["tier"],
                "regime_at_alert": alert["regime"],
                "q_stop": alert["q_stop"],
                "stop_reason": alert["stop_reason"],
                "target_name": target_name,
                "target_kind": target_kind,
                "target_window_days": window_days,
                "hit": hit,
                "event_date": event_date,
                "lead_days": lead,
                "vix_value": vix_value,
                "forward_max_drawdown": max_dd,
                "forward_min_return": min_ret,
            })

    audit_df = pd.DataFrame(rows)
    audit_df.sort_values(["tier", "target_name", "alert_date"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(args.output, index=False)
    print(f"Audit rows written to {args.output} ({len(audit_df)} rows)")


if __name__ == "__main__":
    main()
