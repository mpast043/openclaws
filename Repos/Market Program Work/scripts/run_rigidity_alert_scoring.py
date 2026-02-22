#!/usr/bin/env python3
"""Baseline predictive scoring for rigidity alerts (start-date based)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover
    yf = None

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import log_returns, rolling_correlation_matrices
from src.spectral import laplacian

SIGMA_BASE_LO = 0.5
SIGMA_BASE_HI = 2.0
SIGMA_CENTER = np.sqrt(SIGMA_BASE_LO * SIGMA_BASE_HI)
TAU_THRESHOLD = 0.4
TAU_GAPLESS = 0.5
TAU_CROSSOVER = 1.2
RED_PERSISTENCE = 2
YELLOW_PERSISTENCE = 3

@dataclass
class Config:
    q_int: float
    sigma_scale: float

@dataclass
class RunParams:
    window: int
    alpha: float


def make_adjacency(C: np.ndarray) -> np.ndarray:
    W = np.maximum(np.abs(C) - TAU_THRESHOLD, 0.0)
    np.fill_diagonal(W, 0.0)
    return 0.5 * (W + W.T)


def tau_series(returns: pd.DataFrame, cfg: Config, run: RunParams, max_windows: int | None) -> pd.DataFrame:
    rows = []
    for idx, (date, C_s) in enumerate(rolling_correlation_matrices(returns, window=run.window, alpha=run.alpha, use_abs=True)):
        if max_windows and idx >= max_windows:
            break
        W = make_adjacency(C_s)
        L = laplacian(W)
        eigs = np.linalg.eigvalsh(L)
        eigs = np.maximum(eigs, 0.0)
        positive = eigs[eigs > 1e-12]
        lambda1 = float(eigs[1]) if len(eigs) > 1 else 0.0
        lambda_int = float(np.quantile(positive, cfg.q_int)) if positive.size else 0.0
        delta_lambda = lambda_int - lambda1
        tau_val = cfg.sigma_scale * SIGMA_CENTER * delta_lambda
        rows.append({"date": date, "tau": tau_val})
    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def detect_alerts(tau_df: pd.DataFrame) -> tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    dates = tau_df["date"].values
    tau = tau_df["tau"].values
    red_start: List[pd.Timestamp] = []
    yellow_start: List[pd.Timestamp] = []
    red_streak = 0
    yellow_streak = 0
    for i, val in enumerate(tau):
        if val < TAU_GAPLESS:
            red_streak += 1
            if red_streak == RED_PERSISTENCE:
                red_start.append(pd.Timestamp(dates[i - RED_PERSISTENCE + 1]))
        else:
            red_streak = 0
        if TAU_GAPLESS <= val < TAU_CROSSOVER:
            yellow_streak += 1
            if yellow_streak == YELLOW_PERSISTENCE:
                yellow_start.append(pd.Timestamp(dates[i - YELLOW_PERSISTENCE + 1]))
        else:
            yellow_streak = 0
    return red_start, yellow_start


def download_series(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        chosen = None
        for key in ["Adj Close", "Close"]:
            if key in level0:
                chosen = key
                break
        if chosen is None:
            chosen = level0[0]
        ser = df.xs(chosen, axis=1, level=0)
        if isinstance(ser, pd.DataFrame):
            if ticker in ser.columns:
                ser = ser[ticker]
            else:
                ser = ser.iloc[:, 0]
    else:
        if "Adj Close" in df.columns:
            ser = df["Adj Close"]
        elif "Close" in df.columns:
            ser = df["Close"]
        else:
            ser = df.iloc[:, 0]
    return ser.rename(ticker)


def build_targets(returns: pd.DataFrame, event_summary: pd.DataFrame, include_drawdowns: bool) -> dict[str, tuple[List[pd.Timestamp], int]]:
    targets: dict[str, tuple[List[pd.Timestamp], int]] = {}
    if include_drawdowns:
        dd10 = event_summary[event_summary["drawdown_threshold_pct"] <= -10]["drawdown_date"].dropna()
        targets["dd10"] = (pd.to_datetime(dd10).to_list(), 60)
        dd20 = event_summary[event_summary["drawdown_threshold_pct"] <= -20]["drawdown_date"].dropna()
        targets["dd20"] = (pd.to_datetime(dd20).to_list(), 120)
    return targets


def add_vix_targets(targets: dict[str, tuple[List[pd.Timestamp], int]], vix_series: pd.Series) -> None:
    aligned = vix_series.sort_index()
    targets["vix30"] = (aligned[aligned >= 30].index.to_list(), 30)
    p90 = aligned.quantile(0.9)
    targets["vixp90"] = (aligned[aligned >= p90].index.to_list(), 30)


def score_alerts(alerts: List[pd.Timestamp], target_dates: List[pd.Timestamp], window_days: int) -> tuple[int, int, float | None, float]:
    hits = 0
    diffs: List[int] = []
    for d in alerts:
        upper = d + pd.Timedelta(days=window_days)
        future = [t for t in target_dates if d <= t <= upper]
        if future:
            hits += 1
            diffs.append((future[0] - d).days)
    total = len(alerts)
    false = total - hits
    lead = float(np.median(diffs)) if diffs else None
    hit_rate = hits / total if total else 0.0
    return hits, false, lead, hit_rate


def main():
    parser = argparse.ArgumentParser(description="Predictive scoring for rigidity alert baselines")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "sp500_prices_lcc.csv")
    parser.add_argument("--config", type=str, required=True, help="q_int,sigma_scale")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--start-date", type=str, default="2002-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "validation")
    parser.add_argument("--include-drawdowns", action="store_true")
    args = parser.parse_args()

    q_val, sigma_scale = [float(x.strip()) for x in args.config.split(",")]
    cfg = Config(q_int=q_val, sigma_scale=sigma_scale)

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")
    returns = returns.loc[args.start_date: args.end_date]

    run = RunParams(window=args.window, alpha=args.alpha)
    tau_df = tau_series(returns, cfg, run, args.max_windows)
    red_alerts, yellow_alerts = detect_alerts(tau_df)

    vix_series = download_series("^VIX", returns.index.min().strftime("%Y-%m-%d"), returns.index.max().strftime("%Y-%m-%d"))
    vix_series = vix_series.reindex(returns.index).ffill()

    event_summary = pd.read_csv(ROOT / "data" / "event_timing_summary.csv", parse_dates=["drawdown_date"])
    targets = build_targets(returns, event_summary, args.include_drawdowns)
    add_vix_targets(targets, vix_series)

    row = {
        "q_int": cfg.q_int,
        "sigma_scale": cfg.sigma_scale,
        "window": args.window,
        "alpha": args.alpha,
        "red_alerts": len(red_alerts),
        "yellow_alerts": len(yellow_alerts),
    }

    for tier_name, alerts in [("red", red_alerts), ("yellow", yellow_alerts)]:
        for target_name, (target_dates, window) in targets.items():
            hits, false, lead, hit_rate = score_alerts(alerts, target_dates, window)
            row[f"{tier_name}_{target_name}_hits"] = hits
            row[f"{tier_name}_{target_name}_false"] = false
            row[f"{tier_name}_{target_name}_hit_rate"] = hit_rate
            row[f"{tier_name}_{target_name}_median_lead"] = lead
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"alert_predictive_q{cfg.q_int}_s{cfg.sigma_scale}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved predictive metrics to {out_path}")


if __name__ == "__main__":
    main()
