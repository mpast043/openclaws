#!/usr/bin/env python3
"""Start-date stability check for rigidity alerts (Red + Yellow)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import log_returns, rolling_correlation_matrices
from src.spectral import laplacian

# Thresholds and persistence
SIGMA_BASE_LO = 0.5
SIGMA_BASE_HI = 2.0
SIGMA_CENTER = np.sqrt(SIGMA_BASE_LO * SIGMA_BASE_HI)
TAU_THRESHOLD = 0.4
TAU_GAPLESS = 0.5
TAU_CROSSOVER = 1.2
RED_PERSISTENCE = 2
YELLOW_PERSISTENCE = 3
ALERT_TOLERANCE_DAYS = 2

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


def evaluate_tau_series(returns: pd.DataFrame, cfg: Config, run: RunParams, max_windows: int | None) -> pd.DataFrame:
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
        tau = cfg.sigma_scale * SIGMA_CENTER * delta_lambda
        rows.append({"date": date, "tau": tau})
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


def match_alerts(base: List[pd.Timestamp], pert: List[pd.Timestamp]) -> Tuple[int, List[int], int, int]:
    matched_diffs: List[int] = []
    used = set()
    matches = 0
    for d in base:
        candidates = [j for j, p in enumerate(pert) if j not in used and abs((p - d).days) <= ALERT_TOLERANCE_DAYS]
        if candidates:
            j = candidates[0]
            used.add(j)
            matches += 1
            matched_diffs.append(abs((pert[j] - d).days))
    base_unmatched = len(base) - matches
    pert_unmatched = len(pert) - matches
    return matches, matched_diffs, base_unmatched, pert_unmatched


def overlap_metrics(base: List[pd.Timestamp], pert: List[pd.Timestamp]) -> dict:
    matches, diffs, base_unmatched, pert_unmatched = match_alerts(base, pert)
    union = len(base) + len(pert) - matches
    jaccard = matches / union if union > 0 else 1.0
    median_shift = float(np.median(diffs)) if diffs else np.nan
    return {
        "overlap": jaccard,
        "median_shift": median_shift,
        "base_unmatched": base_unmatched,
        "pert_unmatched": pert_unmatched,
    }


def main():
    parser = argparse.ArgumentParser(description="Start-date overlap stability for rigidity alerts")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "sp500_prices_lcc.csv")
    parser.add_argument("--config", type=str, required=True, help="q_int,sigma_scale")
    parser.add_argument("--windows", type=str, default="40,45")
    parser.add_argument("--alphas", type=str, default="0.1,0.12")
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--start-date", type=str, default="2002-01-01")
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "validation")
    args = parser.parse_args()

    q_val, sigma_scale = [float(x.strip()) for x in args.config.split(",")]
    cfg = Config(q_int=q_val, sigma_scale=sigma_scale)

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")
    returns = returns.loc[args.start_date: args.end_date]

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    baseline_run = RunParams(window=windows[0], alpha=alphas[0])
    baseline_df = evaluate_tau_series(returns, cfg, baseline_run, args.max_windows)
    base_red, base_yellow = detect_alerts(baseline_df)

    records = []

    for w in windows:
        for a in alphas:
            run = RunParams(window=w, alpha=a)
            tau_df = evaluate_tau_series(returns, cfg, run, args.max_windows)
            red_alerts, yellow_alerts = detect_alerts(tau_df)

            if run == baseline_run:
                records.append({
                    "q_int": cfg.q_int,
                    "sigma_scale": cfg.sigma_scale,
                    "window": w,
                    "alpha": a,
                    "type": "baseline",
                    "red_alerts": len(red_alerts),
                    "yellow_alerts": len(yellow_alerts),
                })
                continue

            red_metrics = overlap_metrics(base_red, red_alerts)
            yellow_metrics = overlap_metrics(base_yellow, yellow_alerts)

            row = {
                "q_int": cfg.q_int,
                "sigma_scale": cfg.sigma_scale,
                "window": w,
                "alpha": a,
                "type": "perturbed",
                "red_overlap": red_metrics["overlap"],
                "red_median_shift": red_metrics["median_shift"],
                "red_base_unmatched": red_metrics["base_unmatched"],
                "red_pert_unmatched": red_metrics["pert_unmatched"],
                "yellow_overlap": yellow_metrics["overlap"],
                "yellow_median_shift": yellow_metrics["median_shift"],
                "yellow_base_unmatched": yellow_metrics["base_unmatched"],
                "yellow_pert_unmatched": yellow_metrics["pert_unmatched"],
            }
            records.append(row)

    result = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"alert_start_overlap_q{cfg.q_int}_s{cfg.sigma_scale}.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved start-date overlap results to {out_path}")


if __name__ == "__main__":
    main()
