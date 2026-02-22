#!/usr/bin/env python3
"""Episode-level rigidity alert stability + predictive scoring."""
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

# Constants
SIGMA_BASE_LO = 0.5
SIGMA_BASE_HI = 2.0
SIGMA_CENTER = np.sqrt(SIGMA_BASE_LO * SIGMA_BASE_HI)
TAU_THRESHOLD = 0.4

# Hysteresis thresholds
TAU_GAPLESS_ENTER = 0.5
TAU_GAPLESS_EXIT = 0.7
TAU_REDLITE_ENTER = 0.7
TAU_REDLITE_EXIT = 0.9
TAU_YELLOW_ENTER = 1.0
TAU_YELLOW_EXIT = 1.3

# Persistence lengths
RED_PERSISTENCE = 2
REDLITE_PERSISTENCE = 2
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


class PowerLawFilter:
    """Placeholder spectral filter (kept for parity with earlier scripts)."""

    def __init__(self, d0: float, beta_max: float | None = None, gamma: float = 1.0, lambda0_quantile: float = 0.6):
        self.d0 = d0
        self.beta_max = beta_max if beta_max is not None else (d0 / 2.0)
        self.gamma = max(0.1, gamma)
        self.lambda0_quantile = np.clip(lambda0_quantile, 0.0, 1.0)
        self._lambda0: float | None = None

    def _ensure_lambda0(self, eigenvalues: np.ndarray) -> float:
        if self._lambda0 is not None:
            return self._lambda0
        nonzero = eigenvalues[eigenvalues > 1e-12]
        if nonzero.size == 0:
            self._lambda0 = 1.0
        else:
            self._lambda0 = float(np.quantile(nonzero, self.lambda0_quantile))
        return self._lambda0

    def apply(self, eigenvalues: np.ndarray, C: float) -> np.ndarray:
        lam0 = self._ensure_lambda0(eigenvalues)
        beta = self.beta_max * (1.0 - float(C) ** self.gamma)
        if beta < 1e-12:
            return np.ones_like(eigenvalues)
        ratio = 1.0 + eigenvalues / lam0
        return np.power(ratio, -beta)


def tau_series(returns: pd.DataFrame, cfg: Config, run: RunParams, max_windows: int | None = None) -> pd.DataFrame:
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


def _extract_episodes(tau_df: pd.DataFrame, enter: float, exit: float, persistence: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = tau_df["date"].values
    tau = tau_df["tau"].values
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    state = False
    enter_streak = 0
    exit_streak = 0
    start_idx: int | None = None

    for i, val in enumerate(tau):
        if not state:
            if val < enter:
                enter_streak += 1
                if enter_streak >= persistence:
                    state = True
                    start_idx = i - persistence + 1
                    enter_streak = 0
                    exit_streak = 0
            else:
                enter_streak = 0
        else:
            if val > exit:
                exit_streak += 1
                if exit_streak >= persistence:
                    end_idx = max(i - persistence, start_idx if start_idx is not None else i)
                    episodes.append((pd.Timestamp(dates[start_idx]), pd.Timestamp(dates[end_idx])))
                    state = False
                    start_idx = None
                    exit_streak = 0
                    enter_streak = 0
            else:
                exit_streak = 0
    if state and start_idx is not None:
        episodes.append((pd.Timestamp(dates[start_idx]), pd.Timestamp(dates[-1])))
    return episodes


def build_episodes(tau_df: pd.DataFrame) -> dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    return {
        "yellow": _extract_episodes(tau_df, TAU_YELLOW_ENTER, TAU_YELLOW_EXIT, YELLOW_PERSISTENCE),
        "redlite": _extract_episodes(tau_df, TAU_REDLITE_ENTER, TAU_REDLITE_EXIT, REDLITE_PERSISTENCE),
        "red": _extract_episodes(tau_df, TAU_GAPLESS_ENTER, TAU_GAPLESS_EXIT, RED_PERSISTENCE),
    }


def _days_covered(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> set:
    days = set()
    for start, end in intervals:
        rng = pd.date_range(start, end, freq="D")
        days.update(rng.to_list())
    return days


def interval_metrics(base_eps: List[Tuple[pd.Timestamp, pd.Timestamp]], pert_eps: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> tuple[float, float, float, float]:
    base_days = _days_covered(base_eps)
    pert_days = _days_covered(pert_eps)
    intersection = len(base_days & pert_days)
    union = len(base_days | pert_days)
    iou = intersection / union if union > 0 else 1.0
    recall = intersection / len(base_days) if base_days else 1.0
    precision = intersection / len(pert_days) if pert_days else 1.0
    base_midpoints = [(s + (e - s) / 2) for s, e in base_eps]
    pert_midpoints = [(s + (e - s) / 2) for s, e in pert_eps]
    shifts = []
    if base_midpoints and pert_midpoints:
        pert_nums = np.array([p.value for p in pert_midpoints], dtype=float)
        for b in base_midpoints:
            idx = np.argmin(np.abs(pert_nums - b.value))
            shifts.append(abs((pd.Timestamp(pert_midpoints[idx]) - b).days))
    median_shift = float(np.median(shifts)) if shifts else np.nan
    return iou, recall, precision, median_shift


def compare_episodes(base: dict, pert: dict) -> dict:
    out = {}
    for tier in ["yellow", "redlite", "red"]:
        iou, recall, precision, shift = interval_metrics(base[tier], pert[tier])
        out[f"{tier}_iou"] = iou
        out[f"{tier}_recall"] = recall
        out[f"{tier}_precision"] = precision
        out[f"{tier}_median_mid_shift"] = shift
    return out


def predictive_scores(episodes: dict, targets: dict[str, tuple[List[pd.Timestamp], int]]) -> dict:
    def centers(ep_list):
        return [start + (end - start) / 2 for start, end in ep_list]

    metrics = {}
    for tier in ["yellow", "redlite"]:
        centers_list = centers(episodes[tier])
        total = len(centers_list)
        for target_name, (target_dates, window) in targets.items():
            hits = 0
            leads = []
            for c in centers_list:
                future = [t for t in target_dates if c <= t <= c + pd.Timedelta(days=window)]
                if future:
                    hits += 1
                    leads.append((future[0] - c).days)
            metrics[f"{tier}_{target_name}_hits"] = hits
            metrics[f"{tier}_{target_name}_total"] = total
            metrics[f"{tier}_{target_name}_hit_rate"] = hits / total if total else 0.0
            metrics[f"{tier}_{target_name}_median_lead"] = float(np.median(leads)) if leads else np.nan
            metrics[f"{tier}_{target_name}_false_alarms"] = total - hits
    return metrics


def download_series(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance not available")
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


def build_targets(returns: pd.DataFrame, vix: pd.Series) -> dict[str, tuple[List[pd.Timestamp], int]]:
    aligned_vix = vix.reindex(returns.index).ffill()
    vix30_dates = aligned_vix[aligned_vix >= 30].index.to_list()
    return {"vix30": (vix30_dates, 30)}


def summarize_baseline_eps(eps: dict) -> dict:
    summary = {}
    for tier in ["yellow", "redlite", "red"]:
        durations = [(end - start).days for start, end in eps[tier]]
        summary[f"{tier}_episodes"] = len(eps[tier])
        summary[f"{tier}_median_duration"] = float(np.median(durations)) if durations else np.nan
    return summary


def main():
    parser = argparse.ArgumentParser(description="Rigidity episode stability + predictive scoring")
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

    vix_series = download_series("^VIX", returns.index.min().strftime("%Y-%m-%d"), returns.index.max().strftime("%Y-%m-%d"))
    targets = build_targets(returns, vix_series)

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    baseline_run = RunParams(window=windows[0], alpha=alphas[0])
    baseline_tau = tau_series(returns, cfg, baseline_run, args.max_windows)
    baseline_eps = build_episodes(baseline_tau)

    records = []

    for w in windows:
        for a in alphas:
            run = RunParams(window=w, alpha=a)
            tau_df = tau_series(returns, cfg, run, args.max_windows)
            eps = build_episodes(tau_df)

            if run == baseline_run:
                row = {
                    "q_int": cfg.q_int,
                    "sigma_scale": cfg.sigma_scale,
                    "window": w,
                    "alpha": a,
                    "type": "baseline",
                }
                row.update(summarize_baseline_eps(eps))
                row.update(predictive_scores(eps, targets))
                records.append(row)
            else:
                row = {
                    "q_int": cfg.q_int,
                    "sigma_scale": cfg.sigma_scale,
                    "window": w,
                    "alpha": a,
                    "type": "perturbed",
                }
                row.update(compare_episodes(baseline_eps, eps))
                records.append(row)

    result = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"alert_stability_q{cfg.q_int}_s{cfg.sigma_scale}.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved alert stability results to {out_path}")


if __name__ == "__main__":
    main()
