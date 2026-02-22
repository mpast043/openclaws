#!/usr/bin/env python3
"""Null control tests for the rigidity indicator (tau-based)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import (
    log_returns,
    rolling_correlation_matrices,
)
from src.spectral import laplacian

SIGMA_LO = 0.5
SIGMA_HI = 2.0
SIGMA_CENTER = np.sqrt(SIGMA_LO * SIGMA_HI)
Q_LIST = [0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01]
C_GEO = 0.5
C_INT_VALUES = (0.0, 1.0)
TAU_RIGID = 5.0
TAU_CRITICAL = 1.0
TAU_GAPLESS = 0.5
TAU_THRESHOLD = 0.4


def make_adjacency(C: np.ndarray) -> np.ndarray:
    base = np.abs(C)
    W = np.maximum(base - TAU_THRESHOLD, 0.0)
    np.fill_diagonal(W, 0.0)
    return 0.5 * (W + W.T)


class PowerLawFilter:
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
        ratio = 1.0 + eigenvalues / lam0
        if beta < 1e-15:
            weights = np.ones_like(eigenvalues)
        else:
            weights = np.power(ratio, -beta)
        return np.clip(weights, 0.0, 1.0)


def filtered_log_return_probability(eigenvalues: np.ndarray, weights: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    log_weights = np.full_like(eigenvalues, -np.inf, dtype=float)
    mask = weights > 1e-300
    log_weights[mask] = np.log(weights[mask])
    log_summands = -sigma_values[:, None] * eigenvalues[None, :] + log_weights[None, :]
    active = log_summands[:, mask]
    max_val = np.max(active, axis=1, keepdims=True)
    log_sum = max_val[:, 0] + np.log(np.sum(np.exp(active - max_val), axis=1))
    return log_sum - np.log(len(eigenvalues))


def filtered_spectral_dimension(eigenvalues: np.ndarray, weights: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    ln_P = filtered_log_return_probability(eigenvalues, weights, sigma_values)
    ln_sigma = np.log(sigma_values)
    ds = np.full_like(sigma_values, np.nan, dtype=float)
    for i in range(1, len(sigma_values) - 1):
        d_lnP = (ln_P[i + 1] - ln_P[i - 1]) / (ln_sigma[i + 1] - ln_sigma[i - 1])
        ds[i] = -2.0 * d_lnP
    return ds


def interaction_weights(eigenvalues: np.ndarray, lambda_thresh: float, lambda_max: float, C_int: float) -> np.ndarray:
    weights = np.ones_like(eigenvalues)
    mask = eigenvalues >= lambda_thresh
    if np.any(mask):
        span = lambda_max - lambda_thresh + 1e-12
        weights[mask] = 1.0 + C_int * ((eigenvalues[mask] - lambda_thresh) / span)
    return weights


def evaluate_eigenvalues(eigenvalues: np.ndarray, sigma_values: np.ndarray) -> dict:
    geo_filter = PowerLawFilter(d0=2.0, beta_max=1.5, gamma=0.85, lambda0_quantile=0.6)
    geo_weights = geo_filter.apply(eigenvalues, C_GEO)
    lambda1 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    best_tau = None
    best_delta_lambda = None

    for q in Q_LIST:
        positive = eigenvalues[eigenvalues > 1e-12]
        lambda_int = float(np.quantile(positive, q)) if positive.size else 0.0
        delta_lambda = lambda_int - lambda1
        tau = SIGMA_CENTER * delta_lambda
        lambda_max = float(np.max(eigenvalues)) if eigenvalues.size else 0.0

        for C_int in C_INT_VALUES:
            interaction_weights(eigenvalues, lambda_int, lambda_max, C_int)

        best_tau = tau
        best_delta_lambda = delta_lambda
        if tau < TAU_GAPLESS:
            break

    if best_tau is None:
        best_tau = 0.0
    if best_delta_lambda is None:
        best_delta_lambda = 0.0
    if best_tau >= TAU_RIGID:
        regime = "Rigid"
    elif best_tau >= TAU_CRITICAL:
        regime = "Critical"
    else:
        regime = "Gapless"

    return {
        "lambda1": lambda1,
        "delta_lambda": best_delta_lambda,
        "tau": best_tau,
        "regime": regime,
    }


def build_indicator(returns: pd.DataFrame, cfg_window: int, cfg_alpha: float, sigma_values: np.ndarray, max_windows: int) -> pd.DataFrame:
    rows = []
    for idx, (date, C_s) in enumerate(rolling_correlation_matrices(returns, window=cfg_window, alpha=cfg_alpha, use_abs=True)):
        if max_windows and idx >= max_windows:
            break
        W = make_adjacency(C_s)
        L = laplacian(W)
        eigs = np.linalg.eigvalsh(L)
        eigs = np.maximum(eigs, 0.0)
        metrics = evaluate_eigenvalues(eigs, sigma_values)
        metrics.update({"date": date})
        rows.append(metrics)
    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    return df


def summarize_indicator(df: pd.DataFrame) -> dict:
    regime_counts = df["regime"].value_counts().to_dict()
    transitions = int((df["regime"] != df["regime"].shift(1)).sum())
    tau_stats = df["tau"].describe()
    return {
        "rigid_days": regime_counts.get("Rigid", 0),
        "critical_days": regime_counts.get("Critical", 0),
        "gapless_days": regime_counts.get("Gapless", 0),
        "transitions": transitions,
        "tau_min": tau_stats["min"],
        "tau_median": tau_stats["50%"],
        "tau_max": tau_stats["max"],
    }


def time_shuffle_returns(returns: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = returns.copy()
    for col in shuffled.columns:
        values = shuffled[col].values.copy()
        rng.shuffle(values)
        shuffled[col] = values
    return shuffled


def permute_edges(C: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    triu_idx = np.triu_indices_from(C, k=1)
    values = C[triu_idx]
    rng.shuffle(values)
    C_perm = np.zeros_like(C)
    C_perm[triu_idx] = values
    C_perm = C_perm + C_perm.T
    np.fill_diagonal(C_perm, 1.0)
    return C_perm


def main():
    parser = argparse.ArgumentParser(description="Rigidity null tests")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "sp500_prices_lcc.csv")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "validation" / "null_tests.csv")
    parser.add_argument("--max-windows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")
    sigma_values = np.geomspace(SIGMA_LO / 2.0, SIGMA_HI * 2.0, max(200, returns.shape[1]))

    print("Running baseline indicator on original data")
    baseline = build_indicator(returns, args.window, args.alpha, sigma_values, args.max_windows)
    baseline_summary = summarize_indicator(baseline)
    baseline_summary["test"] = "baseline"

    print("Running time-shuffled null")
    shuffled = time_shuffle_returns(returns, seed=args.seed)
    shuffled_df = build_indicator(shuffled, args.window, args.alpha, sigma_values, args.max_windows)
    shuffled_summary = summarize_indicator(shuffled_df)
    shuffled_summary["test"] = "time_shuffle"

    print("Running edge-permutation null")
    rows = []
    for idx, (date, C_s) in enumerate(rolling_correlation_matrices(returns, window=args.window, alpha=args.alpha, use_abs=True)):
        if idx >= args.max_windows:
            break
        C_perm = permute_edges(C_s, seed=args.seed + idx)
        W = make_adjacency(C_perm)
        L = laplacian(W)
        eigs = np.linalg.eigvalsh(L)
        eigs = np.maximum(eigs, 0.0)
        metrics = evaluate_eigenvalues(eigs, sigma_values)
        metrics.update({"date": date})
        rows.append(metrics)
    perm_df = pd.DataFrame(rows)
    perm_summary = summarize_indicator(perm_df)
    perm_summary["test"] = "edge_permutation"

    result = pd.DataFrame([baseline_summary, shuffled_summary, perm_summary])
    result.to_csv(args.output, index=False)
    print(f"Saved null test results to {args.output}")


if __name__ == "__main__":
    main()
