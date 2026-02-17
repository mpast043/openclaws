#!/usr/bin/env python3
"""Refined sweep over q_int and sigma scaling with entropy/episode scoring."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import log_returns, rolling_correlation_matrices
from src.spectral import laplacian

SIGMA_BASE_LO = 0.5
SIGMA_BASE_HI = 2.0
SIGMA_CENTER = np.sqrt(SIGMA_BASE_LO * SIGMA_BASE_HI)
C_GEO = 0.5
C_INT_VALUES = (0.0, 1.0)
TAU_THRESHOLD = 0.4
TAU_GAPLESS = 0.5
TAU_CROSSOVER = 1.2

def make_adjacency(C: np.ndarray) -> np.ndarray:
    W = np.maximum(np.abs(C) - TAU_THRESHOLD, 0.0)
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


def evaluate_tau(eigenvalues: np.ndarray, q_int: float, sigma_scale: float) -> float:
    geo_filter = PowerLawFilter(d0=2.0, beta_max=1.5, gamma=0.85, lambda0_quantile=0.6)
    geo_weights = geo_filter.apply(eigenvalues, C_GEO)
    lambda1 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    positive = eigenvalues[eigenvalues > 1e-12]
    lambda_int = float(np.quantile(positive, q_int)) if positive.size else 0.0
    delta_lambda = lambda_int - lambda1
    tau = sigma_scale * SIGMA_CENTER * delta_lambda
    return tau


def build_indicator(returns: pd.DataFrame, window: int, alpha: float, q_int: float, sigma_scale: float, max_windows: int) -> pd.DataFrame:
    rows = []
    for idx, (date, C_s) in enumerate(rolling_correlation_matrices(returns, window=window, alpha=alpha, use_abs=True)):
        if max_windows and idx >= max_windows:
            break
        W = make_adjacency(C_s)
        L = laplacian(W)
        eigs = np.linalg.eigvalsh(L)
        eigs = np.maximum(eigs, 0.0)
        tau = evaluate_tau(eigs, q_int, sigma_scale)
        rows.append({"date": date, "tau": tau})
    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    return df


def compute_entropy(frac_gapless: float, frac_crossover: float, frac_rigid: float) -> float:
    probs = [frac_gapless, frac_crossover, frac_rigid]
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return float(-np.sum([p * np.log(p) for p in probs]))


def persistence_counts(regimes: np.ndarray, target: str, k: int) -> int:
    count = 0
    streak = 0
    for val in regimes:
        if val == target:
            streak += 1
            if streak == k:
                count += 1
        else:
            streak = 0
    return count


def summarize(df: pd.DataFrame, q_int: float, sigma_scale: float, lcc_fraction: float) -> dict:
    tau_vals = df["tau"].values
    regimes = np.select(
        [tau_vals < TAU_GAPLESS, tau_vals < TAU_CROSSOVER],
        ["gapless", "crossover"],
        default="rigid",
    )
    frac_gapless = float(np.mean(regimes == "gapless"))
    frac_crossover = float(np.mean(regimes == "crossover"))
    frac_rigid = float(np.mean(regimes == "rigid"))
    entropy = compute_entropy(frac_gapless, frac_crossover, frac_rigid)
    k1 = persistence_counts(regimes, "crossover", 1)
    k2 = persistence_counts(regimes, "crossover", 2)
    k3 = persistence_counts(regimes, "crossover", 3)

    return {
        "q_int": q_int,
        "sigma_scale": sigma_scale,
        "tau_p10": float(np.percentile(tau_vals, 10)),
        "tau_median": float(np.percentile(tau_vals, 50)),
        "tau_p90": float(np.percentile(tau_vals, 90)),
        "frac_gapless": frac_gapless,
        "frac_crossover": frac_crossover,
        "frac_rigid": frac_rigid,
        "entropy": entropy,
        "critical_k1": k1,
        "critical_k2": k2,
        "critical_k3": k3,
        "lcc_fraction_median": lcc_fraction,
    }


def main():
    parser = argparse.ArgumentParser(description="Refined q_int / sigma sweep")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "sp500_prices_lcc.csv")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--q-values", type=str, default="0.80,0.85,0.90,0.95")
    parser.add_argument("--sigma-scales", type=str, default="0.15,0.20,0.25,0.33,0.40,0.50")
    parser.add_argument("--max-windows", type=int, default=800)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "validation" / "q_sigma_sweep_refined.csv")
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")
    q_vals = [float(v.strip()) for v in args.q_values.split(",") if v.strip()]
    sigma_scales = [float(v.strip()) for v in args.sigma_scales.split(",") if v.strip()]

    records = []
    for q, s in product(q_vals, sigma_scales):
        print(f"Running sweep for q={q}, sigma_scale={s}")
        df = build_indicator(returns, args.window, args.alpha, q, s, args.max_windows)
        summary = summarize(df, q, s, 1.0)
        records.append(summary)

    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved sweep to {args.output}")


if __name__ == "__main__":
    main()
