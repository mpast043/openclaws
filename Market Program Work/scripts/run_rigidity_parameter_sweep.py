#!/usr/bin/env python3
"""Parameter sweep for the rigidity indicator (tau-based)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
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

@dataclass
class SweepConfig:
    window: int
    alpha: float
    adjacency: str
    laplacian_type: str


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    d = np.sum(W, axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt)
    I = np.eye(W.shape[0])
    return I - D_inv_sqrt @ W @ D_inv_sqrt


def make_adjacency(C: np.ndarray, use_abs: bool) -> np.ndarray:
    tau = TAU_THRESHOLD
    if use_abs:
        base = np.abs(C)
    else:
        base = np.clip(C, 0.0, None)
    W = np.maximum(base - tau, 0.0)
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
    stop_q = Q_LIST[-1]

    for q in Q_LIST:
        positive = eigenvalues[eigenvalues > 1e-12]
        lambda_int = float(np.quantile(positive, q)) if positive.size else 0.0
        delta_lambda = lambda_int - lambda1
        tau = SIGMA_CENTER * delta_lambda
        lambda_max = float(np.max(eigenvalues)) if eigenvalues.size else 0.0

        geo_ds = []
        for C_int in C_INT_VALUES:
            int_weights = interaction_weights(eigenvalues, lambda_int, lambda_max, C_int)
            total_weights = geo_weights * int_weights
            ds_curve = filtered_spectral_dimension(eigenvalues, total_weights, sigma_values)
            geo_ds.append(ds_curve)

        best_tau = tau
        best_delta_lambda = delta_lambda
        stop_q = q
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
        "q_stop": stop_q,
        "regime": regime,
    }


def build_indicator(prices_path: Path, cfg: SweepConfig, max_windows: int) -> pd.DataFrame:
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")

    rows = []
    sigma_values = np.geomspace(SIGMA_LO / 2.0, SIGMA_HI * 2.0, max(200, returns.shape[1]))

    for idx, (date, C_s) in enumerate(
        rolling_correlation_matrices(returns, window=cfg.window, alpha=cfg.alpha, use_abs=True)
    ):
        if max_windows and idx >= max_windows:
            break
        W = make_adjacency(C_s, use_abs=(cfg.adjacency == "abs"))
        if cfg.laplacian_type == "normalized":
            L = normalized_laplacian(W)
        else:
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


def parse_list(values: str, cast_type):
    return [cast_type(v.strip()) for v in values.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Rigidity parameter sweep")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "sp500_prices_lcc.csv")
    parser.add_argument("--windows", type=str, default="40,60")
    parser.add_argument("--alphas", type=str, default="0.0,0.05")
    parser.add_argument("--adjacency", type=str, default="abs,signed")
    parser.add_argument("--laplacians", type=str, default="combinatorial,normalized")
    parser.add_argument("--max-combos", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "validation" / "parameter_sweep.csv")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    windows = parse_list(args.windows, int)
    alphas = parse_list(args.alphas, float)
    adjacency_modes = parse_list(args.adjacency, str)
    laplacian_modes = parse_list(args.laplacians, str)

    records = []
    combos = list(product(windows, alphas, adjacency_modes, laplacian_modes))
    if args.max_combos:
        combos = combos[: args.max_combos]

    for window, alpha, adjacency_mode, laplacian_mode in combos:
        cfg = SweepConfig(window=window, alpha=alpha, adjacency=adjacency_mode, laplacian_type=laplacian_mode)
        print(f"Running sweep for window={window}, alpha={alpha}, adjacency={adjacency_mode}, laplacian={laplacian_mode}")
        df = build_indicator(args.prices, cfg, max_windows=args.max_windows)
        summary = summarize_indicator(df)
        summary.update({
            "window": window,
            "alpha": alpha,
            "adjacency": adjacency_mode,
            "laplacian": laplacian_mode,
        })
        records.append(summary)

    result = pd.DataFrame(records)
    result.to_csv(args.output, index=False)
    print(f"Saved sweep results to {args.output}")


if __name__ == "__main__":
    main()
