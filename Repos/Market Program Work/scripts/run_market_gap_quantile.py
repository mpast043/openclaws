#!/usr/bin/env python3
"""Gap-collapse diagnostic on correlation graphs with connectivity report and scale-aware window."""
from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import log_returns, shrink_correlation, correlation_to_adjacency
from src.spectral import laplacian

DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "gap_quantile"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_LO = 0.5
SIGMA_HI = 2.0
SIGMA_POINTS = 600
SIGMA_GRID = np.geomspace(SIGMA_LO / 2.0, SIGMA_HI * 2.0, SIGMA_POINTS)
SIGMA_WINDOW_MASK = (SIGMA_GRID >= SIGMA_LO) & (SIGMA_GRID <= SIGMA_HI)
SIGMA_WINDOW = SIGMA_GRID[SIGMA_WINDOW_MASK]
SIGMA_CENTER = np.sqrt(SIGMA_LO * SIGMA_HI)  # ≈ 1.0

TAU_RIGID = 5.0
TAU_CROSSOVER = 0.5
TAU_STOP = 0.2

Q_LIST = [0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01]
C_GEO = 0.5
C_INT_VALUES = (0.0, 1.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


def connected_components(adj_bool: np.ndarray) -> List[List[int]]:
    n = adj_bool.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        comp = []
        visited[start] = True
        while queue:
            node = queue.popleft()
            comp.append(node)
            neighbors = np.where(adj_bool[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        comps.append(comp)
    return comps


def classify_regime(tau: float) -> str:
    if tau >= TAU_RIGID:
        return "Rigid"
    if tau >= TAU_CROSSOVER:
        return "Crossover"
    return "Gapless"


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
    return ds, ln_P


def interaction_weights(eigenvalues: np.ndarray, lambda_thresh: float, lambda_max: float, C_int: float) -> np.ndarray:
    weights = np.ones_like(eigenvalues)
    mask = eigenvalues >= lambda_thresh
    if np.any(mask):
        span = lambda_max - lambda_thresh + 1e-12
        weights[mask] = 1.0 + C_int * ((eigenvalues[mask] - lambda_thresh) / span)
    return weights


def load_market_graph(prices_path: Path, window: int, alpha: float):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any", axis=0)
    block = returns.tail(window).values
    C = np.corrcoef(block.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)
    C_s = shrink_correlation(C, alpha=alpha)
    W = correlation_to_adjacency(C_s)
    W_bool = W > 0.0
    comps = connected_components(W_bool)
    n_nodes = W.shape[0]
    degrees = W_bool.sum(axis=1)
    zero_degree_fraction = float(np.sum(degrees == 0)) / float(n_nodes)
    n_edges = int(np.count_nonzero(np.triu(W_bool)))
    lcc = max(comps, key=len) if comps else list(range(n_nodes))
    lcc_fraction = len(lcc) / n_nodes
    disconnected = len(comps) > 1
    L_full = laplacian(W)
    eigs_full = np.linalg.eigvalsh(L_full)
    zero_eig_mult = int(np.sum(eigs_full < 1e-12))

    idx = np.array(lcc)
    W_lcc = W[np.ix_(idx, idx)]
    L_lcc = laplacian(W_lcc)

    info = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "components": len(comps),
        "lcc_fraction": lcc_fraction,
        "zero_degree_fraction": zero_degree_fraction,
        "zero_eig_mult": zero_eig_mult,
        "disconnected": disconnected,
    }
    return L_lcc, info


def eigenvalues_from_laplacian(L: np.ndarray) -> np.ndarray:
    eigs = np.linalg.eigvalsh(L)
    return np.maximum(eigs, 0.0)


def run_quantile_sweep(eigenvalues: np.ndarray, q_values: List[float]):
    geo_filter = PowerLawFilter(d0=2.0, beta_max=1.5, gamma=0.85, lambda0_quantile=0.6)
    geo_weights = geo_filter.apply(eigenvalues, C_GEO)
    lambda1 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    rows = []
    stop_reason = "none"

    for q in q_values:
        positive = eigenvalues[eigenvalues > 1e-12]
        if positive.size == 0:
            lambda_int = 0.0
        else:
            lambda_int = float(np.quantile(positive, q))
        delta_lambda = lambda_int - lambda1
        tau = SIGMA_CENTER * delta_lambda
        lambda_max = float(np.max(eigenvalues)) if eigenvalues.size else 0.0

        ds_curves = []
        for C_int in C_INT_VALUES:
            int_weights = interaction_weights(eigenvalues, lambda_int, lambda_max, C_int)
            total_weights = geo_weights * int_weights
            ds_curve, _ = filtered_spectral_dimension(eigenvalues, total_weights, SIGMA_GRID)
            ds_curves.append(ds_curve)

        delta_curve = (ds_curves[1] - ds_curves[0])[SIGMA_WINDOW_MASK]
        delta_curve = np.nan_to_num(delta_curve, nan=0.0)
        delta_curve[delta_curve < 0.0] = 0.0
        delta_max = float(np.max(delta_curve)) if delta_curve.size else 0.0

        mask = delta_curve > 1e-12
        if mask.sum() < 5:
            r2_exp = float("nan")
            r2_power = float("nan")
        else:
            sigma = SIGMA_WINDOW[mask]
            log_delta = np.log(delta_curve[mask])
            log_sigma = np.log(sigma)
            y_exp = log_delta + delta_lambda * sigma
            X = np.column_stack([np.ones_like(sigma), log_sigma])
            coef_exp, *_ = np.linalg.lstsq(X, y_exp, rcond=None)
            pred_exp = (X @ coef_exp) - delta_lambda * sigma
            r2_exp = _r2(log_delta, pred_exp)
            coef_pow, *_ = np.linalg.lstsq(X, log_delta, rcond=None)
            pred_pow = X @ coef_pow
            r2_power = _r2(log_delta, pred_pow)

        rows.append({
            "q_int": q,
            "lambda1": lambda1,
            "lambda_int": lambda_int,
            "delta_lambda": delta_lambda,
            "tau": tau,
            "delta_ds_max": delta_max,
            "r2_exp": r2_exp,
            "r2_power": r2_power,
            "regime_tau": classify_regime(tau),
            "delta_curve": delta_curve,
        })

        if tau < TAU_STOP:
            stop_reason = "gapless"
            break

    return rows, stop_reason


def write_summary(rows: List[dict], label: str | None, info: dict) -> Path:
    suffix = f"_{label}" if label else ""
    path = OUTPUT_DIR / f"gap_quantile_summary{suffix}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "q_int",
            "lambda1",
            "lambda_int",
            "delta_lambda",
            "tau",
            "delta_ds_max",
            "R2_exp",
            "R2_power",
            "Regime",
            "n_nodes",
            "n_edges",
            "components",
            "lcc_fraction",
            "zero_degree_fraction",
            "zero_eig_mult",
            "disconnected",
        ])
        for r in rows:
            writer.writerow([
                f"{r['q_int']:.2f}",
                f"{r['lambda1']:.6f}",
                f"{r['lambda_int']:.6f}",
                f"{r['delta_lambda']:.6f}",
                f"{r['tau']:.6f}",
                f"{r['delta_ds_max']:.6f}",
                f"{r['r2_exp']:.3f}",
                f"{r['r2_power']:.3f}",
                r.get("regime", r['regime_tau']),
                info["n_nodes"],
                info["n_edges"],
                info["components"],
                f"{info['lcc_fraction']:.4f}",
                f"{info['zero_degree_fraction']:.4f}",
                info["zero_eig_mult"],
                info["disconnected"],
            ])
    return path


def plot_curves(rows: List[dict], label: str | None):
    if not rows:
        return
    suffix = f"_{label}" if label else ""
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(rows)))
    sigma = SIGMA_WINDOW

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r, c in zip(rows, colors):
        ax.plot(sigma, r["delta_curve"], label=f"q={r['q_int']:.2f}", color=c)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\Delta d_s$")
    ax.set_title("Δd_s vs σ")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"delta_vs_sigma{suffix}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r, c in zip(rows, colors):
        ax.plot(sigma, np.log(r["delta_curve"] + 1e-18), label=f"q={r['q_int']:.2f}", color=c)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\log \Delta d_s$")
    ax.set_title("log Δd_s vs σ")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"log_delta_vs_sigma{suffix}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    log_sigma = np.log(sigma)
    for r, c in zip(rows, colors):
        ax.plot(log_sigma, np.log(r["delta_curve"] + 1e-18), label=f"q={r['q_int']:.2f}", color=c)
    ax.set_xlabel(r"$\log \sigma$")
    ax.set_ylabel(r"$\log \Delta d_s$")
    ax.set_title("log Δd_s vs log σ")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"log_delta_vs_log_sigma{suffix}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Gap collapse diagnostic on Market correlation graph")
    parser.add_argument("--prices", type=Path, default=None, help="CSV of prices (default: data/prices.csv)")
    parser.add_argument("--window", type=int, default=60, help="Rolling window length for correlation")
    parser.add_argument("--alpha", type=float, default=0.05, help="Shrinkage parameter")
    parser.add_argument("--label", type=str, default=None, help="Label suffix for outputs (e.g., spx, qqq)")
    args = parser.parse_args()

    prices_path = args.prices or (DATA_DIR / "prices.csv")
    L, info = load_market_graph(prices_path, args.window, args.alpha)
    eigenvalues = eigenvalues_from_laplacian(L)

    rows, stop_reason = run_quantile_sweep(eigenvalues, Q_LIST)

    # Regime override for disconnected graphs
    if info["disconnected"]:
        for r in rows:
            r["regime"] = "Disconnected"
    else:
        for r in rows:
            r["regime"] = r.get("regime_tau", "Rigid")

    write_summary(rows, args.label, info)
    plot_curves(rows, args.label)

    if info["disconnected"]:
        print("Graph disconnected — results labeled as Disconnected")
    elif stop_reason == "gapless":
        print("Gapless regime reached (σ_center Δλ < 0.2)")
    else:
        print("Sweep completed without reaching gapless threshold")

    print("| q_int | λ₁ | λ_int | Δλ | τ=σ_c Δλ | Δd_s(max) | R²_exp | R²_power | Regime |")
    print("|-----:|----:|------:|----:|-----------:|---------:|-------:|---------:|:-------|")
    for r in rows:
        print(
            f"| {r['q_int']:.2f} | {r['lambda1']:.4f} | {r['lambda_int']:.2f} | {r['delta_lambda']:.2f} | {r['tau']:.2f} | "
            f"{r['delta_ds_max']:.5f} | {r['r2_exp']:.3f} | {r['r2_power']:.3f} | {r.get('regime', r['regime_tau'])} |"
        )


if __name__ == "__main__":
    main()
