#!/usr/bin/env python3
"""Near-degenerate spectral-gap sweep for interaction axis tail detection."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dimshift.framework_spectral import filtered_spectral_dimension
from dimshift.spectral_filters import PowerLawFilter

# ---------------------------------------------------------------------------
# Constants from spec
# ---------------------------------------------------------------------------
C_GEO = 0.5
Q_INT = 0.8
SIGMA_LO = 0.0203566
SIGMA_HI = 0.0369353
SIGMA_FACTORS = [1, 2, 4, 8, 16]
SIGMA_POINTS = 1200
C_INT_VALUES = (0.0, 1.0)
EPSILON_SERIES = [0.30, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
OUTPUT_DIR = ROOT / "outputs" / "gap_tail"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class GapResult:
    epsilon: float
    lambda_1: float
    lambda_int: float
    delta_lambda: float
    sigma_centers: NDArray[np.float64]
    delta_values: NDArray[np.float64]
    delta_ds_max: float
    r2_exp: float
    r2_power: float
    regime: str


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_two_cluster_rgg(
    n_vertices: int,
    radius: float,
    epsilon: float,
    seed: int,
) -> np.ndarray:
    """Construct two-cluster RGG with epsilon*n cross edges."""
    assert n_vertices % 2 == 0, "n_vertices must be even"
    half = n_vertices // 2
    rng = np.random.default_rng(seed)
    pts1 = rng.uniform([0.0, 0.0], [0.2, 1.0], size=(half, 2))
    pts2 = rng.uniform([0.8, 0.0], [1.0, 1.0], size=(n_vertices - half, 2))
    points = np.vstack([pts1, pts2])

    dist = squareform(pdist(points, metric="euclidean"))
    adjacency = (dist < radius).astype(np.float64)
    np.fill_diagonal(adjacency, 0.0)

    n_cross = max(1, int(round(epsilon * n_vertices)))
    cross_edges = set()
    while len(cross_edges) < n_cross:
        i = rng.integers(0, half)
        j = rng.integers(half, n_vertices)
        cross_edges.add((i, j))
    for i, j in cross_edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    return laplacian


# ---------------------------------------------------------------------------
# Interaction weights
# ---------------------------------------------------------------------------

def interaction_weights(
    eigenvalues: NDArray[np.float64],
    lambda_thresh: float,
    lambda_max: float,
    C_int: float,
) -> NDArray[np.float64]:
    weights = np.ones_like(eigenvalues)
    mask = eigenvalues >= lambda_thresh
    if np.any(mask):
        span = lambda_max - lambda_thresh + 1e-12
        weights[mask] = 1.0 + C_int * ((eigenvalues[mask] - lambda_thresh) / span)
    return weights


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------

def fit_models(
    sigma_centers: NDArray[np.float64],
    delta_vals: NDArray[np.float64],
    delta_lambda: float,
) -> tuple[float, float]:
    mask = delta_vals > 1e-12
    if np.sum(mask) < 3:
        return float("nan"), float("nan")

    sigma = sigma_centers[mask]
    log_delta = np.log(delta_vals[mask])
    log_sigma = np.log(sigma)

    # Exponential: log Δ = a + k log σ - Δλ σ
    y_exp = log_delta + delta_lambda * sigma
    X_exp = np.column_stack([np.ones_like(sigma), log_sigma])
    coef_exp, _, _, _ = np.linalg.lstsq(X_exp, y_exp, rcond=None)
    pred_exp = (X_exp @ coef_exp) - delta_lambda * sigma
    r2_exp = _r_squared(log_delta, pred_exp)

    # Power law: log Δ = b + p log σ
    X_pow = np.column_stack([np.ones_like(sigma), log_sigma])
    coef_pow, _, _, _ = np.linalg.lstsq(X_pow, log_delta, rcond=None)
    pred_pow = X_pow @ coef_pow
    r2_power = _r_squared(log_delta, pred_pow)

    return r2_exp, r2_power


def _r_squared(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_gap_tail(
    n_vertices: int,
    radius: float,
    seed: int,
    epsilons: List[float],
) -> tuple[List[GapResult], str]:
    geo_filter = PowerLawFilter(d0=2.0, beta_max=1.5, gamma=0.85, lambda0_quantile=0.6)
    sigma_values = np.geomspace(SIGMA_LO, SIGMA_HI * max(SIGMA_FACTORS), SIGMA_POINTS)

    rows: List[GapResult] = []
    stop_reason = "none"

    for eps in epsilons:
        lap = build_two_cluster_rgg(n_vertices, radius, eps, seed)
        evals = np.sort(np.maximum(np.linalg.eigvalsh(lap), 0.0))
        if len(evals) < 2:
            stop_reason = "numerical"
            rows.append(GapResult(eps, float("nan"), float("nan"), float("nan"), np.array([]), np.array([]), 0.0, float("nan"), float("nan"), "Numerical"))
            break

        lambda1 = float(evals[1])
        positive = evals[evals > 1e-12]
        if positive.size == 0:
            stop_reason = "numerical"
            rows.append(GapResult(eps, lambda1, float("nan"), float("nan"), np.array([]), np.array([]), 0.0, float("nan"), float("nan"), "Numerical"))
            break
        lambda_int = float(np.quantile(positive, Q_INT))
        delta_lambda = lambda_int - lambda1

        geo_weights = geo_filter.apply(evals, C_GEO).weights
        lambda_max = float(np.max(evals))

        ds_curves = []
        for C_int in C_INT_VALUES:
            int_weights = interaction_weights(evals, lambda_int, lambda_max, C_int)
            total_weights = geo_weights * int_weights
            ds_curve, _ = filtered_spectral_dimension(evals, total_weights, sigma_values)
            ds_curves.append(ds_curve)

        sigma_centers = []
        delta_values = []
        for factor in SIGMA_FACTORS:
            lo = SIGMA_LO * factor
            hi = SIGMA_HI * factor
            mask = (sigma_values >= lo) & (sigma_values <= hi)
            if np.sum(mask) < 30:
                continue
            ds_low = float(np.median(ds_curves[0][mask]))
            ds_high = float(np.median(ds_curves[1][mask]))
            sigma_center = float(np.sqrt(lo * hi))
            sigma_centers.append(sigma_center)
            delta_values.append(max(ds_high - ds_low, 1e-12))

        sigma_centers_arr = np.array(sigma_centers)
        delta_values_arr = np.array(delta_values)
        delta_ds_max = float(delta_values_arr.max()) if delta_values_arr.size else 0.0

        r2_exp, r2_power = fit_models(sigma_centers_arr, delta_values_arr, delta_lambda)

        if np.isnan(r2_exp) or r2_exp < 0.9 or lambda1 < 1e-8:
            regime = "Numerical"
            stop_reason = "numerical"
        elif not np.isnan(r2_power) and r2_power >= r2_exp:
            regime = "Critical"
            stop_reason = "power"
        else:
            regime = "Rigid"
            if delta_lambda <= 0.05 * lambda_int:
                stop_reason = "gap"

        rows.append(GapResult(
            epsilon=eps,
            lambda_1=lambda1,
            lambda_int=lambda_int,
            delta_lambda=delta_lambda,
            sigma_centers=sigma_centers_arr,
            delta_values=delta_values_arr,
            delta_ds_max=delta_ds_max,
            r2_exp=r2_exp,
            r2_power=r2_power,
            regime=regime,
        ))

        if stop_reason != "none":
            break

    return rows, stop_reason


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_summary(rows: List[GapResult]) -> Path:
    summary_path = OUTPUT_DIR / "gap_tail_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "lambda_1", "lambda_int", "delta_lambda", "delta_ds_max", "R2_exp", "R2_power", "Regime"])
        for r in rows:
            writer.writerow([
                f"{r.epsilon:.6f}",
                f"{r.lambda_1:.6f}" if np.isfinite(r.lambda_1) else "nan",
                f"{r.lambda_int:.6f}" if np.isfinite(r.lambda_int) else "nan",
                f"{r.delta_lambda:.6f}" if np.isfinite(r.delta_lambda) else "nan",
                f"{r.delta_ds_max:.6f}",
                f"{r.r2_exp:.4f}" if np.isfinite(r.r2_exp) else "nan",
                f"{r.r2_power:.4f}" if np.isfinite(r.r2_power) else "nan",
                r.regime,
            ])
    return summary_path


def print_table(rows: List[GapResult]):
    header = "| ε | λ₁ | λ_int | Δλ | Δd_s(max) | R²_exp | R²_power | Regime |"
    sep = "|---:|----:|------:|----:|---------:|-------:|---------:|:-------|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r.epsilon:.3f} | {r.lambda_1:.4f} | {r.lambda_int:.2f} | {r.delta_lambda:.2f} | "
            f"{r.delta_ds_max:.5f} | {r.r2_exp:.3f} | {r.r2_power:.3f} | {r.regime} |")


def plot_curves(rows: List[GapResult]):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(rows)))

    # Δd_s vs σ
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for r, c in zip(rows, colors):
        ax.plot(r.sigma_centers, r.delta_values, marker="o", label=f"ε={r.epsilon:.3f}", color=c)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\Delta d_s$")
    ax.set_title("Δd_s vs σ")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delta_vs_sigma.png", dpi=200)
    plt.close(fig)

    # log Δd_s vs σ
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for r, c in zip(rows, colors):
        ax.plot(r.sigma_centers, np.log(r.delta_values), marker="o", label=f"ε={r.epsilon:.3f}", color=c)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\log \Delta d_s$")
    ax.set_title("log Δd_s vs σ")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "log_delta_vs_sigma.png", dpi=200)
    plt.close(fig)

    # log Δd_s vs log σ
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for r, c in zip(rows, colors):
        ax.plot(np.log(r.sigma_centers), np.log(r.delta_values), marker="o", label=f"ε={r.epsilon:.3f}", color=c)
    ax.set_xlabel(r"$\log \sigma$")
    ax.set_ylabel(r"$\log \Delta d_s$")
    ax.set_title("log Δd_s vs log σ")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "log_delta_vs_log_sigma.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Near-degenerate spectral-gap test")
    parser.add_argument("--n", type=int, default=400, help="Total vertices (even)")
    parser.add_argument("--radius", type=float, default=0.30, help="RGG radius")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows, stop_reason = run_gap_tail(args.n, args.radius, args.seed, EPSILON_SERIES)
    summary_path = write_summary(rows)
    print("Summary written to", summary_path)
    print_table(rows)
    plot_curves(rows)

    if stop_reason == "power":
        print("Power-law regime detected")
    elif stop_reason == "gap":
        print("Gap-controlled rigidity confirmed")
    else:
        print("No regime transition observed")


if __name__ == "__main__":
    main()
