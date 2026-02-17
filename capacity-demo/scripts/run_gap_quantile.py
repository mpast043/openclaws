#!/usr/bin/env python3
"""Interaction-quantile collapse experiment for fixed rgg400 substrate."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dimshift.framework_spectral import filtered_spectral_dimension
from dimshift.spectral_filters import PowerLawFilter
from dimshift.substrates import random_geometric_graph

# Fixed constants from spec
C_GEO = 0.5
SIGMA_LO = 0.0203566
SIGMA_HI = 0.0369353
SIGMA_POINTS = 600
C_INT_VALUES = (0.0, 1.0)
Q_INT_LIST = [0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01]

OUTPUT_DIR = ROOT / "outputs" / "gap_quantile"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ResultRow(Tuple[float, float, float, float, float, float, float, str]):
    pass


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


def fit_models(
    sigma_vals: NDArray[np.float64],
    delta_vals: NDArray[np.float64],
    delta_lambda: float,
) -> tuple[float, float]:
    mask = delta_vals > 1e-12
    if mask.sum() < 5:
        return float("nan"), float("nan")

    sigma = sigma_vals[mask]
    log_delta = np.log(delta_vals[mask])
    log_sigma = np.log(sigma)

    y_exp = log_delta + delta_lambda * sigma
    X = np.column_stack([np.ones_like(sigma), log_sigma])
    coef_exp, _, _, _ = np.linalg.lstsq(X, y_exp, rcond=None)
    pred_exp = (X @ coef_exp) - delta_lambda * sigma
    r2_exp = _r2(log_delta, pred_exp)

    coef_pow, _, _, _ = np.linalg.lstsq(X, log_delta, rcond=None)
    pred_pow = X @ coef_pow
    r2_power = _r2(log_delta, pred_pow)

    return r2_exp, r2_power


def _r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


def run_quantile_sweep(
    q_list: List[float],
    radius: float,
    seed: int,
) -> tuple[list[dict], str]:
    # Fixed substrate (same as prior two_axis rgg400)
    sub = random_geometric_graph(n_vertices=400, D=2, radius=radius, seed=seed)
    eigenvalues = np.sort(np.maximum(sub.eigenvalues, 0.0))

    geo_filter = PowerLawFilter(d0=2.0, beta_max=1.5, gamma=0.85, lambda0_quantile=0.6)
    geo_weights = geo_filter.apply(eigenvalues, C_GEO).weights

    sigma_values = np.geomspace(SIGMA_LO / 2.0, SIGMA_HI * 4.0, SIGMA_POINTS)
    window_mask = (sigma_values >= SIGMA_LO) & (sigma_values <= SIGMA_HI)
    sigma_window = sigma_values[window_mask]

    rows = []
    stop_reason = "none"

    for q in q_list:
        positive = eigenvalues[eigenvalues > 1e-12]
        lambda_int = float(np.quantile(positive, q))
        lambda1 = float(eigenvalues[1])
        delta_lambda = lambda_int - lambda1
        lambda_max = float(np.max(eigenvalues))

        ds_curves = []
        for C_int in C_INT_VALUES:
            int_weights = interaction_weights(eigenvalues, lambda_int, lambda_max, C_int)
            total_weights = geo_weights * int_weights
            ds_curve, _ = filtered_spectral_dimension(eigenvalues, total_weights, sigma_values)
            ds_curves.append(ds_curve)

        delta_curve = ds_curves[1] - ds_curves[0]
        delta_window = delta_curve[window_mask]
        delta_max = float(delta_window.max())

        r2_exp, r2_power = fit_models(sigma_window, delta_window, delta_lambda)

        if np.isnan(r2_exp) or r2_exp < 0.9:
            regime = "Critical"
            stop_reason = "numerical"
        elif delta_lambda < 1.0:
            regime = "Degenerate"
            stop_reason = "gap"
        elif not np.isnan(r2_power) and r2_power >= r2_exp:
            regime = "Critical"
            stop_reason = "power"
        else:
            regime = "Rigid"

        rows.append({
            "q_int": q,
            "lambda1": lambda1,
            "lambda_int": lambda_int,
            "delta_lambda": delta_lambda,
            "delta_ds_max": delta_max,
            "r2_exp": r2_exp,
            "r2_power": r2_power,
            "regime": regime,
            "sigma": sigma_window,
            "delta_curve": delta_window,
        })

        if stop_reason != "none":
            break

    return rows, stop_reason


def write_table(rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "gap_quantile_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_int", "lambda1", "lambda_int", "delta_lambda", "delta_ds_max", "R2_exp", "R2_power", "Regime"])
        for r in rows:
            writer.writerow([
                f"{r['q_int']:.2f}",
                f"{r['lambda1']:.6f}",
                f"{r['lambda_int']:.6f}",
                f"{r['delta_lambda']:.6f}",
                f"{r['delta_ds_max']:.6f}",
                f"{r['r2_exp']:.3f}",
                f"{r['r2_power']:.3f}",
                r["regime"],
            ])
    return path


def make_plots(rows: list[dict]):
    if not rows:
        return
    colors = plt.cm.magma(np.linspace(0.1, 0.9, len(rows)))
    sigma = rows[0]["sigma"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r, c in zip(rows, colors):
        ax.plot(sigma, r["delta_curve"], label=f"q={r['q_int']:.2f}", color=c)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\Delta d_s$")
    ax.set_title("Δd_s vs σ")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delta_vs_sigma.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r, c in zip(rows, colors):
        ax.plot(sigma, np.log(r["delta_curve"] + 1e-18), label=f"q={r['q_int']:.2f}", color=c)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\log \Delta d_s$")
    ax.set_title("log Δd_s vs σ")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "log_delta_vs_sigma.png", dpi=200)
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
    fig.savefig(OUTPUT_DIR / "log_delta_vs_log_sigma.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Collapse interaction quantile on fixed rgg400")
    parser.add_argument("--radius", type=float, default=0.35, help="RGG radius (fixed substrate)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows, stop_reason = run_quantile_sweep(Q_INT_LIST, args.radius, args.seed)
    write_table(rows)
    make_plots(rows)

    if stop_reason == "gap":
        print("Spectral degeneracy reached")
    elif stop_reason == "power":
        print("Power-law regime detected")
    else:
        print("Gap-controlled rigidity confirmed")

    # Pretty table for console
    print("| q_int | λ₁ | λ_int | Δλ | Δd_s(max) | R²_exp | R²_power | Regime |")
    print("|-----:|----:|------:|----:|---------:|-------:|---------:|:-------|")
    for r in rows:
        print(
            f"| {r['q_int']:.2f} | {r['lambda1']:.4f} | {r['lambda_int']:.2f} | {r['delta_lambda']:.2f} | "
            f"{r['delta_ds_max']:.5f} | {r['r2_exp']:.3f} | {r['r2_power']:.3f} | {r['regime']} |")


if __name__ == "__main__":
    main()
