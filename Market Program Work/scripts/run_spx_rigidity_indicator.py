#!/usr/bin/env python3
"""Rigidity indicator with adaptive q-stop baseline (default) and fixed (q_int, σ_scale) mode."""
from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
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
from src.spectral import laplacian as combinatorial_laplacian

# Baseline (paper) σ window
SIGMA_LO_DEFAULT = 0.0203566
SIGMA_HI_DEFAULT = 0.0369353
SIGMA_POINTS = 600

TAU_THRESHOLD = 0.4
C_GEO = 0.5
C_INT_VALUES = (0.0, 1.0)
DEFAULT_Q_LIST = [0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01]
DEGENERACY_THRESHOLD = 1.0

TAU_RIGID = 5.0
TAU_CRITICAL = 1.0
TAU_GAPLESS = 0.5

REQUIRED_QSTOP_COLUMNS = [
    "date",
    "ticker",
    "q_stop",
    "stop_reason",
    "regime",
    "lambda1",
    "lambda_int",
    "delta_lambda",
    "rs_value",
    "sigma_lo",
    "sigma_hi",
    "sigma_center",
    "window",
    "alpha",
    "adjacency",
    "laplacian",
    "n_nodes",
    "n_edges",
    "n_components",
    "lcc_fraction",
    "zero_degree_fraction",
    "data_hash",
    "prices_file_mtime",
    "git_commit",
]


@dataclass
class Config:
    mode: str
    window: int
    alpha: float
    adjacency: str
    laplacian: str
    q_list: List[float]
    q_int: float
    sigma_scale: float
    sigma_lo: float
    sigma_hi: float
    sigma_center: float
    sigma_grid: np.ndarray
    sigma_window: np.ndarray
    sigma_mask: np.ndarray
    ticker: str
    degeneracy_threshold: float
    prices_path: Path


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


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    d = np.sum(W, axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv = np.diag(inv_sqrt)
    I = np.eye(W.shape[0])
    L = I - D_inv @ W @ D_inv
    return 0.5 * (L + L.T)


def make_adjacency(C: np.ndarray, mode: str) -> np.ndarray:
    if mode == "signed":
        base = np.clip(C, 0.0, None)
    else:
        base = np.abs(C)
    W = np.maximum(base - TAU_THRESHOLD, 0.0)
    np.fill_diagonal(W, 0.0)
    return 0.5 * (W + W.T)


def connected_components(adj_bool: np.ndarray) -> List[List[int]]:
    n = adj_bool.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            neighbors = np.where(adj_bool[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        comps.append(comp)
    return comps


def graph_stats(W: np.ndarray) -> dict:
    adj_bool = W > 0
    n_nodes = W.shape[0]
    degrees = adj_bool.sum(axis=1)
    zero_deg = int(np.sum(degrees == 0))
    comps = connected_components(adj_bool) if n_nodes > 0 else []
    n_components = len(comps) if comps else (n_nodes if n_nodes > 0 else 0)
    lcc_size = max((len(c) for c in comps), default=n_nodes if n_nodes > 0 else 0)
    n_edges = int(np.count_nonzero(np.triu(adj_bool)))
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "lcc_fraction": (lcc_size / n_nodes) if n_nodes else 0.0,
        "zero_degree_fraction": (zero_deg / n_nodes) if n_nodes else 0.0,
    }


def filtered_log_return_probability(eigenvalues: np.ndarray, weights: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    log_weights = np.full_like(eigenvalues, -np.inf, dtype=float)
    mask = weights > 1e-300
    log_weights[mask] = np.log(weights[mask])
    log_summands = -sigma_values[:, None] * eigenvalues[None, :] + log_weights[None, :]
    active = log_summands[:, mask]
    if active.size == 0:
        return np.full(len(sigma_values), -np.inf)
    max_val = np.max(active, axis=1, keepdims=True)
    log_sum = max_val[:, 0] + np.log(np.sum(np.exp(active - max_val), axis=1))
    return log_sum - np.log(np.maximum(np.count_nonzero(mask), 1))


def filtered_spectral_dimension(eigenvalues: np.ndarray, weights: np.ndarray, sigma_values: np.ndarray) -> np.ndarray:
    ln_P = filtered_log_return_probability(eigenvalues, weights, sigma_values)
    ln_sigma = np.log(sigma_values)
    ds = np.full_like(sigma_values, np.nan, dtype=float)
    for i in range(1, len(sigma_values) - 1):
        d_lnP = (ln_P[i + 1] - ln_P[i - 1]) / (ln_sigma[i + 1] - ln_sigma[i - 1])
        ds[i] = -2.0 * d_lnP
    return ds


def delta_curve_metrics(
    eigenvalues: np.ndarray,
    geo_weights: np.ndarray,
    lambda_int: float,
    lambda_max: float,
    sigma_grid: np.ndarray,
    sigma_window: np.ndarray,
    sigma_mask: np.ndarray,
    delta_lambda: float,
) -> tuple[np.ndarray, float, float, float]:
    ds_curves = []
    for C_int in C_INT_VALUES:
        int_weights = interaction_weights(eigenvalues, lambda_int, lambda_max, C_int)
        total_weights = geo_weights * int_weights
        ds_curve = filtered_spectral_dimension(eigenvalues, total_weights, sigma_grid)
        ds_curves.append(ds_curve)
    delta_curve = (ds_curves[1] - ds_curves[0])[sigma_mask]
    delta_curve = np.nan_to_num(delta_curve, nan=0.0)
    delta_curve = np.clip(delta_curve, 0.0, None)
    delta_max = float(np.max(delta_curve)) if delta_curve.size else 0.0
    mask = delta_curve > 1e-12
    if mask.sum() < 5:
        return delta_curve, delta_max, float("nan"), float("nan")
    sigma = sigma_window[mask]
    log_delta = np.log(delta_curve[mask])
    log_sigma = np.log(sigma)
    X = np.column_stack([np.ones_like(sigma), log_sigma])
    y_exp = log_delta + delta_lambda * sigma
    coef_exp, *_ = np.linalg.lstsq(X, y_exp, rcond=None)
    pred_exp = (X @ coef_exp) - delta_lambda * sigma
    r2_exp = _r2(log_delta, pred_exp)
    coef_pow, *_ = np.linalg.lstsq(X, log_delta, rcond=None)
    pred_pow = X @ coef_pow
    r2_power = _r2(log_delta, pred_pow)
    return delta_curve, delta_max, r2_exp, r2_power


def interaction_weights(eigenvalues: np.ndarray, lambda_thresh: float, lambda_max: float, C_int: float) -> np.ndarray:
    weights = np.ones_like(eigenvalues)
    mask = eigenvalues >= lambda_thresh
    if np.any(mask):
        span = lambda_max - lambda_thresh + 1e-12
        weights[mask] = 1.0 + C_int * ((eigenvalues[mask] - lambda_thresh) / span)
    return weights


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


def adaptive_metrics(eigenvalues: np.ndarray, cfg: Config) -> dict:
    geo_filter = PowerLawFilter(d0=2.0, beta_max=1.5, gamma=0.85, lambda0_quantile=0.6)
    geo_weights = geo_filter.apply(eigenvalues, C_GEO)
    lambda1 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    positive = eigenvalues[eigenvalues > 1e-12]
    lambda_max = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    last_metrics = None
    for q in cfg.q_list:
        if positive.size == 0:
            lambda_int = 0.0
        else:
            lambda_int = float(np.quantile(positive, q))
        delta_lambda = lambda_int - lambda1
        tau = cfg.sigma_center * delta_lambda
        _, delta_max, r2_exp, r2_power = delta_curve_metrics(
            eigenvalues,
            geo_weights,
            lambda_int,
            lambda_max,
            cfg.sigma_grid,
            cfg.sigma_window,
            cfg.sigma_mask,
            delta_lambda,
        )
        metrics = {
            "lambda1": lambda1,
            "lambda_int": lambda_int,
            "delta_lambda": delta_lambda,
            "tau": tau,
            "q_stop": q,
            "delta_ds_max": delta_max,
            "r2_exp": r2_exp,
            "r2_power": r2_power,
        }
        if delta_lambda < cfg.degeneracy_threshold:
            return {**metrics, "regime": "Degenerate", "stop_reason": "gap"}
        if not np.isnan(r2_exp) and not np.isnan(r2_power) and r2_power >= r2_exp:
            return {**metrics, "regime": "Critical", "stop_reason": "power"}
        last_metrics = metrics
    if last_metrics is None:
        last_metrics = {
            "lambda1": lambda1,
            "lambda_int": 0.0,
            "delta_lambda": 0.0,
            "tau": 0.0,
            "q_stop": cfg.q_list[-1],
            "delta_ds_max": 0.0,
            "r2_exp": float("nan"),
            "r2_power": float("nan"),
        }
    return {**last_metrics, "regime": "Rigid", "stop_reason": "progress"}


def fixed_metrics(eigenvalues: np.ndarray, cfg: Config) -> dict:
    positive = eigenvalues[eigenvalues > 1e-12]
    lambda1 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    if positive.size == 0:
        lambda_int = 0.0
    else:
        lambda_int = float(np.quantile(positive, cfg.q_int))
    delta_lambda = lambda_int - lambda1
    tau = cfg.sigma_center * cfg.sigma_scale * delta_lambda
    rs_value = float(np.exp(-tau)) if delta_lambda > 0 else 1.0
    if tau >= TAU_RIGID:
        regime = "Rigid"
    elif tau >= TAU_CRITICAL:
        regime = "Critical"
    else:
        regime = "Gapless"
    return {
        "lambda1": lambda1,
        "lambda_int": lambda_int,
        "delta_lambda": delta_lambda,
        "tau": tau,
        "q_stop": cfg.q_int,
        "regime": regime,
        "stop_reason": regime.lower(),
        "rs_value": rs_value,
    }


def classify_event_regime(row: dict, mode: str) -> str:
    if mode == "adaptive":
        if row.get("stop_reason") == "gap":
            return "Gapless"
        if row.get("stop_reason") == "power":
            return "Critical"
        return "Rigid"
    return row.get("regime", "Gapless")


def compute_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "NA"


def compute_data_hash(path: Path) -> tuple[str, str]:
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
    sig = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime}"
    digest = hashlib.md5(sig.encode()).hexdigest()
    return digest, mtime


def write_signal_log(df: pd.DataFrame, out_path: Path, mode: str):
    log_rows = []
    prev_regime = None
    for _, row in df.iterrows():
        regime = classify_event_regime(row, mode)
        if prev_regime is None:
            prev_regime = regime
            continue
        if regime != prev_regime:
            log_rows.append({
                "date": row["date"],
                "regime": regime,
                "tau": row.get("tau", np.nan),
                "rs_value": row.get("rs_value", np.nan),
                "q_stop": row.get("q_stop"),
                "delta_lambda": row.get("delta_lambda"),
                "note": f"Regime changed from {prev_regime} to {regime}",
            })
            prev_regime = regime
    if not log_rows:
        pd.DataFrame(columns=["date", "regime", "tau", "rs_value", "q_stop", "delta_lambda", "note"]).to_csv(out_path, index=False)
        return
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_path, index=False)


def plot_overlay(df: pd.DataFrame, ds_series_path: Path, out_path: Path):
    if not out_path:
        return
    ds = pd.read_csv(ds_series_path)
    rename_map = {ds.columns[0]: "date", ds.columns[1]: "ds"}
    if ds.shape[1] >= 3:
        rename_map[ds.columns[2]] = "ds_regime"
    ds.rename(columns=rename_map, inplace=True)
    ds["date"] = pd.to_datetime(ds["date"])
    merged = pd.merge(ds, df, on="date", how="inner")
    if merged.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(merged["date"], merged["ds"], color="#D55E00", label="Spectral dimension (plateau)")
    ax1.set_ylabel("d_s plateau", color="#D55E00")
    ax1.tick_params(axis="y", labelcolor="#D55E00")
    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["tau"], color="#0072B2", label="τ = σ_c Δλ")
    ax2.set_ylabel("τ", color="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#0072B2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def enforce_adaptive_invariants(df: pd.DataFrame, cfg: Config):
    mask = df["delta_lambda"].abs() > 1e-9
    sigma_est = -np.log(df.loc[mask, "rs_value"]) / df.loc[mask, "delta_lambda"]
    sigma_mean = float(sigma_est.mean()) if len(sigma_est) else float("nan")
    sigma_std = float(sigma_est.std()) if len(sigma_est) else float("nan")
    if len(sigma_est) and (abs(sigma_mean - cfg.sigma_center) > 1e-6 or sigma_std > 1e-6):
        raise RuntimeError(f"σ_center invariant failed: mean={sigma_mean}, std={sigma_std}, expected {cfg.sigma_center}")
    deg = df[df["stop_reason"] == "gap"]
    if not deg.empty and not (deg["delta_lambda"] < cfg.degeneracy_threshold + 1e-9).all():
        raise RuntimeError("Degenerate rows detected with Δλ ≥ threshold")
    if len(sigma_est):
        print(f"σ_center check: mean={sigma_mean:.8f}, std={sigma_std:.8f}")


def export_qstop_history(df: pd.DataFrame, cfg: Config, args: argparse.Namespace, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Path:
    export_df = df.copy()
    digest, mtime = compute_data_hash(cfg.prices_path)
    export_df["data_hash"] = digest
    export_df["prices_file_mtime"] = mtime
    export_df["git_commit"] = compute_git_commit()
    export_df["stop_reason"] = export_df["stop_reason"].fillna(export_df.get("reason"))
    export_df["ticker"] = cfg.ticker
    cols = REQUIRED_QSTOP_COLUMNS + [c for c in export_df.columns if c not in REQUIRED_QSTOP_COLUMNS]
    export_df = export_df[cols]
    sigma_label = f"{cfg.sigma_lo:.6f}-{cfg.sigma_hi:.6f}"
    start_str = start_date.strftime("%Y%m%d") if not pd.isna(start_date) else "na"
    end_str = end_date.strftime("%Y%m%d") if not pd.isna(end_date) else "na"
    filename = f"qstop_history_{cfg.ticker}_w{cfg.window}_a{cfg.alpha}_{cfg.laplacian}_{cfg.adjacency}_sig{sigma_label}_{start_str}_{end_str}.csv"
    out_path = Path(args.output_dir or args.out.parent).resolve() / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_path, index=False)
    print(f"q_stop history exported to {out_path}")
    return out_path


def compare_qstop(new_df: pd.DataFrame, old_path: Path):
    if not old_path.exists():
        raise FileNotFoundError(f"Comparison file not found: {old_path}")
    old = pd.read_csv(old_path, parse_dates=["date"], infer_datetime_format=True)
    new = new_df.copy()
    new_cols = {
        "stop_reason": "stop_reason" if "stop_reason" in new.columns else "reason",
        "regime": "regime",
        "q_stop": "q_stop",
    }
    old_stop_col = "stop_reason" if "stop_reason" in old.columns else ("reason" if "reason" in old.columns else None)
    if old_stop_col is None:
        old_stop_col = "stop_reason"
        old[old_stop_col] = "unknown"
    merged = new[["date", "q_stop", new_cols["stop_reason"], "regime"]].merge(
        old[["date", "q_stop", old_stop_col, "regime"]],
        on="date",
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
    )
    diffs = merged[(merged["_merge"] == "both") & ((merged["q_stop_new"] != merged["q_stop_old"]) | (merged[f"{new_cols['stop_reason']}_new"] != merged[f"{old_stop_col}_old"]) | (merged["regime_new"] != merged["regime_old"]))]
    added = merged[merged["_merge"] == "right_only"]
    removed = merged[merged["_merge"] == "left_only"]
    print(f"q_stop comparison vs {old_path}:")
    print(f"  Matching rows with differences: {len(diffs)}")
    print(f"  Rows only in new run: {len(removed)}")
    print(f"  Rows only in reference: {len(added)}")
    if not diffs.empty:
        sample = diffs.iloc[0]
        print("  First mismatch:")
        print(sample[["date", "q_stop_new", "q_stop_old", f"{new_cols['stop_reason']}_new", f"{old_stop_col}_old", "regime_new", "regime_old"]])


def parse_float_list(values: str) -> List[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def build_config(args: argparse.Namespace, prices_path: Path) -> Config:
    sigma_center = args.sigma_center or math.sqrt(args.sigma_lo * args.sigma_hi)
    sigma_grid = np.geomspace(args.sigma_lo / 2.0, args.sigma_hi * 2.0, SIGMA_POINTS)
    sigma_mask = (sigma_grid >= args.sigma_lo) & (sigma_grid <= args.sigma_hi)
    sigma_window = sigma_grid[sigma_mask]
    q_list = parse_float_list(args.q_list) if isinstance(args.q_list, str) else DEFAULT_Q_LIST
    return Config(
        mode=args.mode,
        window=args.window,
        alpha=args.alpha,
        adjacency=args.adjacency,
        laplacian=args.laplacian,
        q_list=q_list,
        q_int=args.q_int,
        sigma_scale=args.sigma_scale,
        sigma_lo=args.sigma_lo,
        sigma_hi=args.sigma_hi,
        sigma_center=sigma_center,
        sigma_grid=sigma_grid,
        sigma_window=sigma_window,
        sigma_mask=sigma_mask,
        ticker=args.ticker,
        degeneracy_threshold=args.degeneracy_threshold,
        prices_path=prices_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Rigidity indicator with adaptive/fixed modes")
    parser.add_argument("--prices", type=Path, default=ROOT / "data" / "sp500_prices.csv")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--adjacency", choices=["abs", "signed"], default="abs")
    parser.add_argument("--laplacian", choices=["combinatorial", "normalized"], default="combinatorial")
    parser.add_argument("--q-list", type=str, default=",".join(str(q) for q in DEFAULT_Q_LIST))
    parser.add_argument("--q-int", type=float, default=0.80)
    parser.add_argument("--sigma-scale", type=float, default=0.20)
    parser.add_argument("--sigma-lo", type=float, default=SIGMA_LO_DEFAULT)
    parser.add_argument("--sigma-hi", type=float, default=SIGMA_HI_DEFAULT)
    parser.add_argument("--sigma-center", type=float, default=None)
    parser.add_argument("--degeneracy-threshold", type=float, default=DEGENERACY_THRESHOLD)
    parser.add_argument("--ticker", type=str, default="SPX")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--ds", type=Path, default=ROOT / "data" / "ds_series.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "rigidity_signal_spx.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for q_stop exports (defaults to output file directory)")
    parser.add_argument("--export-qstop-history", action="store_true")
    parser.add_argument("--compare-qstop", type=Path, default=None)
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")
    if args.start_date or args.end_date:
        returns = returns.loc[args.start_date: args.end_date]
    cfg = build_config(args, args.prices)

    rows = []
    sigma_center = cfg.sigma_center
    adjacency_stats = None
    iterator = rolling_correlation_matrices(returns, window=cfg.window, alpha=cfg.alpha, use_abs=True)
    for idx, (date, C_s) in enumerate(iterator):
        if args.max_windows and idx >= args.max_windows:
            break
        W = make_adjacency(C_s, cfg.adjacency)
        stats = graph_stats(W)
        L = combinatorial_laplacian(W) if cfg.laplacian == "combinatorial" else normalized_laplacian(W)
        eigs = np.linalg.eigvalsh(L)
        eigs = np.maximum(eigs, 0.0)
        if len(eigs) < 2:
            continue
        if cfg.mode == "adaptive":
            metrics = adaptive_metrics(eigs, cfg)
            rs_value = float(np.exp(-sigma_center * metrics["delta_lambda"])) if metrics["delta_lambda"] > 0 else 1.0
            metrics["rs_value"] = rs_value
        else:
            metrics = fixed_metrics(eigs, cfg)
        row = {
            **metrics,
            **stats,
            "date": pd.to_datetime(date),
            "sigma_lo": cfg.sigma_lo,
            "sigma_hi": cfg.sigma_hi,
            "sigma_center": sigma_center,
            "window": cfg.window,
            "alpha": cfg.alpha,
            "adjacency": cfg.adjacency,
            "laplacian": cfg.laplacian,
            "ticker": cfg.ticker,
        }
        rows.append(row)
    if not rows:
        raise RuntimeError("No windows processed; check inputs or date filters")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if cfg.mode == "adaptive":
        enforce_adaptive_invariants(df, cfg)
    # Write primary indicator file
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.mode == "adaptive":
        indicator_cols = ["lambda1", "delta_lambda", "q_stop", "regime", "stop_reason", "rs_value", "date"]
    else:
        indicator_cols = ["lambda1", "delta_lambda", "tau", "q_stop", "regime", "rs_value", "date"]
    df[indicator_cols].to_csv(out_path, index=False)
    print(f"Indicator saved to {out_path}")

    log_path = out_path.with_name(out_path.stem.replace("signal", "events") + out_path.suffix)
    write_signal_log(df, log_path, cfg.mode)
    print(f"Signal log saved to {log_path}")

    overlay_path = out_path.with_name(out_path.stem.replace("signal", "overlay") + ".png")
    plot_overlay(df, args.ds, overlay_path)
    print(f"Overlay saved to {overlay_path}")

    qstop_path = None
    if cfg.mode != "adaptive" and (args.export_qstop_history or args.compare_qstop):
        raise ValueError("q_stop history features are only supported in adaptive mode")
    if cfg.mode == "adaptive" and (args.export_qstop_history or args.compare_qstop):
        start_date = df["date"].min()
        end_date = df["date"].max()
        qstop_path = export_qstop_history(df, cfg, args, start_date, end_date)
    if cfg.mode == "adaptive" and args.compare_qstop is not None:
        compare_source = qstop_path if isinstance(qstop_path, Path) else df
        if isinstance(compare_source, pd.DataFrame):
            compare_qstop(compare_source, args.compare_qstop)
        else:
            fresh = pd.read_csv(compare_source, parse_dates=["date"])
            compare_qstop(fresh, args.compare_qstop)


if __name__ == "__main__":
    main()
