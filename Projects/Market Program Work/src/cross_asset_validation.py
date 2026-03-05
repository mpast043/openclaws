"""
Cross-asset structural validation: does ds-based compression overlay generalize?
Fixed rules only. No tuning. Same ds (S&P 500) and same percentile rules for all assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple

from .portfolio_eval import (
    load_vix,
    load_index_prices,
    daily_returns,
    metrics,
    compression_flag,
    vix_high_flag,
    overlay_returns,
    ROLLING_DAYS,
    RF,
    OOS_START,
    PERCENTILE_DEFAULT,
    VIX_HIGH_PERCENTILE,
)

# Fixed asset list: foreign equity ETFs, commodity ETFs
CROSS_ASSET_TICKERS = [
    "EFA",   # Developed ex-US
    "EEM",   # Emerging Markets
    "EWJ",   # Japan
    "VGK",   # Europe
    "GLD",   # Gold
    "DBC",   # Broad commodities
    "USO",   # Oil proxy
]


def _load_prices(start: str, end: str, ticker: str) -> pd.Series:
    """Load ETF/index daily close. Use full available range (yfinance)."""
    return load_index_prices(start, end, ticker=ticker)


def _one_asset_results(
    asset_returns: pd.Series,
    ds_aligned: pd.Series,
    vix_aligned: pd.Series,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], float, float, float, int]:
    """
    For one asset: baseline, ds overlay, VIX overlay metrics (full + OOS) and overlap stats.
    Returns (baseline_full, baseline_oos, ds_full, ds_oos, vix_full, vix_oos, pct_ds, pct_vix, pct_overlap, n_days).
    """
    common = asset_returns.index.intersection(ds_aligned.index).intersection(vix_aligned.index)
    ret = asset_returns.reindex(common).dropna()
    ds = ds_aligned.reindex(common).ffill().bfill()
    vix = vix_aligned.reindex(common).ffill().bfill()
    common = ret.index.intersection(ds.dropna().index).intersection(vix.dropna().index)
    ret = ret.loc[common]
    ds = ds.loc[common]
    vix = vix.loc[common]
    if len(ret) < ROLLING_DAYS:
        return (
            {}, {}, {}, {}, {}, {},
            np.nan, np.nan, np.nan, 0,
        )
    comp = compression_flag(ds, PERCENTILE_DEFAULT, ROLLING_DAYS)
    vix_h = vix_high_flag(vix, VIX_HIGH_PERCENTILE, ROLLING_DAYS)
    base_ret = ret
    ds_ret = overlay_returns(ret, comp, 0.7)
    vix_ret = overlay_returns(ret, vix_h, 0.7)
    oos = ret.index >= OOS_START
    baseline_full = metrics(base_ret, rf=RF)
    baseline_oos = metrics(base_ret[oos], rf=RF) if oos.sum() > 0 else {}
    ds_full = metrics(ds_ret.dropna(), rf=RF)
    ds_oos = metrics(ds_ret[oos].dropna(), rf=RF) if oos.sum() > 0 else {}
    vix_full = metrics(vix_ret.dropna(), rf=RF)
    vix_oos = metrics(vix_ret[oos].dropna(), rf=RF) if oos.sum() > 0 else {}
    n = len(common)
    pct_ds = 100.0 * comp.reindex(common).fillna(False).sum() / n if n else np.nan
    pct_vix = 100.0 * vix_h.reindex(common).fillna(False).sum() / n if n else np.nan
    both = ((comp.reindex(common).fillna(False)) & (vix_h.reindex(common).fillna(False))).sum()
    pct_overlap = 100.0 * both / n if n else np.nan
    return (
        baseline_full, baseline_oos,
        ds_full, ds_oos,
        vix_full, vix_oos,
        pct_ds, pct_vix, pct_overlap, int(n),
    )


def run_cross_asset_validation(
    ds_series: pd.Series,
    data_dir: Union[str, Path] = "data",
) -> Dict[str, Any]:
    """
    Fixed rules. No tuning. For each ETF: baseline, ds overlay (20th pct), VIX overlay (80th pct).
    Same ds series (S&P 500) for all. Full sample + OOS (2016+). Metrics + overlap per asset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    start = ds_series.index.min().strftime("%Y-%m-%d")
    end = ds_series.index.max().strftime("%Y-%m-%d")

    vix = load_vix(start, end)
    ds_aligned = ds_series.copy()
    if ds_aligned.index.tz is not None:
        ds_aligned = ds_aligned.tz_localize(None)
    vix_aligned = vix

    # Summary comparison: one row per (asset, strategy) with all metrics
    comparison_rows: List[Dict] = []
    # Overlay vs baseline differences: asset, strategy (ds_overlay / vix_overlay), max_dd_diff, sharpe_diff, cagr_diff (full and oos)
    diff_rows: List[Dict] = []
    # ds vs VIX overlay comparison: asset, full/oos, sharpe_ds_minus_vix, max_dd_ds_minus_vix, cagr_ds_minus_vix
    ds_vs_vix_rows: List[Dict] = []
    # Regime overlap: asset, pct_days_ds_active, pct_days_vix_high_active, pct_overlap, n_days
    overlap_rows: List[Dict] = []

    for ticker in CROSS_ASSET_TICKERS:
        try:
            px = _load_prices(start, end, ticker)
        except Exception:
            continue
        ret = daily_returns(px)
        if ret.empty or len(ret) < ROLLING_DAYS:
            continue
        (
            baseline_full, baseline_oos,
            ds_full, ds_oos,
            vix_full, vix_oos,
            pct_ds, pct_vix, pct_overlap, n_days,
        ) = _one_asset_results(ret, ds_aligned, vix_aligned)
        if not baseline_full:
            continue
        # Comparison table
        for label, m in [
            ("baseline_full", baseline_full),
            ("ds_overlay_full", ds_full),
            ("vix_overlay_full", vix_full),
        ]:
            comparison_rows.append({"asset": ticker, "strategy": label, **{k: round(v, 4) if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else v for k, v in m.items()}})
        if baseline_oos:
            for label, m in [
                ("baseline_oos", baseline_oos),
                ("ds_overlay_oos", ds_oos),
                ("vix_overlay_oos", vix_oos),
            ]:
                comparison_rows.append({"asset": ticker, "strategy": label, **{k: round(v, 4) if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) else v for k, v in m.items()}})
        # Overlay vs baseline diffs (one row per asset per strategy with full and oos columns)
        for strat_name, m_full, m_oos in [("ds_overlay", ds_full, ds_oos), ("vix_overlay", vix_full, vix_oos)]:
            row = {
                "asset": ticker,
                "strategy": strat_name,
                "max_dd_diff_full": m_full.get("max_dd", np.nan) - baseline_full.get("max_dd", np.nan),
                "sharpe_diff_full": m_full.get("sharpe", np.nan) - baseline_full.get("sharpe", np.nan),
                "cagr_diff_full": m_full.get("cagr", np.nan) - baseline_full.get("cagr", np.nan),
            }
            if baseline_oos and m_oos:
                row["max_dd_diff_oos"] = m_oos.get("max_dd", np.nan) - baseline_oos.get("max_dd", np.nan)
                row["sharpe_diff_oos"] = m_oos.get("sharpe", np.nan) - baseline_oos.get("sharpe", np.nan)
                row["cagr_diff_oos"] = m_oos.get("cagr", np.nan) - baseline_oos.get("cagr", np.nan)
            diff_rows.append(row)
        # ds vs VIX
        ds_vs_vix_rows.append({
            "asset": ticker,
            "sample": "full",
            "sharpe_ds_minus_vix": ds_full.get("sharpe", np.nan) - vix_full.get("sharpe", np.nan),
            "max_dd_ds_minus_vix": ds_full.get("max_dd", np.nan) - vix_full.get("max_dd", np.nan),
            "cagr_ds_minus_vix": ds_full.get("cagr", np.nan) - vix_full.get("cagr", np.nan),
        })
        if ds_oos and vix_oos:
            ds_vs_vix_rows.append({
                "asset": ticker,
                "sample": "oos",
                "sharpe_ds_minus_vix": ds_oos.get("sharpe", np.nan) - vix_oos.get("sharpe", np.nan),
                "max_dd_ds_minus_vix": ds_oos.get("max_dd", np.nan) - vix_oos.get("max_dd", np.nan),
                "cagr_ds_minus_vix": ds_oos.get("cagr", np.nan) - vix_oos.get("cagr", np.nan),
            })
        # Overlap
        overlap_rows.append({
            "asset": ticker,
            "pct_days_ds_compression_active": round(pct_ds, 2) if not np.isnan(pct_ds) else np.nan,
            "pct_days_vix_high_active": round(pct_vix, 2) if not np.isnan(pct_vix) else np.nan,
            "pct_overlap_both_active": round(pct_overlap, 2) if not np.isnan(pct_overlap) else np.nan,
            "n_days": n_days,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    diff_df = pd.DataFrame(diff_rows)
    ds_vs_vix_df = pd.DataFrame(ds_vs_vix_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    for c in comparison_df.columns:
        if comparison_df[c].dtype == float:
            comparison_df[c] = comparison_df[c].round(4)
    for c in diff_df.columns:
        if diff_df[c].dtype == float:
            diff_df[c] = diff_df[c].round(4)
    for c in ds_vs_vix_df.columns:
        if ds_vs_vix_df[c].dtype == float:
            ds_vs_vix_df[c] = ds_vs_vix_df[c].round(4)

    comparison_path = data_dir / "cross_asset_comparison.csv"
    diff_path = data_dir / "cross_asset_overlay_vs_baseline_diffs.csv"
    ds_vs_vix_path = data_dir / "cross_asset_ds_vs_vix_overlay.csv"
    overlap_path = data_dir / "cross_asset_regime_overlap.csv"

    comparison_df.to_csv(comparison_path, index=False)
    diff_df.to_csv(diff_path, index=False)
    ds_vs_vix_df.to_csv(ds_vs_vix_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    return {
        "comparison_df": comparison_df,
        "diff_df": diff_df,
        "ds_vs_vix_df": ds_vs_vix_df,
        "overlap_df": overlap_df,
        "comparison_path": str(comparison_path),
        "diff_path": str(diff_path),
        "ds_vs_vix_path": str(ds_vs_vix_path),
        "overlap_path": str(overlap_path),
    }
