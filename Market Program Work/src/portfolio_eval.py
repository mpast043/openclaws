"""
Portfolio evaluation: does ds compression signal improve outcomes?
Fixed rules. No parameter optimization. Baseline vs overlay (70/30 when compressed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Union, Dict, Any

ROLLING_DAYS = 252
RF = 0.0
OOS_START = "2016-01-01"
PERCENTILE_DEFAULT = 20
PERCENTILE_STABILITY = (15, 25)  # report all, do not choose best
VIX_HIGH_PERCENTILE = 80  # fixed: VIX_high = VIX > rolling 252d 80th pct


def load_vix(start: str, end: str) -> pd.Series:
    """VIX daily close, same date range as asset."""
    import yfinance as yf
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    vix = vix["Close"].squeeze()
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    return vix


def load_index_prices(start: str, end: str, ticker: str = "^GSPC") -> pd.Series:
    """Index daily prices (^GSPC, QQQ, EFA, EEM, IWM)."""
    import yfinance as yf
    px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    px = px["Close"].squeeze()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily return (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change().dropna()


def cumulative_return(returns: pd.Series) -> pd.Series:
    """Cumulative gross return (1+r_1)*...*(1+r_t)."""
    return (1 + returns).cumprod()


def metrics(returns: pd.Series, rf: float = RF) -> Dict[str, float]:
    """CAGR, annualized vol, Sharpe (rf=0), Sortino, max drawdown, Calmar."""
    if returns.empty or len(returns) < 2:
        return {k: np.nan for k in ["cagr", "ann_vol", "sharpe", "sortino", "max_dd", "calmar"]}
    n = len(returns)
    days_per_year = 252
    total_ret = (1 + returns).prod() - 1
    years = n / days_per_year
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(days_per_year) if returns.std() > 0 else np.nan
    excess = returns - rf / days_per_year
    sharpe = (excess.mean() / returns.std() * np.sqrt(days_per_year)) if returns.std() > 0 else np.nan
    downside = returns[returns < 0]
    down_std = downside.std() * np.sqrt(days_per_year) if len(downside) > 0 and downside.std() > 0 else np.nan
    sortino = (excess.mean() / downside.std() * np.sqrt(days_per_year)) if (len(downside) > 0 and downside.std() > 0) else np.nan
    cum = (1 + returns).cumprod()
    run_max = cum.cummax()
    dd = (cum - run_max) / run_max
    max_dd = dd.min()
    calmar = cagr / (-max_dd) if max_dd < 0 else np.nan
    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
    }


def compression_flag(ds_series: pd.Series, percentile: float, window: int = ROLLING_DAYS) -> pd.Series:
    """Compression = ds < rolling window percentile. Boolean."""
    roll = ds_series.rolling(window, min_periods=window).quantile(percentile / 100.0)
    return (ds_series < roll) & roll.notna()


def vix_high_flag(vix: pd.Series, percentile: float = VIX_HIGH_PERCENTILE, window: int = ROLLING_DAYS) -> pd.Series:
    """VIX_high = VIX > rolling window percentile. Boolean."""
    roll = vix.rolling(window, min_periods=window).quantile(percentile / 100.0)
    return (vix > roll) & roll.notna()


def overlay_returns(
    spy_returns: pd.Series,
    compressed: pd.Series,
    equity_weight: float = 0.7,
) -> pd.Series:
    """If not compressed: 100% SPY. If compressed: equity_weight SPY, (1-equity_weight) cash (0%)."""
    w = np.where(compressed.reindex(spy_returns.index).fillna(False).values, equity_weight, 1.0)
    return pd.Series(spy_returns.values * w, index=spy_returns.index)


def run_portfolio_eval(
    ds_series: pd.Series,
    data_dir: Union[str, Path] = "data",
    benchmark_ticker: str = "^GSPC",
) -> Dict[str, Any]:
    """
    Fixed rules. Period aligned to ds_series. Baseline = 100% SPY. Overlay = 70/30 when compressed.
    Percentile = 20 (default), plus 15 and 25 for stability. Full sample and OOS (2016+) metrics.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    start = ds_series.index.min().strftime("%Y-%m-%d")
    end = ds_series.index.max().strftime("%Y-%m-%d")

    px = load_index_prices(start, end, ticker=benchmark_ticker)
    spy_ret = daily_returns(px)
    common = ds_series.index.intersection(spy_ret.index)
    ds_aligned = ds_series.reindex(common).dropna()
    spy_aligned = spy_ret.reindex(common).dropna()
    common = ds_aligned.index.intersection(spy_aligned.index)
    ds_aligned = ds_aligned.loc[common]
    spy_aligned = spy_aligned.loc[common]

    # VIX (same date range, for overlay and overlap)
    vix = load_vix(start, end)
    vix = vix.reindex(spy_aligned.index).dropna(how="all")
    common_v = spy_aligned.index.intersection(vix.index)
    spy_aligned = spy_aligned.loc[common_v]
    ds_aligned = ds_aligned.reindex(common_v).dropna()
    common = spy_aligned.index.intersection(ds_aligned.index)
    spy_aligned = spy_aligned.loc[common]
    ds_aligned = ds_aligned.loc[common]
    vix = vix.reindex(common).ffill().bfill()

    # Baseline
    baseline_metrics_full = metrics(spy_aligned, rf=RF)
    oos_mask = spy_aligned.index >= OOS_START
    baseline_metrics_oos = metrics(spy_aligned[oos_mask], rf=RF) if oos_mask.any() else {}

    # Overlay at 20th percentile (fixed)
    compressed_20 = compression_flag(ds_aligned, PERCENTILE_DEFAULT, ROLLING_DAYS)
    port_ret_20 = overlay_returns(spy_aligned, compressed_20, equity_weight=0.7)
    overlay_20_full = metrics(port_ret_20.dropna(), rf=RF)
    overlay_20_oos = metrics(port_ret_20[oos_mask].dropna(), rf=RF) if oos_mask.any() else {}

    # Stability: 15th and 25th percentile
    compressed_15 = compression_flag(ds_aligned, PERCENTILE_STABILITY[0], ROLLING_DAYS)
    port_ret_15 = overlay_returns(spy_aligned, compressed_15, equity_weight=0.7)
    overlay_15_full = metrics(port_ret_15.dropna(), rf=RF)
    overlay_15_oos = metrics(port_ret_15[oos_mask].dropna(), rf=RF) if oos_mask.any() else {}

    compressed_25 = compression_flag(ds_aligned, PERCENTILE_STABILITY[1], ROLLING_DAYS)
    port_ret_25 = overlay_returns(spy_aligned, compressed_25, equity_weight=0.7)
    overlay_25_full = metrics(port_ret_25.dropna(), rf=RF)
    overlay_25_oos = metrics(port_ret_25[oos_mask].dropna(), rf=RF) if oos_mask.any() else {}

    # VIX overlay (80th percentile, fixed): VIX_high → 70% asset / 30% cash, else 100%
    vix_high = vix_high_flag(vix, percentile=VIX_HIGH_PERCENTILE, window=ROLLING_DAYS)
    port_ret_vix = overlay_returns(spy_aligned, vix_high, equity_weight=0.7)
    vix_overlay_full = metrics(port_ret_vix.dropna(), rf=RF)
    vix_overlay_oos = metrics(port_ret_vix[oos_mask].dropna(), rf=RF) if oos_mask.any() else {}

    # Metrics table
    def row(name: str, m: dict) -> dict:
        return {"strategy": name, **{k: round(v, 4) if isinstance(v, (int, float)) and not np.isnan(v) else v for k, v in m.items()}}

    rows = [
        row("baseline_full", baseline_metrics_full),
        row("overlay_20pct_full", overlay_20_full),
        row("overlay_15pct_full", overlay_15_full),
        row("overlay_25pct_full", overlay_25_full),
    ]
    if baseline_metrics_oos:
        rows.append(row("baseline_oos", baseline_metrics_oos))
        rows.append(row("overlay_20pct_oos", overlay_20_oos))
        rows.append(row("overlay_15pct_oos", overlay_15_oos))
        rows.append(row("overlay_25pct_oos", overlay_25_oos))

    metrics_df = pd.DataFrame(rows)
    suffix = "_" + benchmark_ticker.replace("^", "").lower() if benchmark_ticker != "^GSPC" else ""
    metrics_path = data_dir / f"portfolio_eval_metrics{suffix}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Differences (baseline vs overlay 20%) — full and OOS
    diff_full = {
        "max_dd_diff": overlay_20_full.get("max_dd", np.nan) - baseline_metrics_full.get("max_dd", np.nan),
        "sharpe_diff": overlay_20_full.get("sharpe", np.nan) - baseline_metrics_full.get("sharpe", np.nan),
        "cagr_diff": overlay_20_full.get("cagr", np.nan) - baseline_metrics_full.get("cagr", np.nan),
    }
    diff_oos = {}
    if overlay_20_oos and baseline_metrics_oos:
        diff_oos = {
            "max_dd_diff_oos": overlay_20_oos.get("max_dd", np.nan) - baseline_metrics_oos.get("max_dd", np.nan),
            "sharpe_diff_oos": overlay_20_oos.get("sharpe", np.nan) - baseline_metrics_oos.get("sharpe", np.nan),
            "cagr_diff_oos": overlay_20_oos.get("cagr", np.nan) - baseline_metrics_oos.get("cagr", np.nan),
        }
    diff_df = pd.DataFrame([{**diff_full, **diff_oos}])
    diff_path = data_dir / f"portfolio_eval_differences{suffix}.csv"
    diff_df.to_csv(diff_path, index=False)

    # STEP 2 — Comparison table: Baseline | ds overlay (20%) | VIX overlay (80%); full + OOS
    def diff_vs_baseline(m_baseline: dict, m_strategy: dict) -> dict:
        return {
            "max_dd_diff": m_strategy.get("max_dd", np.nan) - m_baseline.get("max_dd", np.nan),
            "sharpe_diff": m_strategy.get("sharpe", np.nan) - m_baseline.get("sharpe", np.nan),
            "cagr_diff": m_strategy.get("cagr", np.nan) - m_baseline.get("cagr", np.nan),
        }
    comparison_rows = [
        {"strategy": "baseline", **baseline_metrics_full, "max_dd_diff": np.nan, "sharpe_diff": np.nan, "cagr_diff": np.nan},
        {"strategy": "ds_overlay_20pct", **overlay_20_full, **diff_vs_baseline(baseline_metrics_full, overlay_20_full)},
        {"strategy": "vix_overlay_80pct", **vix_overlay_full, **diff_vs_baseline(baseline_metrics_full, vix_overlay_full)},
    ]
    if baseline_metrics_oos:
        comparison_rows.append({"strategy": "baseline_oos", **baseline_metrics_oos, "max_dd_diff": np.nan, "sharpe_diff": np.nan, "cagr_diff": np.nan})
        comparison_rows.append({"strategy": "ds_overlay_20pct_oos", **overlay_20_oos, **diff_vs_baseline(baseline_metrics_oos, overlay_20_oos)})
        comparison_rows.append({"strategy": "vix_overlay_80pct_oos", **vix_overlay_oos, **diff_vs_baseline(baseline_metrics_oos, vix_overlay_oos)})
    comparison_df = pd.DataFrame(comparison_rows)
    for c in comparison_df.columns:
        if comparison_df[c].dtype == float:
            comparison_df[c] = comparison_df[c].round(4)
    comparison_path = data_dir / f"portfolio_eval_comparison{suffix}.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # STEP 3 — Regime overlap: % days ds active, % days VIX_high active, % overlap (both active)
    n_days = len(common)
    ds_active = compressed_20.reindex(common).fillna(False)
    vix_active = vix_high.reindex(common).fillna(False)
    pct_ds_active = 100.0 * ds_active.sum() / n_days if n_days else np.nan
    pct_vix_high_active = 100.0 * vix_active.sum() / n_days if n_days else np.nan
    both = (ds_active & vix_active).sum()
    pct_overlap = 100.0 * both / n_days if n_days else np.nan
    overlap_df = pd.DataFrame([{
        "pct_days_ds_compression_active": round(pct_ds_active, 2),
        "pct_days_vix_high_active": round(pct_vix_high_active, 2),
        "pct_overlap_both_active": round(pct_overlap, 2),
        "n_days": n_days,
    }])
    overlap_path = data_dir / f"portfolio_eval_overlap{suffix}.csv"
    overlap_df.to_csv(overlap_path, index=False)
    # Append overlap block to comparison CSV so one file has both
    with open(comparison_path, "a") as f:
        f.write("\nREGIME_OVERLAP\n")
    overlap_df.to_csv(comparison_path, mode="a", index=False)

    # Equity curves: baseline vs overlay (20th), highlight compression periods
    cum_baseline = cumulative_return(spy_aligned)
    cum_overlay = cumulative_return(port_ret_20)
    x = mdates.date2num(cum_baseline.index.to_pydatetime())
    fig, ax = plt.subplots(figsize=(12, 5))
    (l0,) = ax.plot(x, cum_baseline.values, color="C0", linewidth=1, label="Baseline (100% SPY)")
    (l1,) = ax.plot(x, cum_overlay.values, color="C1", linewidth=1, label="Overlay (70/30 when compressed)")
    comp_20 = compressed_20.reindex(cum_baseline.index).fillna(False)
    if comp_20.any():
        in_comp = comp_20.values
        starts = np.where(np.diff(np.concatenate([[False], in_comp, [False]]).astype(int)) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[False], in_comp, [False]]).astype(int)) == -1)[0]
        for s, e in zip(starts, ends):
            if s < len(x) and e <= len(x):
                ax.axvspan(x[s], x[e], alpha=0.2, color="gray")
    patch = mpatches.Patch(facecolor="gray", alpha=0.2, edgecolor="none", label="Compression periods")
    ax.legend(handles=[l0, l1, patch], loc="upper left")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylabel("Cumulative gross return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Portfolio evaluation ({benchmark_ticker}): baseline vs overlay (compression = ds < 252d 20th pct)")
    plt.tight_layout()
    plot_path = data_dir / f"portfolio_eval_equity_curves{suffix}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "metrics_df": metrics_df,
        "diff_df": diff_df,
        "comparison_df": comparison_df,
        "overlap_df": overlap_df,
        "metrics_path": str(metrics_path),
        "diff_path": str(diff_path),
        "comparison_path": str(comparison_path),
        "overlap_path": str(overlap_path),
        "plot_path": str(plot_path),
        "baseline_metrics_full": baseline_metrics_full,
        "overlay_20_full": overlay_20_full,
        "overlay_20_oos": overlay_20_oos,
        "vix_overlay_full": vix_overlay_full,
        "vix_overlay_oos": vix_overlay_oos,
        "baseline_metrics_oos": baseline_metrics_oos,
        "pct_ds_active": pct_ds_active,
        "pct_vix_high_active": pct_vix_high_active,
        "pct_overlap": pct_overlap,
    }
