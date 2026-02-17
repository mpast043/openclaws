"""
Phase 4 — Validation.
Compare ds(t) with VIX, realized volatility (20d std), S&P 500 level.
Crisis overlays: 2008, March 2020, 2022. Cross-correlation ds vs VIX lags -30 to +30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, List


def load_vix_and_benchmark(start: str, end: str, benchmark_ticker: str = "^GSPC") -> tuple[pd.Series, pd.Series]:
    """Download VIX and benchmark index (e.g. ^GSPC, QQQ, EFA, EEM, IWM) for date range."""
    import yfinance as yf

    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    bench = yf.download(benchmark_ticker, start=start, end=end, progress=False, auto_adjust=True)
    vix = vix["Close"].squeeze()
    bench = bench["Close"].squeeze()
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    if bench.index.tz is not None:
        bench.index = bench.index.tz_localize(None)
    return vix, bench


def realized_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Rolling window standard deviation of log returns."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret.rolling(window).std()


def cross_correlation_lead_lag(
    x: pd.Series, y: pd.Series, max_lag: int = 30
) -> pd.DataFrame:
    """
    Cross-correlation at lags -max_lag to +max_lag.
    Positive lag = x leads y (x shifted forward).
    Returns DataFrame with columns lag, corr.
    """
    common_x, common_y = x.align(y, join="inner")
    valid = common_x.notna() & common_y.notna()
    x_c = common_x[valid].values
    y_c = common_y[valid].values
    if len(x_c) < 2 * max_lag:
        max_lag = len(x_c) // 4
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = x_c[:lag], y_c[-lag:]
        elif lag > 0:
            a, b = x_c[lag:], y_c[:-lag]
        else:
            a, b = x_c, y_c
        n = min(len(a), len(b))
        a, b = a[-n:], b[-n:]
        if n < 10:
            continue
        r = np.corrcoef(a, b)[0, 1]
        rows.append({"lag": lag, "corr": r})
    df = pd.DataFrame(rows)
    return df


def crisis_regions() -> list[tuple[str, str, str]]:
    """(start_date, end_date, label) for crisis overlays."""
    return [
        ("2008-01-01", "2009-06-30", "2008 crisis"),
        ("2020-02-01", "2020-05-31", "Mar 2020 crash"),
        ("2022-01-01", "2022-12-31", "2022 tightening"),
    ]


def plot_ds_with_overlays(
    ds_series: pd.Series,
    output_path: Union[str, Path],
    vix: Optional[pd.Series] = None,
    rv: Optional[pd.Series] = None,
    spx: Optional[pd.Series] = None,
    drawdown: Optional[pd.Series] = None,
    ds_drop_dates: Optional[List] = None,
    drawdown_crossing_dates: Optional[List] = None,
):
    """ds(t) plot with crisis shaded regions, optional VIX/drawdown, and event-timing vertical lines."""
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    # Ensure datetime index (timezone-naive)
    if not isinstance(ds_series.index, pd.DatetimeIndex):
        ds_series = pd.Series(ds_series.values, index=pd.to_datetime(ds_series.index))
    if ds_series.index.tz is not None:
        ds_series = pd.Series(ds_series.values, index=ds_series.index.tz_localize(None))
    if vix is not None:
        if not isinstance(vix.index, pd.DatetimeIndex):
            vix = pd.Series(vix.values, index=pd.to_datetime(vix.index))
        if vix.index.tz is not None:
            vix = pd.Series(vix.values, index=vix.index.tz_localize(None))
    if drawdown is not None and not isinstance(drawdown.index, pd.DatetimeIndex):
        drawdown = pd.Series(drawdown.values, index=pd.to_datetime(drawdown.index))
    if drawdown is not None and drawdown.index.tz is not None:
        drawdown = pd.Series(drawdown.values, index=drawdown.index.tz_localize(None))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ds_clean = ds_series.dropna()
    first_date = ds_clean.index.min().strftime("%Y-%m-%d")
    last_date = ds_clean.index.max().strftime("%Y-%m-%d")

    x_dates = mdates.date2num(ds_clean.index.to_pydatetime())

    # Crisis shaded regions
    for start, end, label in crisis_regions():
        t0 = mdates.date2num(pd.Timestamp(start))
        t1 = mdates.date2num(pd.Timestamp(end))
        ax1.axvspan(t0, t1, alpha=0.25, color="gray", label="_nolegend_")
    crisis_patch = mpatches.Patch(facecolor="gray", alpha=0.25, edgecolor="gray", label="Crisis periods (2008, Mar 2020, 2022)")

    # Vertical lines: ds-drop events (dashed), drawdown crossings (dotted)
    if ds_drop_dates:
        for d in ds_drop_dates:
            t = mdates.date2num(pd.Timestamp(d))
            ax1.axvline(t, color="C2", linestyle="--", alpha=0.7, linewidth=0.8)
    if drawdown_crossing_dates:
        for d in drawdown_crossing_dates:
            t = mdates.date2num(pd.Timestamp(d))
            ax1.axvline(t, color="C3", linestyle=":", alpha=0.7, linewidth=0.8)

    # Left axis: spectral dimension (blue)
    (line_ds,) = ax1.plot(
        x_dates, ds_clean.values, color="C0", linewidth=1, label="Spectral dimension ds(t)"
    )
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax1.xaxis.get_major_locator()))
    ax1.set_ylabel("Spectral dimension ds", color="C0")
    ax1.set_xlabel("Date")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    legend_handles = [line_ds, crisis_patch]
    legend_labels = ["Spectral dimension ds(t)", "Crisis periods (2008, Mar 2020, 2022)"]
    if ds_drop_dates:
        legend_handles.append(plt.Line2D([0], [0], color="C2", linestyle="--", linewidth=1.5, label="ds-drop"))
        legend_labels.append("ds-drop")
    if drawdown_crossing_dates:
        legend_handles.append(plt.Line2D([0], [0], color="C3", linestyle=":", linewidth=1.5, label="Drawdown crossing"))

    ax2 = None
    if vix is not None:
        ax2 = ax1.twinx()
        common = vix.reindex(ds_clean.index).dropna()
        x_vix = mdates.date2num(common.index.to_pydatetime())
        (line_vix,) = ax2.plot(
            x_vix, common.values, color="C1", alpha=0.85, linewidth=0.8, label="VIX"
        )
        ax2.set_ylabel("VIX", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        legend_handles.append(line_vix)
        legend_labels.append("VIX")

    if drawdown is not None:
        ax_dd = ax1.twinx()
        ax_dd.spines["right"].set_position(("outward", 50 if ax2 is not None else 0))
        dd_aligned = drawdown.reindex(ds_clean.index).dropna()
        if len(dd_aligned) > 0:
            x_dd = mdates.date2num(dd_aligned.index.to_pydatetime())
            (line_dd,) = ax_dd.plot(x_dd, dd_aligned.values * 100, color="C4", alpha=0.7, linewidth=0.7, label="S&P drawdown %")
            ax_dd.set_ylabel("S&P 500 drawdown %", color="C4")
            ax_dd.tick_params(axis="y", labelcolor="C4")
            legend_handles.append(line_dd)
            legend_labels.append("S&P drawdown %")

    ax1.legend(legend_handles, legend_labels, loc="upper left", framealpha=0.9)
    fig.suptitle(f"Spectral dimension ds(t) and VIX — Data: {first_date} to {last_date}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def lead_lag_table(ds_series: pd.Series, vix: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """Cross-correlation table; report lag at max |corr| and value."""
    cc = cross_correlation_lead_lag(ds_series, vix, max_lag=max_lag)
    if cc.empty:
        return cc
    idx_max = cc["corr"].abs().idxmax()
    best = cc.loc[idx_max]
    summary = pd.DataFrame(
        [
            {
                "best_lag_days": int(best["lag"]),
                "corr_at_best": round(best["corr"], 4),
                "interpretation": "ds leads VIX" if best["lag"] > 0 else ("ds lags VIX" if best["lag"] < 0 else "coincident"),
            }
        ]
    )
    return summary


def run_validation(
    ds_series: pd.Series,
    data_dir: Union[str, Path] = "data",
    benchmark_ticker: str = "^GSPC",
) -> dict:
    """
    Load VIX and benchmark index, compute realized vol, plot ds with overlays, cross-corr ds-VIX.
    Returns dict with series, lead_lag summary, and paths.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    start = ds_series.index.min().strftime("%Y-%m-%d")
    end = ds_series.index.max().strftime("%Y-%m-%d")

    vix, spx = load_vix_and_benchmark(start, end, benchmark_ticker=benchmark_ticker)
    rv = realized_volatility(spx, window=20)

    suffix = "_" + benchmark_ticker.replace("^", "").lower() if benchmark_ticker != "^GSPC" else ""
    plot_path = data_dir / f"ds_with_crisis_overlays{suffix}.png"
    plot_ds_with_overlays(ds_series, plot_path, vix=vix)

    cc_df = cross_correlation_lead_lag(ds_series, vix, max_lag=30)
    cc_path = data_dir / f"cross_corr_ds_vix{suffix}.csv"
    cc_df.to_csv(cc_path, index=False)

    lead_lag = lead_lag_table(ds_series, vix, max_lag=30)
    return {
        "ds_series": ds_series,
        "vix": vix,
        "rv": rv,
        "spx": spx,
        "cross_corr": cc_df,
        "lead_lag_summary": lead_lag,
        "plot_path": str(plot_path),
        "cross_corr_path": str(cc_path),
    }
