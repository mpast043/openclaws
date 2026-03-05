"""
Event-timing analysis: does ds(t) drop before major drawdowns?
Drawdown from ^GSPC; programmatic crisis onset (-10%, -20%, -30%); ds-drop = ds < rolling median - 1*std; lead table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, List, Optional

ROLLING_DAYS = 252  # 1 year for median/std
DRAWDOWN_THRESHOLDS = (-0.10, -0.20, -0.30)  # -10%, -20%, -30%
DS_DROP_STD = 1.0  # ds < median - 1*sigma
CRISIS_LOOKBACK_DAYS = 365  # max days before crossing to look for ds-drop
FALSE_POSITIVE_WINDOW_DAYS = 30  # ds-drop without crisis within 30 days = false positive


def load_index_drawdown(start: str, end: str, ticker: str = "^GSPC") -> pd.Series:
    """Download index, compute cumulative drawdown DD_t = (P_t / max_{s<=t} P_s) - 1."""
    import yfinance as yf

    px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    px = px["Close"].squeeze()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    cummax = px.cummax()
    drawdown = (px / cummax) - 1.0
    return drawdown


def first_crossing_dates(drawdown: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    First date in each episode when drawdown crosses below threshold.
    Episode = contiguous period below threshold; take first date of each.
    """
    below = drawdown <= threshold
    # First date of each contiguous run of True
    starts = below & ~below.shift(1, fill_value=False)
    return drawdown.index[starts].tolist()


def crisis_onset_table(drawdown: pd.Series) -> pd.DataFrame:
    """
    For each threshold -10%, -20%, -30%, record first crossing date per major episode.
    Merge nearby episodes (same drawdown run) so we get one row per crisis episode per threshold.
    """
    rows = []
    for thresh in DRAWDOWN_THRESHOLDS:
        dates = first_crossing_dates(drawdown, thresh)
        for d in dates:
            rows.append({"drawdown_threshold_pct": thresh * 100, "drawdown_date": d})
    if not rows:
        return pd.DataFrame(columns=["drawdown_threshold_pct", "drawdown_date"])
    df = pd.DataFrame(rows)
    df["drawdown_date"] = pd.to_datetime(df["drawdown_date"])
    return df


def ds_drop_dates(ds_series: pd.Series, window: int = ROLLING_DAYS, n_std: float = DS_DROP_STD) -> pd.DatetimeIndex:
    """
    Structural compression: ds(t) < rolling median - n_std * rolling std.
    Return dates where this holds.
    """
    med = ds_series.rolling(window, min_periods=window // 2).median()
    std = ds_series.rolling(window, min_periods=window // 2).std()
    threshold = med - n_std * std
    below = ds_series < threshold
    # Only where threshold is valid (no nan)
    valid = med.notna() & std.notna() & (std > 0)
    flagged = below & valid
    return ds_series.index[flagged].tolist()


def lead_time_for_episode(
    drawdown_date: pd.Timestamp,
    ds_drop_dates_list: List[pd.Timestamp],
    lookback: int = CRISIS_LOOKBACK_DAYS,
) -> Tuple[Optional[pd.Timestamp], int]:
    """
    Find most recent ds-drop on or before drawdown_date within lookback window.
    Return (ds_drop_date, lead_days). lead_days > 0 means ds dropped before drawdown.
    """
    cutoff = drawdown_date - pd.Timedelta(days=lookback)
    candidates = [d for d in ds_drop_dates_list if cutoff <= d <= drawdown_date]
    if not candidates:
        return None, 0
    last_ds_drop = max(candidates)
    lead = (drawdown_date - last_ds_drop).days
    return last_ds_drop, lead


def run_event_timing(
    ds_series: pd.Series,
    data_dir: Union[str, Path] = "data",
    benchmark_ticker: str = "^GSPC",
) -> dict:
    """
    Run event-timing analysis. Returns dict with drawdown series, ds_drop_dates,
    onset_table, lead_table, summary (avg_lead_days, pct_ds_leads, false_positives_per_year),
    and path to saved CSV.
    """
    data_dir = Path(data_dir)
    start = ds_series.index.min().strftime("%Y-%m-%d")
    end = ds_series.index.max().strftime("%Y-%m-%d")

    drawdown = load_index_drawdown(start, end, ticker=benchmark_ticker)
    drawdown = drawdown.reindex(ds_series.index)

    onset_df = crisis_onset_table(drawdown)
    ds_drops = ds_drop_dates(ds_series, window=ROLLING_DAYS, n_std=DS_DROP_STD)
    ds_drop_list = sorted(pd.to_datetime(ds_drops).tolist())

    # Build lead table: Crisis (label), Drawdown Threshold, ds_drop_date, Drawdown date, Lead (days)
    lead_rows = []
    for _, row in onset_df.iterrows():
        thresh = row["drawdown_threshold_pct"]
        dd_date = pd.Timestamp(row["drawdown_date"])
        ds_date, lead = lead_time_for_episode(dd_date, ds_drop_list, lookback=CRISIS_LOOKBACK_DAYS)
        crisis_label = f"DD_{thresh:.0f}pct_{dd_date.strftime('%Y-%m')}"
        lead_rows.append({
            "crisis": crisis_label,
            "drawdown_threshold_pct": thresh,
            "ds_drop_date": ds_date.strftime("%Y-%m-%d") if ds_date is not None else None,
            "drawdown_date": dd_date.strftime("%Y-%m-%d"),
            "lead_days": lead,
        })
    lead_df = pd.DataFrame(lead_rows)

    # Summary: average lead (among those where ds leads), % crises where ds leads, false positives per year
    if not lead_df.empty and "lead_days" in lead_df.columns:
        leads = lead_df["lead_days"].dropna()
        positive_leads = leads[leads > 0]
        avg_lead = float(positive_leads.mean()) if len(positive_leads) > 0 else np.nan
        n_crises = len(lead_df)
        n_ds_leads = (leads > 0).sum()
        pct_ds_leads = 100.0 * n_ds_leads / n_crises if n_crises else np.nan
    else:
        avg_lead = np.nan
        pct_ds_leads = np.nan
        n_crises = 0

    # False positives: ds-drop dates with no drawdown crossing within next FALSE_POSITIVE_WINDOW_DAYS
    onset_dates = set(pd.to_datetime(onset_df["drawdown_date"]).tolist())
    false_pos = 0
    for d in ds_drop_list:
        window_end = d + pd.Timedelta(days=FALSE_POSITIVE_WINDOW_DAYS)
        has_crisis = any(dd <= window_end and dd >= d for dd in onset_dates)
        if not has_crisis:
            false_pos += 1
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    false_pos_per_year = false_pos / years if years > 0 else np.nan

    summary = {
        "avg_lead_days": avg_lead,
        "pct_crises_ds_leads": pct_ds_leads,
        "false_positives_per_year": false_pos_per_year,
        "n_crisis_onset_events": n_crises,
        "n_ds_drop_events": len(ds_drop_list),
    }

    suffix = "_" + benchmark_ticker.replace("^", "").lower() if benchmark_ticker != "^GSPC" else ""
    out_path = data_dir / f"event_timing_summary{suffix}.csv"
    lead_df.to_csv(out_path, index=False)

    return {
        "drawdown": drawdown,
        "ds_drop_dates": ds_drop_list,
        "onset_table": onset_df,
        "lead_table": lead_df,
        "summary": summary,
        "summary_for_memo": {
            "avg_lead_days": avg_lead,
            "pct_ds_leads": pct_ds_leads,
            "false_positives_per_year": false_pos_per_year,
        },
        "csv_path": str(out_path),
    }
