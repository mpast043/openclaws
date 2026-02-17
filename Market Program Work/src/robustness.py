"""
Phase 5 — Robustness.
Sensitivity to: window (40, 60, 90), shrinkage α (0.01, 0.05, 0.10), signed vs |C|.
Report whether qualitative behavior persists.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

from .correlations import log_returns
from .spectral import compute_ds_series


def run_robustness(
    prices: pd.DataFrame,
    data_dir: Union[str, Path] = "data",
) -> pd.DataFrame:
    """
    Run ds pipeline for (window, alpha, use_abs) combinations.
    Returns summary table: config columns + correlation with baseline ds (window=60, alpha=0.05, use_abs=True).
    """
    data_dir = Path(data_dir)
    returns = log_returns(prices)

    # Baseline: window=60, alpha=0.05, use_abs=True
    print("    baseline (window=60, α=0.05)...")
    baseline = pd.Series(
        {
            d: v
            for d, v in compute_ds_series(
                returns, window=60, alpha=0.05, use_abs=True, progress_interval=500
            )
            if v is not None
        }
    )
    baseline = baseline.sort_index()

    rows = []
    for i, (window, alpha, use_abs) in enumerate(
        [(w, a, u) for w in [40, 60, 90] for a in [0.01, 0.05, 0.10] for u in [True, False]]
    ):
        print(f"    robustness {i + 1}/18 (window={window}, α={alpha}, use_abs={use_abs})...")
        series = pd.Series(
            {
                d: v
                for d, v in compute_ds_series(
                    returns,
                    window=window,
                    alpha=alpha,
                    use_abs=use_abs,
                    progress_interval=None,  # quiet in robustness
                )
                if v is not None
            }
        )
        series = series.sort_index()
        common = baseline.align(series, join="inner")
        valid = common[0].notna() & common[1].notna()
        if valid.sum() < 20:
            corr_baseline = np.nan
        else:
            corr_baseline = np.corrcoef(common[0][valid], common[1][valid])[0, 1]
        rows.append(
            {
                "window": window,
                "alpha": alpha,
                "use_abs": use_abs,
                "corr_with_baseline": round(corr_baseline, 4),
                "mean_ds": round(series.mean(), 4),
                "n_dates": len(series),
            }
        )

    out = pd.DataFrame(rows)
    out_path = data_dir / "robustness_sensitivity.csv"
    out.to_csv(out_path, index=False)
    return out
