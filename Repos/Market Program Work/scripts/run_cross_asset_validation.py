#!/usr/bin/env python3
"""
Cross-asset structural validation: ds overlay vs VIX overlay across EFA, EEM, EWJ, VGK, GLD, DBC, USO.
Fixed rules only. No tuning. Requires data/ds_series.csv from main pipeline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.cross_asset_validation import run_cross_asset_validation


def main():
    data_dir = PROJECT_ROOT / "data"
    ds_path = data_dir / "ds_series.csv"
    if not ds_path.exists():
        print("Missing data/ds_series.csv. Run the main pipeline first.")
        sys.exit(1)
    df = pd.read_csv(ds_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ds_series = df.iloc[:, 0] if df.ndim > 1 else df.squeeze()

    out = run_cross_asset_validation(ds_series, data_dir=data_dir)

    print("CROSS_ASSET_COMPARISON")
    print(out["comparison_df"].to_string(index=False))
    print("\nOVERLAY_VS_BASELINE_DIFFS")
    print(out["diff_df"].to_string(index=False))
    print("\nDS_VS_VIX_OVERLAY")
    print(out["ds_vs_vix_df"].to_string(index=False))
    print("\nREGIME_OVERLAP")
    print(out["overlap_df"].to_string(index=False))
    print("\nSaved:", out["comparison_path"], out["diff_path"], out["ds_vs_vix_path"], out["overlap_path"])


if __name__ == "__main__":
    main()
