#!/usr/bin/env python3
"""Extract the largest connected component from a correlation graph and save filtered prices."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.correlations import log_returns, shrink_correlation, correlation_to_adjacency


def connected_components(adj_bool: np.ndarray):
    n = adj_bool.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps = []
    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        comp = []
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


def main():
    parser = argparse.ArgumentParser(description="Extract LCC from price data")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tickers-output", type=Path, required=True)
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    returns = log_returns(prices).dropna(how="any")
    block = returns.tail(args.window).values
    C = np.corrcoef(block.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)
    C_s = shrink_correlation(C, alpha=args.alpha)
    W = correlation_to_adjacency(C_s)
    adj_bool = W > 0.0
    comps = connected_components(adj_bool)
    lcc = max(comps, key=len) if comps else list(range(W.shape[0]))
    tickers = prices.columns
    lcc_tickers = tickers[lcc]

    filtered = prices[lcc_tickers]
    filtered.to_csv(args.output, float_format="%.6f")
    pd.Series(lcc_tickers).to_csv(args.tickers_output, index=False, header=["ticker"])
    print(f"Saved LCC prices ({len(lcc_tickers)} tickers) to {args.output}")


if __name__ == "__main__":
    main()
