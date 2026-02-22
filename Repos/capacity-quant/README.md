# Capacity-Quant

Signal filtering using capacity-constrained information geometry.

## Core Idea

Financial markets generate massive data. Not all observations are equally predictive.
This project applies capacity-constrained filtering to select the most information-rich
data points for signal generation.

## Quick Start

```bash
cd /Users/meganpastore/Clawdbot/Repos/capacity-quant
pip install -r requirements.txt

# Download BTC data and run first backtest
python examples/btc_example.py
```

## Structure

```
capacity_quant/
├── filter.py          # Capacity-constrained data filtering
├── signals.py         # Trading signals (MA, momentum, etc.)
├── backtest.py        # Simple backtest engine
└── data.py            # Data fetching utilities

examples/
├── btc_example.py     # BTC/USD capacity filter demo
└── compare_baselines.py  # Compare capacity vs standard signals
```

## The Capacity Filter

Traditional approach: Use all data points (exhaustive computation).

Capacity approach: Select `C * N` most information-dense observations based on
spectral properties of the data manifold.

```python
from capacity_quant.filter import capacity_filter_time_series

filtered_data = capacity_filter_time_series(
    market_data,
    capacity=0.5,  # Use 50% most informative data points
    window=100     # Analysis window size
)
```

## First Milestone

[ ] BTC data download
[ ] Capacity filter implementation
[ ] Baseline MA signal
[ ] Capacity-enhanced MA signal
[ ] Backtest comparison
[ ] Results documentation
