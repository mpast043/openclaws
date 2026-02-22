"""
Compare Baseline vs Capacity-Filtered Trading Signals

This script:
1. Fetches BTC data
2. Runs baseline MA crossover strategy
3. Runs capacity-filtered MA crossover 
4. Backtests both and compares metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from capacity_quant.data import fetch_crypto_data
from capacity_quant.signals import ma_signal, capacity_ma_signal
from capacity_quant.backtest import backtest_strategy, compare_strategies


def main():
    print("=" * 70)
    print("BASELINE vs CAPACITY-FILTERED STRATEGY COMPARISON")
    print("=" * 70)
    
    # Step 1: Fetch data
    print("\n[1] Loading market data...")
    try:
        data = fetch_crypto_data('BTC-USD', period='1y', interval='1h')
    except Exception as e:
        print(f"Error: {e}")
        print("Creating synthetic data for demo...")
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=2000, freq='H')
        returns = np.random.randn(2000) * 0.02 + 0.0001
        prices = 50000 * np.exp(np.cumsum(returns))
        data = pd.DataFrame({
            'Open': prices * 0.995,
            'High': prices * 1.015,
            'Low': prices * 0.985,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 2000)
        }, index=dates)
    
    print(f"Data shape: {data.shape}")
    
    # Step 2: Generate signals
    print("\n[2] Generating trading signals...")
    
    # Baseline: Regular MA
    baseline = ma_signal(data, short_window=5, long_window=20)
    print(f"  Baseline MA: {len(baseline)} signals")
    
    # Capacity-filtered: Multiple capacities
    capacities = [0.3, 0.5, 0.7]
    capacity_signals = {}
    
    for cap in capacities:
        cap_signal = capacity_ma_signal(data, capacity=cap, short_window=3, long_window=10)
        capacity_signals[f"Cap_{int(cap*100)}%"] = cap_signal
        print(f"  Capacity {cap:.0%}: {len(cap_signal)} signals ({len(cap_signal)/len(data):.1%} of data)")
    
    # Step 3: Backtest
    print("\n[3] Running backtests...")
    
    all_strategies = {'Baseline_MA': baseline, **capacity_signals}
    results = compare_strategies(data, all_strategies, initial_capital=10000.0)
    
    print("\n" + "-" * 70)
    print("BACKTEST RESULTS")
    print("-" * 70)
    print(results.to_string(index=False))
    print("-" * 70)
    
    # Step 4: Detailed analysis
    print("\n[4] Detailed metrics...")
    
    for name, strat_data in all_strategies.items():
        metrics = backtest_strategy(strat_data)
        print(f"\n{name}:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  # Trades: {metrics['num_trades']}")
    
    # Step 5: Visualization
    print("\n[5] Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Equity curves
    ax1 = axes[0, 0]
    for name, strat_data in all_strategies.items():
        metrics = backtest_strategy(strat_data)
        equity = metrics['equity_curve']
        ax1.plot(equity.index, equity.values / equity.values[0], 
                 label=name, linewidth=1.5, alpha=0.8)
    ax1.set_title('Equity Curves')
    ax1.set_ylabel('Portfolio Value (normalized)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal counts
    ax2 = axes[0, 1]
    names = list(all_strategies.keys())
    counts = [len(s) for s in all_strategies.values()]
    colors = ['blue'] + ['orange', 'green', 'red'][:len(capacities)]
    ax2.bar(names, counts, color=colors, alpha=0.7)
    ax2.set_title('Signal Count by Strategy')
    ax2.set_ylabel('# Data Points')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Sharpe comparison
    ax3 = axes[1, 0]
    sharpes = [backtest_strategy(s)['sharpe_ratio'] for s in all_strategies.values()]
    ax3.bar(names, sharpes, color=colors, alpha=0.7)
    ax3.set_title('Sharpe Ratio Comparison')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Shasrpe=1')
    ax3.legend()
    
    # Plot 4: Return vs Drawdown scatter
    ax4 = axes[1, 1]
    for i, (name, strat_data) in enumerate(all_strategies.items()):
        m = backtest_strategy(strat_data)
        ax4.scatter(m['max_drawdown']*100, m['total_return']*100, 
                   s=200, c=colors[i], alpha=0.7, label=name)
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Total Return (%)')
    ax4.set_title('Return vs Risk')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = '../outputs/strategy_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Step 6: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_sharpe = results.loc[results['sharpe_ratio'].idxmax()]
    best_return = results.loc[results['total_return'].idxmax()]
    
    print(f"Best Sharpe Ratio: {best_sharpe['sharpe_ratio']:.3f} ({best_sharpe['strategy']})")
    print(f"Best Return: {best_return['total_return']:.2%} ({best_return['strategy']})")
    
    baseline_sharpe = results[results['strategy'] == 'Baseline_MA']['sharpe_ratio'].values[0]
    if best_sharpe['strategy'] != 'Baseline_MA':
        improvement = (best_sharpe['sharpe_ratio'] - baseline_sharpe) / abs(baseline_sharpe) * 100
        print(f"\nCapacity improvement: {improvement:+.1f}% vs baseline")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("  1. Tune capacity parameter (try 0.2-0.8 range)")
    print("  2. Test on multiple markets (ETH, SPY, EUR/USD)")
    print("  3. Add position sizing based on signal strength")
    print("  4. Out-of-sample validation (train 2022-2023, test 2024)")
    print("=" * 70)


if __name__ == '__main__':
    main()
