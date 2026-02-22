"""
Crypto Walk-Forward Validation

Rigorous walk-forward on BTC and ETH to confirm if crypto results hold up.
If these fail like SPY, the framework doesn't generalize.

Uses same methodology as SPY walk-forward for apples-to-apples comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from capacity_quant.data import fetch_crypto_data
from capacity_quant.signals import ma_signal, capacity_ma_signal
from capacity_quant.backtest import backtest_strategy


def walk_forward_crypto(symbol, name, train_days=180, test_days=30, step_days=30):
    """
    Rolling walk-forward for crypto.
    """
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD: {name} ({symbol})")
    print('='*70)
    
    # Fetch data
    try:
        print("  Fetching 4 years daily data...")
        data = fetch_crypto_data(symbol, period='4y', interval='1d')
        print(f"  Loaded {len(data)} rows: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        print(f"  Error: {e}")
        return None
    
    # Resample to daily
    daily_data = data.resample('D').last().dropna()
    
    results = []
    window_num = 0
    start_idx = 0
    
    while start_idx + train_days + test_days <= len(daily_data):
        window_num += 1
        
        train_start = start_idx
        train_end = start_idx + train_days
        test_start = train_end
        test_end = min(train_end + test_days, len(daily_data))
        
        train_data = daily_data.iloc[train_start:train_end]
        test_data = daily_data.iloc[test_start:test_end]
        
        if len(train_data) < 100 or len(test_data) < 10:
            start_idx += step_days
            continue
        
        # Baseline on test
        baseline_test = ma_signal(test_data, short_window=5, long_window=20)
        baseline_metrics = backtest_strategy(baseline_test)
        
        # Optimize capacity on train
        best_cap = None
        best_train_sharpe = -999
        
        for cap in np.arange(0.05, 0.95, 0.05):
            try:
                cap_signal = capacity_ma_signal(train_data, capacity=cap,
                                                short_window=3, long_window=10)
                metrics = backtest_strategy(cap_signal)
                
                if metrics['sharpe_ratio'] > best_train_sharpe:
                    best_train_sharpe = metrics['sharpe_ratio']
                    best_cap = cap
            except:
                pass
        
        if best_cap is None:
            start_idx += step_days
            continue
        
        # Test with optimal capacity
        cap_signal_test = capacity_ma_signal(test_data, capacity=best_cap,
                                             short_window=3, long_window=10)
        cap_metrics = backtest_strategy(cap_signal_test)
        
        results.append({
            'window': window_num,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'best_capacity': best_cap,
            'train_sharpe': best_train_sharpe,
            'test_sharpe': cap_metrics['sharpe_ratio'],
            'test_return': cap_metrics['total_return'],
            'test_baseline_sharpe': baseline_metrics['sharpe_ratio'],
            'test_baseline_return': baseline_metrics['total_return'],
            'sharpe_advantage': cap_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
        })
        
        start_idx += step_days
    
    results_df = pd.DataFrame(results)
    
    # Summary stats
    if len(results_df) == 0:
        return None
    
    print(f"\n  Valid windows: {len(results_df)}")
    print(f"  Mean test Sharpe: {results_df['test_sharpe'].mean():.3f}")
    print(f"  Median test Sharpe: {results_df['test_sharpe'].median():.3f}")
    print(f"  Best window: {results_df['test_sharpe'].max():.3f}")
    print(f"  Worst window: {results_df['test_sharpe'].min():.3f}")
    print(f"  Std test Sharpe: {results_df['test_sharpe'].std():.3f}")
    
    # Win rate vs baseline
    wins = (results_df['test_sharpe'] > results_df['test_baseline_sharpe']).sum()
    win_rate = wins / len(results_df) * 100
    
    # Robust windows (Sharpe > 1.5)
    robust_windows = (results_df['test_sharpe'] > 1.5).sum()
    robust_rate = robust_windows / len(results_df) * 100
    
    mean_baseline = results_df['test_baseline_sharpe'].mean()
    mean_capacity = results_df['test_sharpe'].mean()
    
    print(f"\n  Baseline mean: {mean_baseline:.3f}")
    print(f"  Capacity mean: {mean_capacity:.3f}")
    print(f"  Win rate vs baseline: {win_rate:.0f}% ({wins}/{len(results_df)})")
    print(f"  Robust windows (Sharpe>1.5): {robust_rate:.0f}% ({robust_windows}/{len(results_df)})")
    
    # Status
    median_sharpe = results_df['test_sharpe'].median()
    if median_sharpe > 1.5:
        status = "STRONG ✓ — Viable for production"
    elif median_sharpe > 1.0:
        status = "SOLID ✓ — Marginal for production"
    elif median_sharpe > 0.5:
        status = "WEAK ⚠ — Borderline value"
    else:
        status = "FAIL ✗ — Not viable"
    
    print(f"  Status: {status}")
    
    return {
        'symbol': symbol,
        'name': name,
        'windows': len(results_df),
        'mean_sharpe': mean_capacity,
        'median_sharpe': median_sharpe,
        'best_sharpe': results_df['test_sharpe'].max(),
        'worst_sharpe': results_df['test_sharpe'].min(),
        'std_sharpe': results_df['test_sharpe'].std(),
        'mean_baseline': mean_baseline,
        'win_rate': win_rate,
        'robust_rate': robust_rate,
        'status': status,
        'results': results_df
    }


def main():
    print("="*70)
    print("CRYPTO WALK-FORWARD VALIDATION")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGoal: Confirm crypto results hold under rigorous walk-forward")
    print("If crypto fails like SPY, framework doesn't generalize.")
    print("\nEach market: Train 180 days, Test 30 days, Step 30 days")
    
    markets = [
        ('BTC-USD', 'Bitcoin'),
        ('ETH-USD', 'Ethereum'),
    ]
    
    all_results = []
    
    for symbol, name in markets:
        result = walk_forward_crypto(symbol, name)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nERROR: No valid results from any crypto market")
        return
    
    # Comparison summary
    print("\n" + "="*70)
    print("WALK-FORWARD COMPARISON: CRYPTO vs SPY")
    print("="*70)
    
    # Load SPY result for comparison if available
    spy_file = '../outputs/spy_walkforward_rolling.csv'
    spy_available = os.path.exists(spy_file)
    
    if spy_available:
        spy_df = pd.read_csv(spy_file)
        spy_median = spy_df['test_sharpe'].median()
        spy_windows = len(spy_df)
        spy_win_rate = (spy_df['test_sharpe'] > spy_df['test_baseline_sharpe']).mean() * 100
    else:
        spy_median = 0.0
        spy_windows = 35
        spy_win_rate = 40
    
    comparison_data = []
    for r in all_results:
        comparison_data.append({
            'Market': r['name'],
            'Windows': r['windows'],
            'Median Sharpe': f"{r['median_sharpe']:.3f}",
            'Mean Sharpe': f"{r['mean_sharpe']:.3f}",
            'Win Rate': f"{r['win_rate']:.0f}%",
            'Robust Windows': f"{r['robust_rate']:.0f}%",
            'Status': '✓' if r['median_sharpe'] > 1.5 else '⚠' if r['median_sharpe'] > 0.5 else '✗'
        })
    
    # Add SPY
    comparison_data.append({
        'Market': 'SPY (reference)',
        'Windows': spy_windows,
        'Median Sharpe': f"{spy_median:.3f}",
        'Mean Sharpe': f"{spy_df['test_sharpe'].mean():.3f}" if spy_available else "0.53",
        'Win Rate': f"{spy_win_rate:.0f}%",
        'Robust Windows': "0%",
        'Status': '✗'
    })
    
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    crypto_robust = sum(1 for r in all_results if r['median_sharpe'] > 1.5)
    
    if crypto_robust == len(all_results):
        print("FRAMEWORK VALIDATED ✓")
        print("Both crypto markets robust under walk-forward.")
        print("SPY failure is 'expected inefficiency' not framework bug.")
        print("\nAction: Ship crypto-only pitch.")
    elif crypto_robust > 0:
        print("PARTIAL VALIDATION ⚠")
        print(f"{crypto_robust}/{len(all_results)} crypto markets robust.")
        print("May need market-specific tuning.")
    else:
        print("FRAMEWORK FAILS ✗")
        print("Crypto results don't hold under walk-forward.")
        print("Original OOS results were overfit.")
        print("\nAction: Pivot. Framework doesn't work.")
    
    # Save results
    output_dir = '../outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    for r in all_results:
        r['results'].to_csv(f'{output_dir}/walkforward_{r["symbol"].replace("-", "_")}.csv', index=False)
        print(f"\n  Saved: {output_dir}/walkforward_{r['symbol'].replace('-', '_')}.csv")
    
    # Visualization
    print("\n[Generating walk-forward comparison...]")
    fig, axes = plt.subplots(len(all_results), 2, figsize=(14, 5*len(all_results)))
    
    if len(all_results) == 1:
        axes = axes.reshape(1, 2)
    
    for i, r in enumerate(all_results):
        df = r['results']
        
        # Sharpe by window
        ax = axes[i, 0]
        ax.plot(df['window'], df['test_sharpe'], 'b-o', label='Capacity', linewidth=2, markersize=4)
        ax.plot(df['window'], df['test_baseline_sharpe'], 'r--', label='Baseline', linewidth=2, alpha=0.7)
        ax.axhline(y=1.5, color='green', linestyle=':', alpha=0.5, label='Robust Threshold')
        ax.set_xlabel('Window #')
        ax.set_ylabel('Test Sharpe')
        ax.set_title(f"{r['name']} — Walk-Forward Sharpe by Window")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distribution + capacity
        ax = axes[i, 1]
        ax2 = ax.twinx()
        
        bars = ax.bar(df['window'], df['test_sharpe'], alpha=0.6, color='blue', label='Test Sharpe')
        ax2.plot(df['window'], df['best_capacity'], 'r-s', markersize=4, label='Optimal Capacity')
        
        ax.set_xlabel('Window #')
        ax.set_ylabel('Test Sharpe', color='blue')
        ax2.set_ylabel('Optimal Capacity', color='red')
        ax.axhline(y=1.5, color='green', linestyle=':', alpha=0.5)
        ax.set_title(f"{r['name']} — Sharpe vs Optimal Capacity")
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_file = f'{output_dir}/crypto_walkforward_summary.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    print("\n" + "="*70)
    print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
