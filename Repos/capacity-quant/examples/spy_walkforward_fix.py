"""
SPY Walk-Forward Optimization Fix

Addresses overfitting by using walk-forward methodology:
- Rolling train/test windows across full dataset
- Optimize capacity on each training window
- Test on immediate subsequent period
- Aggregate results = robust Sharpe

This eliminates "lucky" train/test splits and gives
realistic performance expectations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from capacity_quant.data import fetch_equity_data
from capacity_quant.signals import ma_signal, capacity_ma_signal
from capacity_quant.backtest import backtest_strategy


def walk_forward_optimization(data, train_days=180, test_days=30, step_days=30):
    """
    Walk-forward optimization with rolling windows.
    
    Args:
        data: Full price dataframe
        train_days: Days to train on
        test_days: Days to test on
        step_days: How much to slide window each iteration
    
    Returns:
        List of results per window, aggregate metrics
    """
    results = []
    window_num = 0
    
    # Convert to daily index for easier slicing
    daily_data = data.resample('D').last().dropna()
    
    start_idx = 0
    
    while start_idx + train_days + test_days <= len(daily_data):
        window_num += 1
        
        # Define windows
        train_start = start_idx
        train_end = start_idx + train_days
        test_start = train_end
        test_end = min(train_end + test_days, len(daily_data))
        
        train_data = daily_data.iloc[train_start:train_end]
        test_data = daily_data.iloc[test_start:test_end]
        
        if len(train_data) < 50 or len(test_data) < 10:
            start_idx += step_days
            continue
        
        print(f"\n  Window {window_num}:")
        print(f"    Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
        print(f"    Test:  {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")
        
        # Baseline on test
        baseline_test = ma_signal(test_data, short_window=5, long_window=20)
        baseline_metrics = backtest_strategy(baseline_test)
        
        # Grid search capacity on train
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
            print(f"    ERROR: No valid capacity found")
            start_idx += step_days
            continue
        
        print(f"    Optimal capacity: {best_cap:.2f} (train Sharpe: {best_train_sharpe:.3f})")
        
        # Apply optimal capacity to test
        cap_signal_test = capacity_ma_signal(test_data, capacity=best_cap,
                                             short_window=3, long_window=10)
        cap_metrics = backtest_strategy(cap_signal_test)
        
        print(f"    Test Sharpe: {cap_metrics['sharpe_ratio']:.3f} (baseline: {baseline_metrics['sharpe_ratio']:.3f})")
        
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
            'test_baseline_return': baseline_metrics['total_return']
        })
        
        start_idx += step_days
    
    return pd.DataFrame(results)


def expanding_window_validation(data, min_train_days=180, step_days=60):
    """
    Alternative: Expanding window (train grows, test fixed size).
    """
    results = []
    window_num = 0
    
    daily_data = data.resample('D').last().dropna()
    
    current_train_days = min_train_days
    test_days = 60
    
    while current_train_days + test_days <= len(daily_data):
        window_num += 1
        
        train_data = daily_data.iloc[:current_train_days]
        test_data = daily_data.iloc[current_train_days:current_train_days + test_days]
        
        print(f"\n  Expanding Window {window_num}:")
        print(f"    Train: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
        print(f"    Test:  {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")
        
        # Optimize on growing train
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
            break
        
        # Test
        baseline_test = ma_signal(test_data, short_window=5, long_window=20)
        baseline_metrics = backtest_strategy(baseline_test)
        
        cap_signal_test = capacity_ma_signal(test_data, capacity=best_cap,
                                             short_window=3, long_window=10)
        cap_metrics = backtest_strategy(cap_signal_test)
        
        print(f"    Best cap: {best_cap:.2f}, Test Sharpe: {cap_metrics['sharpe_ratio']:.3f}")
        
        results.append({
            'window': window_num,
            'train_size': len(train_data),
            'best_capacity': best_cap,
            'train_sharpe': best_train_sharpe,
            'test_sharpe': cap_metrics['sharpe_ratio'],
            'test_baseline_sharpe': baseline_metrics['sharpe_ratio']
        })
        
        current_train_days += step_days
    
    return pd.DataFrame(results)


def main():
    print("="*70)
    print("SPY WALK-FORWARD OPTIMIZATION FIX")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGoal: Fix overfitting via walk-forward validation")
    print("Previous result: 58.5% Sharpe decay (OVERFIT)")
    
    # Fetch SPY data
    print("\n[Fetching SPY data...]")
    try:
        data = fetch_equity_data('SPY', period='5y', interval='1d')
        print(f"  Loaded {len(data)} rows: {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        print(f"  Error: {e}")
        return
    
    # Method 1: Rolling Window
    print("\n" + "="*70)
    print("METHOD 1: ROLLING WALK-FORWARD")
    print("="*70)
    print("Train: 180 days, Test: 30 days, Step: 30 days")
    
    rolling_results = walk_forward_optimization(data, train_days=180, test_days=30, step_days=30)
    
    if len(rolling_results) == 0:
        print("ERROR: No valid rolling windows")
        return
    
    print("\n" + "-"*50)
    print("ROLLING RESULTS SUMMARY")
    print("-"*50)
    print(f"  Valid windows: {len(rolling_results)}")
    print(f"  Mean test Sharpe: {rolling_results['test_sharpe'].mean():.3f}")
    print(f"  Std test Sharpe:  {rolling_results['test_sharpe'].std():.3f}")
    print(f"  Median test Sharpe: {rolling_results['test_sharpe'].median():.3f}")
    print(f"  Best window: {rolling_results['test_sharpe'].max():.3f}")
    print(f"  Worst window: {rolling_results['test_sharpe'].min():.3f}")
    
    # Compare to baseline
    mean_baseline = rolling_results['test_baseline_sharpe'].mean()
    mean_capacity = rolling_results['test_sharpe'].mean()
    improvement = (mean_capacity - mean_baseline) / abs(mean_baseline) * 100 if mean_baseline != 0 else 0
    
    print(f"\n  Baseline (MA) mean: {mean_baseline:.3f}")
    print(f"  Capacity mean:      {mean_capacity:.3f}")
    print(f"  Improvement:        {improvement:+.1f}%")
    
    # Win rate
    wins = (rolling_results['test_sharpe'] > rolling_results['test_baseline_sharpe']).sum()
    win_rate = wins / len(rolling_results) * 100
    print(f"  Win rate vs baseline: {win_rate:.0f}% ({wins}/{len(rolling_results)})")
    
    # Method 2: Expanding Window
    print("\n" + "="*70)
    print("METHOD 2: EXPANDING WINDOW")
    print("="*70)
    print("Min train: 180 days, growing, test: 60 days, step: 60 days")
    
    expanding_results = expanding_window_validation(data, min_train_days=180, step_days=60)
    
    if len(expanding_results) > 0:
        print("\n" + "-"*50)
        print("EXPANDING RESULTS SUMMARY")
        print("-"*50)
        print(f"  Valid windows: {len(expanding_results)}")
        print(f"  Mean test Sharpe: {expanding_results['test_sharpe'].mean():.3f}")
        print(f"  Median test Sharpe: {expanding_results['test_sharpe'].median():.3f}")
        
        mean_exp_baseline = expanding_results['test_baseline_sharpe'].mean()
        mean_exp_capacity = expanding_results['test_sharpe'].mean()
        exp_improvement = (mean_exp_capacity - mean_exp_baseline) / abs(mean_exp_baseline) * 100 if mean_exp_baseline != 0 else 0
        print(f"  Improvement vs baseline: {exp_improvement:+.1f}%")
    
    # Final Report
    print("\n" + "="*70)
    print("FINAL ROBUST SHARPE ESTIMATE")
    print("="*70)
    
    # Use rolling as primary (more windows = more robust)
    robust_sharpe = rolling_results['test_sharpe'].median()  # Median less sensitive to outliers
    
    print(f"  Previous (naive split): 2.857 Sharpe")
    print(f"  New (walk-forward):       {robust_sharpe:.3f} Sharpe")
    print(f"  Decay corrected:          {(1 - robust_sharpe/2.857)*100:.1f}%")
    
    if robust_sharpe > 1.5:
        status = "SOLID — Viable for production"
    elif robust_sharpe > 1.0:
        status = "ACCEPTABLE — Borderline for production"
    else:
        status = "WEAK — Not viable for production"
    
    print(f"\n  Status: {status}")
    
    # Save results
    output_dir = '../outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    rolling_results.to_csv(f'{output_dir}/spy_walkforward_rolling.csv', index=False)
    expanding_results.to_csv(f'{output_dir}/spy_walkforward_expanding.csv', index=False)
    print(f"\n  Saved: {output_dir}/spy_walkforward_*.csv")
    
    # Visualization
    print("\n[Generating walk-forward visualization...]")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rolling: Sharpe by window
    ax = axes[0, 0]
    ax.plot(rolling_results['window'], rolling_results['test_sharpe'], 'b-o', label='Capacity', linewidth=2)
    ax.plot(rolling_results['window'], rolling_results['test_baseline_sharpe'], 'r--', label='Baseline', linewidth=2)
    ax.set_xlabel('Window #')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Rolling Walk-Forward: Sharpe by Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
    
    # Rolling: Capacity evolution
    ax = axes[0, 1]
    ax.plot(rolling_results['window'], rolling_results['best_capacity'], 'g-s', linewidth=2)
    ax.set_xlabel('Window #')
    ax.set_ylabel('Optimal Capacity')
    ax.set_title('Rolling: Optimal Capacity per Window')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Distribution of test Sharpes
    ax = axes[1, 0]
    ax.hist(rolling_results['test_sharpe'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=rolling_results['test_sharpe'].mean(), color='red', linestyle='--', 
               label=f"Mean: {rolling_results['test_sharpe'].mean():.3f}")
    ax.axvline(x=rolling_results['test_sharpe'].median(), color='green', linestyle='--',
               label=f"Median: {rolling_results['test_sharpe'].median():.3f}")
    ax.set_xlabel('Test Sharpe Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Test Sharpe Ratios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Expanding window comparison
    if len(expanding_results) > 0:
        ax = axes[1, 1]
        ax.plot(expanding_results['train_size'], expanding_results['test_sharpe'], 'b-o', label='Capacity')
        ax.plot(expanding_results['train_size'], expanding_results['test_baseline_sharpe'], 'r--', label='Baseline')
        ax.set_xlabel('Training Window Size (days)')
        ax.set_ylabel('Test Sharpe')
        ax.set_title('Expanding Window: Test Sharpe vs Train Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = f'{output_dir}/spy_walkforward_fix.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    print("\n" + "="*70)
    print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
