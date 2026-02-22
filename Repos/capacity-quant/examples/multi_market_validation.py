"""
Multi-Market Validation + Capacity Grid Search

Tests capacity filtering across:
- ETH-USD (crypto)
- SPY (equities)
- EURUSD=X (forex)

Grid searches capacity values: 0.1, 0.15, 0.2, ..., 0.9
Saves best capacity per market.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from capacity_quant.data import fetch_crypto_data, fetch_equity_data, fetch_forex_data
from capacity_quant.signals import ma_signal, capacity_ma_signal
from capacity_quant.backtest import backtest_strategy


def grid_search_capacity(data, symbol, capacities=np.arange(0.1, 1.0, 0.05)):
    """
    Grid search capacity values and return best.
    """
    results = []
    
    print(f"\n  Grid search: {len(capacities)} capacity values...")
    
    for cap in capacities:
        try:
            cap_signal = capacity_ma_signal(data, capacity=cap, short_window=3, long_window=10)
            metrics = backtest_strategy(cap_signal)
            
            results.append({
                'symbol': symbol,
                'capacity': cap,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': metrics['num_trades'],
                'num_points': len(cap_signal)
            })
            
        except Exception as e:
            print(f"    Cap {cap:.2f} failed: {e}")
    
    return pd.DataFrame(results)


def validate_market(symbol, name, fetch_func):
    """
    Validate a single market with baseline + grid search.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING: {name} ({symbol})")
    print('='*60)
    
    # Fetch data
    try:
        data = fetch_func(symbol, period='1y', interval='1h')
        print(f"  Loaded {len(data)} rows")
    except Exception as e:
        print(f"  ERROR fetching {symbol}: {e}")
        # Use synthetic data
        print("  Using synthetic data for demo...")
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range('2024-01-01', periods=2000, freq='H')
        returns = np.random.randn(2000) * 0.015 + 0.00005
        prices = 100 * np.exp(np.cumsum(returns))
        data = pd.DataFrame({
            'Open': prices * 0.995,
            'High': prices * 1.015,
            'Low': prices * 0.985,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 2000)
        }, index=dates)
    
    # Baseline
    print("  Running baseline MA...")
    baseline = ma_signal(data, short_window=5, long_window=20)
    baseline_metrics = backtest_strategy(baseline)
    print(f"    Sharpe: {baseline_metrics['sharpe_ratio']:.3f}, Return: {baseline_metrics['total_return']:.1%}")
    
    # Grid search
    grid_results = grid_search_capacity(data, symbol)
    
    if len(grid_results) == 0:
        print("  No valid capacity results!")
        return None
    
    # Best by Sharpe
    best_sharpe = grid_results.loc[grid_results['sharpe_ratio'].idxmax()]
    print(f"\n  Best capacity: {best_sharpe['capacity']:.2f}")
    print(f"    Sharpe: {best_sharpe['sharpe_ratio']:.3f} (vs baseline {baseline_metrics['sharpe_ratio']:.3f})")
    print(f"    Return: {best_sharpe['total_return']:.1%}")
    print(f"    Drawdown: {best_sharpe['max_drawdown']:.1%}")
    
    improvement = (best_sharpe['sharpe_ratio'] - baseline_metrics['sharpe_ratio']) / \
                  (abs(baseline_metrics['sharpe_ratio']) + 0.001) * 100
    print(f"    Improvement: {improvement:+.1f}%")
    
    return {
        'symbol': symbol,
        'name': name,
        'baseline_sharpe': baseline_metrics['sharpe_ratio'],
        'baseline_return': baseline_metrics['total_return'],
        'best_capacity': best_sharpe['capacity'],
        'best_sharpe': best_sharpe['sharpe_ratio'],
        'best_return': best_sharpe['total_return'],
        'improvement_pct': improvement,
        'grid_results': grid_results
    }


def main():
    print("="*70)
    print("MULTI-MARKET VALIDATION + CAPACITY GRID SEARCH")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    markets = [
        ('ETH-USD', 'Ethereum', fetch_crypto_data),
        ('SPY', 'SPY ETF', fetch_equity_data),
        ('EURUSD=X', 'EUR/USD', fetch_forex_data),
    ]
    
    all_results = []
    
    for symbol, name, fetch_func in markets:
        result = validate_market(symbol, name, fetch_func)
        if result:
            all_results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Best Capacity Per Market")
    print("="*70)
    
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Market': r['name'],
            'Symbol': r['symbol'],
            'Baseline Sharpe': f"{r['baseline_sharpe']:.3f}",
            'Best Capacity': f"{r['best_capacity']:.2f}",
            'Best Sharpe': f"{r['best_sharpe']:.3f}",
            'Improvement': f"{r['improvement_pct']:+.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    output_dir = '../outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary_df.to_csv(f'{output_dir}/multi_market_summary.csv', index=False)
    print(f"\n  Saved: {output_dir}/multi_market_summary.csv")
    
    # Save detailed grid results
    for r in all_results:
        grid_file = f"{output_dir}/grid_{r['symbol'].replace('=', '').replace('-', '_')}.csv"
        r['grid_results'].to_csv(grid_file, index=False)
        print(f"  Saved: {grid_file}")
    
    # Visualization
    print("\n[Plotting capacity curves...]")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, r in enumerate(all_results):
        ax = axes[i]
        grid = r['grid_results']
        
        ax.plot(grid['capacity'], grid['sharpe_ratio'], 'b-o', linewidth=2, markersize=4)
        ax.axhline(y=r['baseline_sharpe'], color='r', linestyle='--', 
                  label=f"Baseline ({r['baseline_sharpe']:.3f})")
        ax.axvline(x=r['best_capacity'], color='g', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Capacity')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f"{r['name']} ({r['symbol']})\nBest: {r['best_capacity']:.2f} → {r['best_sharpe']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary bar chart
    ax = axes[3]
    names = [r['name'] for r in all_results]
    improvements = [r['improvement_pct'] for r in all_results]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax.bar(names, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Improvement %')
    ax.set_title('Sharpe Improvement vs Baseline')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_file = f'{output_dir}/multi_market_validation.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {plot_file}")
    plt.close()
    
    # Best overall
    print("\n" + "="*70)
    print("BEST PERFORMER")
    print("="*70)
    best = max(all_results, key=lambda x: x['best_sharpe'])
    print(f"Market: {best['name']}")
    print(f"Best capacity: {best['best_capacity']:.2f}")
    print(f"Sharpe: {best['baseline_sharpe']:.3f} → {best['best_sharpe']:.3f}")
    print(f"Improvement: {best['improvement_pct']:+.1f}%")
    
    print("\n" + "="*70)
    print(f"Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
