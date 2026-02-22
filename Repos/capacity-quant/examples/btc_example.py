"""
BTC Capacity Filter Demo

First implementation: Download BTC data, apply capacity filter,
compare information density before/after filtering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from capacity_quant.data import fetch_crypto_data
from capacity_quant.filter import (
    capacity_filter_time_series,
    spectral_selectivity,
    local_information_density
)


def main():
    print("=" * 60)
    print("BTC CAPACITY FILTER DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n[1] Fetching BTC data...")
    try:
        data = fetch_crypto_data('BTC-USD', period='6mo', interval='1h')
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Create synthetic data for testing
        print("Using synthetic test data instead...")
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        trend = np.linspace(40000, 60000, 1000)
        noise = np.random.randn(1000) * 2000
        cyclical = np.sin(np.arange(1000) * 0.1) * 5000
        prices = trend + noise + cyclical
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
    
    # Step 2: Analyze spectral properties
    print("\n[2] Analyzing spectral properties...")
    spectral_info = spectral_selectivity(data, price_col='Close')
    print(f"  Dominant frequency: {spectral_info['dominant_frequency']:.6f}")
    print(f"  Spectral entropy: {spectral_info['spectral_entropy']:.4f}")
    print(f"  Normalized entropy: {spectral_info['normalized_entropy']:.4f}")
    print(f"  Recommendation: {spectral_info['recommendation']}")
    
    # Step 3: Capacity filtering
    print("\n[3] Applying capacity filters...")
    
    capacities = [0.25, 0.5, 0.75, 1.0]
    filtered_results = {}
    
    for cap in capacities:
        filtered = capacity_filter_time_series(data, capacity=cap, window=20, price_col='Close')
        filtered_results[cap] = filtered
        print(f"  Capacity {cap:.0%}: {len(filtered)} / {len(data)} points selected ({len(filtered)/len(data):.1%})")
    
    # Step 4: Visualize
    print("\n[4] Generating visualization...")
    fig, axes = plt.subplots(len(capacities) + 1, 1, figsize=(14, 12), sharex=True)
    
    # Full data
    axes[0].plot(data.index, data['Close'], 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title(f'Original Data ({len(data)} points)')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Filtered data
    colors = ['red', 'orange', 'green', 'blue']
    for i, (cap, filtered) in enumerate(filtered_results.items()):
        ax = axes[i + 1]
        # Plot original faint
        ax.plot(data.index, data['Close'], 'gray', alpha=0.2, linewidth=0.5)
        # Highlight filtered points
        ax.scatter(filtered.index, filtered['Close'], c=colors[i], s=10, alpha=0.8)
        ax.set_title(f'Capacity Filtered ({cap:.0%}): {len(filtered)} points selected')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    # Save
    output_path = '../outputs/btc_capacity_filter.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    plt.close()
    
    # Step 5: Information density analysis
    print("\n[5] Information density analysis...")
    prices = data['Close'].values
    info_scores = local_information_density(prices, window=20)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Price with info score coloring
    ax1.plot(data.index, prices, 'b-', alpha=0.5, linewidth=0.5)
    scatter = ax1.scatter(data.index, prices, c=info_scores, cmap='viridis', s=5, alpha=0.7)
    ax1.set_ylabel('Price ($)')
    ax1.set_title('BTC Price Colored by Information Density')
    plt.colorbar(scatter, ax=ax1, label='Information Score')
    
    # Information scores over time
    ax2.plot(data.index, info_scores, 'g-', linewidth=0.5)
    ax2.set_ylabel('Information Score')
    ax2.set_xlabel('Date')
    ax2.set_title('Local Information Density (Spectral Entropy)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = '../outputs/btc_information_density.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path2}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("COMPLETE. Next step: Run compare_baselines.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
