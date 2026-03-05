#!/usr/bin/env python3
"""
Polymarket Arbitrage Profit Projection - Realistic Scenarios

Based on actual data from live scans and paper trading.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any


random.seed(42)  # Reproducible results


@dataclass
class Scenario:
    name: str
    description: str
    starting_capital: float
    
    scans_per_day: int = 24        # 24 scans/day = 1/hour
    opps_per_scan: int = 5         # 5 opportunities found per scan
    fill_rate: float = 0.70          # 70% actually fill (from paper)
    
    avg_edge: float = 0.04           # $0.04 per share ($4 per 100 shares)
    avg_shares: int = 200            # 200 shares = ~$150-170 deployed
    fee_rate: float = 0.02           # 2% total fees
    
    max_positions: int = 5           # Concurrent position limit
    hold_hours: int = 24             # Average hold time
    hold_hours_std: int = 8          # Variance in hold time
    
    capital_allocation: float = 0.25   # 25% per trade max


SCENARIOS = {
    "conservative": Scenario(
        name="CONSERVATIVE (Bond)",
        description="Hold to expiry, 3 positions max, 48hr average hold",
        starting_capital=200.0,
        scans_per_day=24,
        opps_per_scan=3,
        fill_rate=0.60,
        avg_edge=0.035,
        max_positions=3,
        hold_hours=48,
        hold_hours_std=12,
    ),
    "realistic": Scenario(
        name="REALISTIC (Hybrid)",
        description="Swing when possible, 24hr avg hold, 5 positions",
        starting_capital=200.0,
        scans_per_day=24,
        opps_per_scan=5,
        fill_rate=0.70,
        avg_edge=0.040,
        max_positions=5,
        hold_hours=24,
        hold_hours_std=8,
    ),
    "optimistic": Scenario(
        name="OPTIMISTIC (Active)",
        description="8hr swing exits, 10 positions, 80% fill rate",
        starting_capital=200.0,
        scans_per_day=48,
        opps_per_scan=8,
        fill_rate=0.80,
        avg_edge=0.045,
        max_positions=10,
        hold_hours=8,
        hold_hours_std=4,
    ),
    "moon": Scenario(
        name="MOON (Unlikely)",
        description="4hr hold, 20 positions, perfect execution",
        starting_capital=200.0,
        scans_per_day=96,
        opps_per_scan=12,
        fill_rate=0.95,
        avg_edge=0.055,
        max_positions=20,
        hold_hours=4,
        hold_hours_std=2,
        fee_rate=0.015,
    ),
    "full_reinvest": Scenario(
        name="FULL REINVEST (Aggressive)",
        description="100% reinvest, 8hr hold, 15 positions, 80% fill, 30% per trade",
        starting_capital=200.0,
        scans_per_day=48,
        opps_per_scan=8,
        fill_rate=0.80,
        avg_edge=0.045,
        max_positions=15,
        hold_hours=8,
        hold_hours_std=3,
        capital_allocation=0.30,
    ),
}


def run_projection(scenario: Scenario, days: int = 30) -> dict[str, Any]:
    """Run hourly simulation for a scenario."""
    capital = scenario.starting_capital
    total_trades = 0
    total_profit = 0.0
    daily_results = []
    
    # Track open positions: list of dicts with entry_hour, shares, cost, exit_hour
    positions = []
    
    total_hours = days * 24
    scan_interval = 24 / scenario.scans_per_day
    next_scan_hour = 0.0
    
    for hour in range(total_hours):
        # Process exits first (positions that hit their exit time)
        exited_this_hour = []
        day_profit = 0.0
        
        for i, pos in enumerate(positions):
            if pos["exit_hour"] <= hour:
                # Close position
                profit = pos["shares"] * scenario.avg_edge * (1 - scenario.fee_rate)
                capital += pos["cost"] + profit
                total_profit += profit
                day_profit += profit
                exited_this_hour.append(i)
        
        # Remove exited (reverse order)
        for i in reversed(exited_this_hour):
            positions.pop(i)
        
        # Check for new scan
        if hour >= next_scan_hour:
            next_scan_hour += scan_interval
            
            # Try to enter positions
            for _ in range(scenario.opps_per_scan):
                if len(positions) >= scenario.max_positions:
                    break
                
                # Fill rate check
                if random.random() > scenario.fill_rate:
                    continue
                
                # Calculate size based on current capital
                max_trade = capital * scenario.capital_allocation
                shares = min(scenario.avg_shares, int(max_trade / 0.85))
                if shares < 50:
                    continue  # Too small
                
                cost = shares * 0.85  # Average ~$0.85/share
                if cost > capital * 0.95:
                    continue  # Not enough liquid capital
                
                # Hold time with variance
                hold = max(4, int(random.gauss(scenario.hold_hours, scenario.hold_hours_std)))
                
                positions.append({
                    "shares": shares,
                    "cost": cost,
                    "exit_hour": hour + hold,
                })
                capital -= cost
                total_trades += 1
        
        # Record daily progress at day boundaries
        if (hour + 1) % 24 == 0:
            day = (hour + 1) // 24
            deployed = sum(p["cost"] for p in positions)
            total_value = capital + deployed + total_profit
            
            daily_results.append({
                "day": day,
                "available": round(capital, 2),
                "deployed": round(deployed, 2),
                "total_value": round(total_value, 2),
                "trades_cum": total_trades,
                "day_profit": round(day_profit, 2),
                "cumulative_profit": round(total_profit, 2),
            })
    
    # Final value including unrealized PnL
    deployed = sum(p["cost"] for p in positions)
    unrealized = sum(p["shares"] * scenario.avg_edge * (1 - scenario.fee_rate) for p in positions)
    final_value = capital + deployed + unrealized
    
    roi = ((final_value - scenario.starting_capital) / scenario.starting_capital) * 100
    
    return {
        "scenario_name": scenario.name,
        "description": scenario.description,
        "starting": scenario.starting_capital,
        "final": round(final_value, 2),
        "roi": round(roi, 1),
        "multiplier": round(final_value / scenario.starting_capital, 1),
        "trades": total_trades,
        "avg_profit": round(total_profit / total_trades, 2) if total_trades > 0 else 0,
        "daily": daily_results,
    }


def print_comparison(results: list[dict[str, Any]]) -> None:
    """Print comparison table of all scenarios."""
    print(f"\n{'═' * 85}")
    print(f"  POLYMARKET ARBITRAGE: 30-DAY PROFIT SCENARIOS")
    print(f"  Starting Capital: $200.00")
    print(f"{'═' * 85}\n")
    
    print(f"  {'Scenario':<30} {'Final':<10} {'Return':<10} {'Trades':<8} {'Avg $/Trade':<12}")
    print("  " + "-" * 80)
    
    for r in results:
        print(f"  {r['scenario_name']:<30} ${r['final']:<9,.0f} {r['multiplier']:<9}× {r['trades']:<8} ${r['avg_profit']:<11,.2f}")
    
    print("  " + "-" * 80)
    
    # Show realistic vs moon
    realistic = [r for r in results if "REALISTIC" in r['scenario_name']][0]
    moon = [r for r in results if "MOON" in r['scenario_name']][0]
    
    print(f"\n  KEY INSIGHTS:")
    print(f"  • Realistic expectation: {realistic['multiplier']}× ({realistic['roi']:.0f}% return)")
    print(f"  • Unlikely best-case:  {moon['multiplier']}× ({moon['roi']:.0f}% return)")
    print(f"  • Gap: {moon['multiplier']/realistic['multiplier']:.1f}× difference between realistic and moon")
    print(f"  • Most profit comes from compounding, not individual trades")


def print_detailed(result: dict[str, Any]) -> None:
    """Print day-by-day for one scenario."""
    print(f"\n{'═' * 85}")
    print(f"  SCENARIO: {result['scenario_name']}")
    print(f"  {result['description']}")
    print(f"{'═' * 85}\n")
    
    print(f"  Starting: ${result['starting']:,.2f}")
    print(f"  Final:    ${result['final']:,.2f}")
    print(f"  Return:   {result['roi']:.0f}% ({result['multiplier']}× multiplier)")
    print(f"  Trades:   {result['trades']}")
    print(f"  Avg Profit/Trade: ${result['avg_profit']:.2f}\n")
    
    print(f"  {'Day':<5} {'Avail':<10} {'Deployed':<10} {'Value':<10} {'Daily':<10} {'Cum P&L':<10}")
    print("  " + "-" * 60)
    
    for d in result["daily"]:
        if d["day"] <= 7 or d["day"] % 5 == 0 or d["day"] == 30:
            print(f"  {d['day']:<5} ${d['available']:<9,.0f} ${d['deployed']:<9,.0f} ${d['total_value']:<9,.0f} "
                  f"${d['day_profit']:<9.2f} ${d['cumulative_profit']:<9.2f}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Arb Projection")
    parser.add_argument("--scenario", choices=["conservative", "realistic", "optimistic", "moon", "full_reinvest"], 
                       default="realistic")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--capital", type=float, default=200.0)
    parser.add_argument("--all", action="store_true", help="Compare all scenarios")
    
    args = parser.parse_args()
    
    if args.all:
        results = []
        for name in ["conservative", "realistic", "optimistic", "moon", "full_reinvest"]:
            scenario = SCENARIOS[name]
            scenario.starting_capital = args.capital
            result = run_projection(scenario, args.days)
            results.append(result)
        print_comparison(results)
        
        # Also print detailed for full_reinvest
        print("\n\n")
        fr = [r for r in results if "FULL REINVEST" in r['scenario_name']][0]
        print_detailed(fr)
    else:
        scenario = SCENARIOS[args.scenario]
        scenario.starting_capital = args.capital
        result = run_projection(scenario, args.days)
        print_detailed(result)


if __name__ == "__main__":
    main()
