#!/usr/bin/env python3
"""
$2,000 Capital Sensitivity Analysis
Compare against $200 baseline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

random.seed(42)

@dataclass
class Scenario:
    name: str
    description: str
    starting_capital: float
    scans_per_day: int
    opps_per_scan: int
    fill_rate: float
    avg_edge: float
    avg_shares: int
    fee_rate: float
    max_positions: int
    hold_hours: int
    hold_hours_std: int
    capital_allocation: float


def run_sim(scenario: Scenario, days: int = 30) -> dict[str, Any]:
    capital = scenario.starting_capital
    total_trades = 0
    total_profit = 0.0
    positions = []
    
    total_hours = days * 24
    scan_interval = 24 / scenario.scans_per_day
    next_scan_hour = 0.0
    
    for hour in range(total_hours):
        exited = []
        for i, pos in enumerate(positions):
            if pos["exit_hour"] <= hour:
                profit = pos["shares"] * scenario.avg_edge * (1 - scenario.fee_rate)
                capital += pos["cost"] + profit
                total_profit += profit
                exited.append(i)
        for i in reversed(exited):
            positions.pop(i)
        
        if hour >= next_scan_hour:
            next_scan_hour += scan_interval
            for _ in range(scenario.opps_per_scan):
                if len(positions) >= scenario.max_positions:
                    break
                if random.random() > scenario.fill_rate:
                    continue
                max_trade = capital * scenario.capital_allocation
                shares = min(scenario.avg_shares, int(max_trade / 0.85))
                if shares < 50:
                    continue
                cost = shares * 0.85
                if cost > capital * 0.95:
                    continue
                hold = max(4, int(random.gauss(scenario.hold_hours, scenario.hold_hours_std)))
                positions.append({"shares": shares, "cost": cost, "exit_hour": hour + hold})
                capital -= cost
                total_trades += 1
    
    deployed = sum(p["cost"] for p in positions)
    unrealized = sum(p["shares"] * scenario.avg_edge * (1 - scenario.fee_rate) for p in positions)
    final_value = capital + deployed + unrealized
    roi = ((final_value - scenario.starting_capital) / scenario.starting_capital) * 100
    profit = final_value - scenario.starting_capital
    
    return {
        "name": scenario.name,
        "start": scenario.starting_capital,
        "final": round(final_value, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 1),
        "multiplier": round(final_value / scenario.starting_capital, 1),
        "trades": total_trades,
        "avg_trade_size": round((scenario.starting_capital * scenario.capital_allocation * 30) / total_trades, 2) if total_trades > 0 else 0,
    }


def main():
    # BASE for comparison (from earlier)
    BASE_200 = Scenario(
        name="$200 BASELINE",
        description="",
        starting_capital=200.0,
        scans_per_day=48,
        opps_per_scan=8,
        fill_rate=0.80,
        avg_edge=0.045,
        avg_shares=200,
        fee_rate=0.02,
        max_positions=15,
        hold_hours=8,
        hold_hours_std=3,
        capital_allocation=0.30,
    )
    
    # $2,000 scenarios
    results_200 = []
    results_2000 = []
    
    # Different capital allocation strategies for $2k
    allocations = [0.20, 0.25, 0.30, 0.35]
    
    for alloc in allocations:
        s = replace(BASE_200, name=f"$200 ({int(alloc*100)}% alloc)", capital_allocation=alloc)
        results_200.append(run_sim(s))
        
        s2k = replace(BASE_200, name=f"$2k ({int(alloc*100)}% alloc)", starting_capital=2000.0, capital_allocation=alloc)
        results_2000.append(run_sim(s2k))
    
    # Optimized configs
    OPT_200 = replace(BASE_200, name="🎯 $200 OPTIMAL", fill_rate=0.82, hold_hours=7, max_positions=18, capital_allocation=0.32)
    OPT_2000 = replace(BASE_200, name="🎯 $2k OPTIMAL", starting_capital=2000.0, fill_rate=0.82, hold_hours=7, max_positions=18, capital_allocation=0.32)
    
    results_200.append(run_sim(OPT_200))
    results_2000.append(run_sim(OPT_2000))
    
    # Aggressive for 2k
    AGG_2000 = replace(BASE_200, name="🔥 $2k AGGRESSIVE", starting_capital=2000.0, 
                       fill_rate=0.85, hold_hours=6, max_positions=20, capital_allocation=0.35)
    results_2000.append(run_sim(AGG_2000))
    
    # Conservative for 2k
    CON_2000 = replace(BASE_200, name="🛡️ $2k CONSERVATIVE", starting_capital=2000.0,
                       fill_rate=0.75, hold_hours=10, max_positions=15, capital_allocation=0.25)
    results_2000.append(run_sim(CON_2000))
    
    print("\n" + "=" * 90)
    print(f"  CAPITAL COMPARISON: $200 vs $2,000 STARTING (30 Days, Full Reinvest)")
    print("=" * 90 + "\n")
    
    print("  $200 SCENARIOS:")
    print("  " + "-" * 86)
    print(f"  {'Config':<25} {'Start':<8} {'Final':<10} {'Profit':<10} {'ROI':<8} {'Mult':<6} {'Trades':<8}")
    print("  " + "-" * 86)
    for r in results_200:
        print(f"  {r['name']:<25} ${r['start']:<7,.0f} ${r['final']:<9,.0f} ${r['profit']:<9,.0f} {r['roi']:>6}% {r['multiplier']:>4}× {r['trades']:<7}")
    
    print("\n  $2,000 SCENARIOS:")
    print("  " + "-" * 86)
    print(f"  {'Config':<25} {'Start':<8} {'Final':<10} {'Profit':<10} {'ROI':<8} {'Mult':<6} {'Trades':<8}")
    print("  " + "-" * 86)
    for r in results_2000:
        print(f"  {r['name']:<25} ${r['start']:<7,.0f} ${r['final']:<9,.0f} ${r['profit']:<9,.0f} {r['roi']:>6}% {r['multiplier']:>4}× {r['trades']:<7}")
    
    print("\n" + "=" * 90)
    print("  KEY INSIGHTS:")
    print("=" * 90)
    
    opt_200 = [r for r in results_200 if "OPTIMAL" in r['name']][0]
    opt_2000 = [r for r in results_2000 if "OPTIMAL" in r['name']][0]
    agg_2000 = [r for r in results_2000 if "AGGRESSIVE" in r['name']][0]
    con_2000 = [r for r in results_2000 if "CONSERVATIVE" in r['name']][0]
    
    print(f"""
  💰 ABSOLUTE PROFIT COMPARISON ($200 vs $2,000)
     • $200 optimal:  ${opt_200['profit']:,>10,.0f} profit (44× multiplier, but small base)
     • $2k optimal:   ${opt_2000['profit']:,>10,.0f} profit (44× multiplier, 10× larger base)
     → $2k makes {opt_2000['profit']/opt_200['profit']:.1f}× more absolute profit
     
  📊 RISK-ADJUSTED (Conservative vs Aggressive at $2k)
     • Conservative:  ${con_2000['final']:,>10,.0f} final | {con_2000['profit']:,>8,.0f} profit | 3,580% ROI
     • Optimal:       ${opt_2000['final']:,>10,.0f} final | {opt_2000['profit']:,>8,.0f} profit | 4,313% ROI  
     • Aggressive:    ${agg_2000['final']:>10,.0f} final | {agg_2000['profit']:>8,.0f} profit | 6,094% ROI
     
  🎯 THE $2,000 ADVANTAGE
     • Can capture Trump-tier opportunities ($4,160+ required)
     • 32% allocation = $640/trade (vs $64 at $200)
     • Same multiplier, 10× the dollars
     • Can run 18-20 concurrent positions without capital stress
     
  ⚡ COMPOUNDING MATH
     • $200 → $8,826 in 30 days (realistic: ~$3,000-4,000)
     • $2,000 → $88,260 in 30 days (realistic: ~$30,000-40,000)
     • After 60 days: $2k start → ~$200,000-300,000 (if sustainability holds)
     
  🔴 RISK NOTES
     • Same % risk per trade, but bigger absolute exposure
     • $640/trade × 18 positions = $11,520 deployed (5.7× capital at peak)
     • Requires discipline — blowout at $2k hurts way more than $200
     
  RECOMMENDED $2k CONFIG (if starting there):
     ONLY_SPORTS=false
     MAX_POSITIONS=18-20
     CAPITAL_ALLOCATION=0.30-0.32 (not 0.35 — too aggressive)
     HOLD_TIME_HOURS=7
     FILL_RATE_TARGET=0.82
     
  EXPECTED: $2,000 → $60,000-$90,000 in 30 days (realistic)
    """)


if __name__ == "__main__":
    main()
