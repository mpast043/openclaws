#!/usr/bin/env python3
"""
Simulation test for realistic paper trading.
Demonstrates latency, slippage, and failure modes.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Mock the bot components for simulation
@dataclass
class BookLevel:
    price: float
    size: float

@dataclass  
class BookSnapshot:
    token_id: str
    bids: list
    asks: list
    asof_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def best_bid(self):
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self):
        return self.asks[0] if self.asks else None

@dataclass
class PairCandidate:
    pair_label: str
    leg_a: Any
    leg_b: Any

@dataclass
class Opportunity:
    candidate: PairCandidate
    book_a: BookSnapshot
    book_b: BookSnapshot
    target_shares: float
    gross_cost_total: float
    fee_total_a: float = 0
    fee_total_b: float = 0
    edge_total: float = 0
    fill_a: Any = None
    fill_b: Any = None

class MockBotConfig:
    """Config with paper realism settings."""
    min_shares = 5
    hedge_price_buffer = 0.002
    
    # Paper realism
    paper_simulate_latency = True
    paper_latency_ms_min = 50
    paper_latency_ms_max = 200
    paper_simulate_slippage = True
    paper_slippage_fail_threshold = 0.0010
    paper_leg_b_failure_rate = 0.15
    paper_hedge_enabled = True
    paper_hedge_success_rate = 0.85

class MockBookReader:
    """Simulates book movements realistically."""
    def __init__(self):
        self.movement_simulator = random.Random()
    
    def get_book_snapshot(self, token_id: str) -> BookSnapshot:
        # Realistic price movements
        r = self.movement_simulator.random()
        if r < 0.10:
            # Minimal noise (< 5 bps) - 80% of cases should pass threshold
            slip_a = self.movement_simulator.uniform(0, 0.0004) 
            slip_b = self.movement_simulator.uniform(0, 0.0004)
        elif r < 0.25:
            # Moderate (5-9 bps) - still within threshold
            slip_a = self.movement_simulator.uniform(0.0005, 0.0009)
            slip_b = self.movement_simulator.uniform(0.0005, 0.0009)
        else:
            # Near threshold or just over (10+ bps) - 15% fail rate
            slip_a = self.movement_simulator.uniform(0.0008, 0.0020)
            slip_b = self.movement_simulator.uniform(0.0008, 0.0020)
            
        # Return token-specific snapshots
        if 'a' in token_id:
            asks_a = [BookLevel(price=0.5200 + slip_a, size=100)]
            bids_a = [BookLevel(price=0.5180 + slip_a * 0.8, size=100)]
            return BookSnapshot(token_id=token_id, bids=bids_a, asks=asks_a)
        else:
            asks_b = [BookLevel(price=0.4700 + slip_b, size=100)]
            bids_b = [BookLevel(price=0.4680 + slip_b * 0.8, size=100)]
            return BookSnapshot(token_id=token_id, bids=bids_b, asks=asks_b)

class RealisticPaperExecutor:
    """Simplified version showing the simulation logic."""
    
    def __init__(self, cfg, book_reader=None):
        self.cfg = cfg
        self.books = book_reader
        self._rng = random.Random()
        self.results = []
    
    def _simulate_latency(self):
        delay_ms = self._rng.uniform(
            self.cfg.paper_latency_ms_min, 
            self.cfg.paper_latency_ms_max
        )
        print(f"    ⏱  Simulating latency: {delay_ms:.0f}ms")
    
    def _check_slippage(self, opp: Opportunity):
        if not self.cfg.paper_simulate_slippage or not self.books:
            return True, {}
        
        fresh_a = self.books.get_book_snapshot(opp.candidate.leg_a)
        fresh_b = self.books.get_book_snapshot(opp.candidate.leg_b)
        
        slip_a = slip_b = 0
        if fresh_a.best_ask and opp.book_a.best_ask:
            slip_a = fresh_a.best_ask.price - opp.book_a.best_ask.price
        if fresh_b.best_ask and opp.book_b.best_ask:
            slip_b = fresh_b.best_ask.price - opp.book_b.best_ask.price
        
        print(f"    📊 Slippage check: A={slip_a:.4f}, B={slip_b:.4f} (threshold={self.cfg.paper_slippage_fail_threshold})")
        
        if slip_a > self.cfg.paper_slippage_fail_threshold:
            return False, {"leg": "A", "slip": slip_a}
        if slip_b > self.cfg.paper_slippage_fail_threshold:
            return False, {"leg": "B", "slip": slip_b}
        
        return True, {}
    
    def _simulate_leg_b_failure(self):
        fails = self._rng.random() < self.cfg.paper_leg_b_failure_rate
        return fails
    
    def _simulate_hedge(self, opp):
        if not self.cfg.paper_hedge_enabled:
            return False, "hedge_disabled"
        
        bid_a = opp.book_a.best_bid.price if opp.book_a.best_bid else 0
        hedge_px = max(0.001, round(bid_a - self.cfg.hedge_price_buffer, 4))
        
        success = self._rng.random() < self.cfg.paper_hedge_success_rate
        if success:
            print(f"    🛡️  Hedge executed at {hedge_px}")
            return True, f"hedge_filled_at_{hedge_px}"
        else:
            print(f"    ⚠️  Hedge failed at {hedge_px}")
            return False, "hedge_no_fill"
    
    def execute(self, opp: Opportunity):
        print(f"\n{'='*60}")
        print(f"📈 Opportunity: {opp.candidate.pair_label}")
        print(f"   Target: {opp.target_shares} shares | Edge: ${opp.edge_total:.2f}")
        
        # Step 1: Latency
        self._simulate_latency()
        
        # Step 2: Slippage check
        ok, info = self._check_slippage(opp)
        if not ok:
            status = "SLIP_FAIL"
            print(f"   ❌ Result: {status} - Price moved {info['slip']:.4f} on leg {info['leg']}")
            self.results.append((status, opp.edge_total, 0))
            return {"ok": False, "status": status}
        
        # Step 3: Inter-leg latency
        self._simulate_latency()
        
        # Step 4: Leg B failure simulation
        if self._simulate_leg_b_failure():
            print(f"    💥 Leg B FOK failed!")
            hedge_ok, note = self._simulate_hedge(opp)
            status = "HEDGED_FAILSAFE" if hedge_ok else "UNHEDGED_ALERT"
            
            # Calculate P&L
            if hedge_ok:
                hedge_cost = opp.target_shares * (opp.book_a.best_bid.price - 0.002)
                gross_cost = opp.target_shares * opp.book_a.best_ask.price
                fees = gross_cost * 0.0002  # ~2bps fee estimate
                pnl = opp.target_shares - gross_cost - hedged_cost - fees
                pnl = -abs(pnl) - fees  # Loss from hedge
            else:
                pnl = -opp.target_shares * opp.book_a.best_ask.price  # Full exposure
            
            print(f"   ❌ Result: {status} | PnL: ${pnl:.2f}")
            self.results.append((status, opp.edge_total, pnl))
            return {"ok": False, "status": status, "hedge_ok": hedge_ok}
        
        # Success!
        status = "FILLED"
        realized_pnl = opp.edge_total  # Minus any slippage we absorbed
        print(f"   ✅ Result: {status} | PnL: ${realized_pnl:.2f}")
        self.results.append((status, opp.edge_total, realized_pnl))
        return {"ok": True, "status": status, "pnl": realized_pnl}

def run_simulation(n_trades: int = 20):
    """Run N simulated paper trades and show stats."""
    
    print("="*60)
    print("🧪 PAPER TRADING REALISM SIMULATION")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Latency: 50-200ms")
    print(f"  Slippage threshold: 10 bps")
    print(f"  Leg B failure rate: 15%")
    print(f"  Hedge success rate: 85%")
    print()
    
    cfg = MockBotConfig()
    books = MockBookReader()
    executor = RealisticPaperExecutor(cfg, book_reader=books)
    
    # Create synthetic opportunities
    for i in range(n_trades):
        opp = Opportunity(
            candidate=PairCandidate(
                pair_label=f"NBA Game {i+1} [YES]",
                leg_a="token_a",
                leg_b="token_b"
            ),
            book_a=BookSnapshot(
                token_id="token_a",
                asks=[BookLevel(0.5200, 100)],  # Match base price
                bids=[BookLevel(0.5180, 100)]
            ),
            book_b=BookSnapshot(
                token_id="token_b", 
                asks=[BookLevel(0.4700, 100)],  # Match base price
                bids=[BookLevel(0.4680, 100)]
            ),
            target_shares=random.randint(100, 500),
            gross_cost_total=random.uniform(450, 550),
            edge_total=random.uniform(2, 15)
        )
        executor.execute(opp)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SIMULATION SUMMARY")
    print("="*60)
    
    results = executor.results
    by_status = {}
    total_expected = 0
    total_realized = 0
    
    for status, expected, realized in results:
        by_status[status] = by_status.get(status, 0) + 1
        total_expected += expected
        total_realized += realized
    
    print(f"\nTotal trades: {len(results)}")
    print(f"\nBy status:")
    for status, count in sorted(by_status.items()):
        pct = count / len(results) * 100
        print(f"  {status:20s} : {count:3d} ({pct:5.1f}%)")
    
    print(f"\nPnL Comparison:")
    print(f"  Expected (naive):  ${total_expected:8.2f}")
    print(f"  Realized (sim):  ${total_realized:8.2f}")
    print(f"  Difference:      ${total_expected - total_realized:8.2f}")
    print(f"  Realism ratio:   {total_realized/max(total_expected, 0.01)*100:5.1f}%")
    
    print(f"\n🏁 This shows why naive paper PnL overestimates actual performance!")

if __name__ == "__main__":
    run_simulation(20)
