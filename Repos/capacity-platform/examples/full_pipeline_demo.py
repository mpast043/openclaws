#!/usr/bin/env python3
"""
Full Pipeline Demo: All Three Steps

1. Step 1: Measure substrate (real or simulated)
2. Step 2: Evaluate gates → Admit/Decline  
3. Step 3: Track outcomes → Validate/Falsify

Shows complete capacity-governed workflow end-to-end.
"""

import sys
import time
import random

sys.path.insert(0, '/tmp/openclaws/Repos/capacity-platform')

from capacity_kernel import (
    CapacityKernel,
    CapacityVector,
    SubstrateMonitor,
    create_truth_infrastructure,
    PredictionOutcome,
)


def simulate_workload_outcome(c_geo: float, c_int: float) -> dict:
    """Simulate actual workload outcome."""
    load = c_geo * 0.8 + random.uniform(-0.1, 0.1)
    stress = c_geo * c_int
    
    # Higher stress → worse outcomes
    stability = max(0.0, 1.0 - stress * 1.5)
    latency = 50 + stress * 200 + max(0, load - 0.8) * 500
    error_rate = max(0.0, (stress - 0.5) * 0.1)
    
    return {
        'load': load,
        'stability': stability,
        'latency_ms': latency,
        'error_rate': error_rate
    }


def demo_full_pipeline():
    print("=" * 70)
    print("FULL PIPELINE: STEP 1 → STEP 2 → STEP 3")
    print("=" * 70)
    
    # Setup
    print("\n🔧 Initializing...")
    kernel = CapacityKernel(use_real_substrate=False)  # Simulated substrate
    ledger, step3_gates, tracker = create_truth_infrastructure()
    
    # Run simulated admission cycle
    print("\n" + "=" * 70)
    print("ADMISSION CYCLE (20 Requests)")
    print("=" * 70)
    
    admitted = 0
    declined = 0
    
    for i in range(20):
        # Random capacity request
        c_geo = 0.3 + (i * 0.03)  # Gradually increasing
        c_int = 0.2 + random.uniform(0, 0.6)
        
        print(f"\n📨 Request {i+1}: C_geo={c_geo:.2f}, C_int={c_int:.2f}")
        
        # STEP 2: Request capacity (Step 2 gates evaluated internally)
        capacity = CapacityVector(C_geo=c_geo, C_int=c_int)
        token = kernel.request_capacity(
            workload_id=f"req-{i+1}",
            C=capacity
        )
        
        # Check if admission was granted
        is_admitted = kernel.check_capacity(token)
        
        if is_admitted:
            admitted += 1
            print(f"   ✅ ADMITTED (token: {token.token_id[:8]}...)")
            
            # Get current gate states from kernel (stored in _last_gate_state)
            gate_state = kernel._last_gate_state
            
            if gate_state:
                print(f"   Gates: fit={gate_state.fit.value:.4f}, "
                      f"gluing={gate_state.gluing.value:.4f}, "
                      f"λ₁={gate_state.uv.value:.4f}")
                
                # Record prediction for Step 3
                pred_id = ledger.record_prediction(
                    substrate_id="host",
                    c_geo=c_geo,
                    c_int=c_int,
                    gate_metrics={
                        'fit_error': gate_state.fit.value,
                        'gluing_delta': gate_state.gluing.value,
                        'lambda1': gate_state.uv.value,
                        'lambda_int': 1.0 - gate_state.gluing.value,
                        'isolation_metric': gate_state.isolation.value
                    },
                    outcome_window_seconds=10.0
                )
            else:
                # Record without gate metrics if unavailable
                pred_id = ledger.record_prediction(
                    substrate_id="host",
                    c_geo=c_geo,
                    c_int=c_int,
                    gate_metrics={
                        'fit_error': 0.02,
                        'gluing_delta': 0.001,
                        'lambda1': 0.01,
                        'lambda_int': 0.5,
                        'isolation_metric': 0.05
                    },
                    outcome_window_seconds=10.0
                )
            
            # Simulate workload execution outcome
            time.sleep(0.1)  # Brief delay
            outcome = simulate_workload_outcome(c_geo, c_int)
            
            # Validate prediction
            if outcome['latency_ms'] > 100 or outcome['error_rate'] > 0.01:
                ledger.validate_outcome(
                    pred_id,
                    PredictionOutcome.FALSIFIED,
                    f"Latency: {outcome['latency_ms']:.0f}ms, "
                    f"Errors: {outcome['error_rate']:.2%}"
                )
                print(f"   ❌ OUTCOME: Falsified! "
                      f"(latency={outcome['latency_ms']:.0f}ms)")
            else:
                ledger.validate_outcome(
                    pred_id,
                    PredictionOutcome.VALIDATED,
                    f"Stable: {outcome['stability']:.1%}"
                )
                print(f"   ✅ OUTCOME: Validated "
                      f"(stability={outcome['stability']:.1%})")
        else:
            declined += 1
            print(f"   ❌ DECLINED")
            
            # Simulate: was this a missed opportunity?
            outcome = simulate_workload_outcome(c_geo, c_int)
            if outcome['stability'] > 0.8:
                print(f"   ⚠️  Missed stable opportunity! "
                      f"(would have been: {outcome['stability']:.1%})")
    
    # Summary
    print("\n" + "=" * 70)
    print("ADMISSION SUMMARY")
    print("=" * 70)
    print(f"   Total requests: 20")
    print(f"   ✅ Admitted: {admitted}")
    print(f"   ❌ Declined: {declined}")
    print(f"   Admission rate: {admitted/20*100:.1f}%")
    
    # Step 3 Evaluation
    print("\n" + "=" * 70)
    print("STEP 3 VALIDATION")
    print("=" * 70)
    
    results = step3_gates.evaluate_all()
    
    print(f"\n   Precision: {results['precision']['value']:.1%} "
          f"({'✅' if results['precision']['pass'] else '❌'})")
    print(f"   Recall: {results['recall']['value']:.1%} "
          f"({'✅' if results['recall']['pass'] else '❌'})")
    print(f"   Calibration: {results['calibration']['value']:.1%} "
          f"({'✅' if results['calibration']['pass'] else '❌'})")
    
    print(f"\n   Falsification rate: {ledger.get_falsification_rate():.1%}")
    
    # Falsification analysis
    print("\n" + "=" * 70)
    print("FALSIFICATION ANALYSIS")
    print("=" * 70)
    
    falsified = ledger.get_falsified_predictions()
    if falsified:
        print(f"\n   {len(falsified)} falsified prediction(s):")
        
        # Analyze patterns
        high_capacity = [f for f in falsified if f.c_geo > 0.7]
        high_isolation = [f for f in falsified if f.isolation > 0.1]
        
        print(f"   - High C_geo (>0.7): {len(high_capacity)}")
        print(f"   - High isolation (>0.1): {len(high_isolation)}")
        
        for f in falsified[:3]:  # Show first 3
            print(f"\n   • C_geo={f.c_geo:.2f}: {f.falsification_reason}")
    else:
        print("\n   No falsifications")
    
    print("\n" + "=" * 70)
    print("Pipeline complete: Substrate → Gates → Admission → Truth → Learn")
    print("=" * 70)


if __name__ == "__main__":
    # Set seed for reproducible demo
    random.seed(42)
    demo_full_pipeline()
