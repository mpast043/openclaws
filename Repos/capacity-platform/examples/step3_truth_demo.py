#!/usr/bin/env python3
"""
Demo: Step 3 Truth Infrastructure

Shows falsification gates validating Step 2 predictions against reality.
"""

import sys
import time

sys.path.insert(0, '/tmp/openclaws/Repos/capacity-platform')

from capacity_kernel import (
    TruthLedger,
    Step3Gates,
    OutcomeTracker,
    PredictionOutcome,
    create_truth_infrastructure,
)


def demo_truth_infrastructure():
    print("=" * 70)
    print("STEP 3 TRUTH INFRASTRUCTURE DEMO")
    print("=" * 70)
    
    # Create infrastructure
    ledger, gates, tracker = create_truth_infrastructure(check_interval=1.0)
    
    print("\n1️⃣  RECORDING PREDICTIONS")
    print("-" * 50)
    
    # Simulate capacity admissions
    predictions = []
    for i in range(10):
        pred_id = ledger.record_prediction(
            substrate_id="host",
            c_geo=0.5 + i * 0.05,
            c_int=0.3 + i * 0.03,
            gate_metrics={
                'fit_error': 0.02 + i * 0.001,
                'gluing_delta': 0.001,
                'lambda1': 0.05 + i * 0.01,
                'lambda_int': 0.5 + i * 0.05,
                'isolation_metric': 0.05
            },
            outcome_window_seconds=5.0
        )
        predictions.append(pred_id)
        print(f"  Prediction {i+1}: {pred_id[:8]}... C_geo={0.5+i*0.05:.2f}")
    
    print(f"\nRecorded {len(predictions)} predictions")
    print(f"Pending: {len(ledger.get_pending_predictions())}")
    
    print("\n2️⃣  SIMULATING OUTCOMES")
    print("-" * 50)
    
    # Validate some as successful, some as falsified
    import random
    random.seed(42)
    
    for i, pred_id in enumerate(predictions):
        # Simulate outcome
        if i < 7:  # 70% success rate
            ledger.validate_outcome(
                pred_id,
                PredictionOutcome.VALIDATED,
                "Stable execution completed"
            )
            outcome_str = "✅ VALIDATED"
        elif i == 7:
            ledger.validate_outcome(
                pred_id,
                PredictionOutcome.FALSIFIED,
                "Latency exceeded threshold (>100ms)"
            )
            outcome_str = "❌ FALSIFIED"
        elif i == 8:
            ledger.validate_outcome(
                pred_id,
                PredictionOutcome.MISSED,
                "Opportunity declined, should have admitted"
            )
            outcome_str = "⛔ MISSED"
        else:
            ledger.validate_outcome(
                pred_id,
                PredictionOutcome.TIMEOUT,
                "Observation window expired"
            )
            outcome_str = "⏱️  TIMEOUT"
        
        print(f"  {pred_id[:8]}... {outcome_str}")
    
    print("\n3️⃣  STEP 3 GATE EVALUATION")
    print("-" * 50)
    
    results = gates.evaluate_all()
    
    print(f"\n📊 Precision Gate:")
    print(f"   Result: {'✅ PASS' if results['precision']['pass'] else '❌ FAIL'}")
    print(f"   Value: {results['precision']['value']:.2%}")
    print(f"   Threshold: {results['precision']['threshold']:.0%}")
    print(f"   {results['precision'].get('numerator', 0)}/{results['precision'].get('denominator', 0)} correct")
    
    print(f"\n📊 Recall Gate:")
    print(f"   Result: {'✅ PASS' if results['recall']['pass'] else '❌ FAIL'}")
    print(f"   Value: {results['recall']['value']:.2%}")
    print(f"   Threshold: {results['recall']['threshold']:.0%}")
    print(f"   Validated: {results['recall'].get('validated', 0)}, Missed: {results['recall'].get('missed', 0)}")
    
    print(f"\n📊 Calibration Gate:")
    print(f"   Result: {'✅ PASS' if results['calibration']['pass'] else '❌ FAIL'}")
    print(f"   Value: {results['calibration']['value']:.2%}")
    print(f"   Calibration error: {results['calibration'].get('calibration_error', 0):.2%}")
    print(f"   Samples: {results['calibration'].get('n_samples', 0)}")
    
    print("\n4️⃣  STATISTICS")
    print("-" * 50)
    print(f"   Total predictions: {ledger.validation_stats['total_predictions']}")
    print(f"   ✅ Validated: {ledger.validation_stats['validated']}")
    print(f"   ❌ Falsified: {ledger.validation_stats['falsified']}")
    print(f"   ⛔ Missed: {ledger.validation_stats['missed']}")
    print(f"   ⏱️  Timeouts: {ledger.validation_stats['timeouts']}")
    print(f"   Falsification rate: {ledger.get_falsification_rate():.1%}")
    
    print("\n5️⃣  FALSIFICATION ANALYSIS")
    print("-" * 50)
    
    falsified = ledger.get_falsified_predictions()
    if falsified:
        print(f"   Found {len(falsified)} falsified prediction(s):")
        for f in falsified:
            print(f"\n   Prediction: {f.prediction_id[:8]}...")
            print(f"   Capacity: C_geo={f.c_geo:.2f}, C_int={f.c_int:.2f}")
            print(f"   Gates at admission:")
            print(f"     - fit_error: {f.fit_error:.4f}")
            print(f"     - gluing: {f.gluing_delta:.4f}")
            print(f"     - isolation: {f.isolation:.4f}")
            print(f"   Reason: {f.falsification_reason}")
    else:
        print("   No falsified predictions to analyze")
    
    print("\n" + "=" * 70)
    print("Step 3 gates now operational.")
    print("Step 2 predicts, Step 3 validates predictions against outcomes.")
    print("=" * 70)


if __name__ == "__main__":
    demo_truth_infrastructure()
