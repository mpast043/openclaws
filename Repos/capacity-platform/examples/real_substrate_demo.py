#!/usr/bin/env python3
"""
Demo: Real substrate measurement integration.

This script demonstrates the P0 deliverable: live substrate measurement
feeding into gate evaluation and capacity admission.
"""

import time
from datetime import timedelta

from capacity_kernel import (
    CapacityKernel,
    CapacityVector,
    SubstrateMonitor,
    create_real_substrate,
)


def demo_one_shot_measurement():
    """Show a single real substrate measurement."""
    print("=" * 60)
    print("DEMO 1: One-Shot Real Substrate Measurement")
    print("=" * 60)
    
    state = create_real_substrate()
    print(f"\n📊 Substrate: {state.substrate_id}")
    print(f"🕐 Timestamp: {state.timestamp:%H:%M:%S}")
    print(f"\n🔧 Resources:")
    print(f"   CPU: {state.resources.cpu_percent:.1f}%")
    print(f"   Memory: {state.resources.memory_percent:.1f}%")
    print(f"   Network: {state.resources.network_io_mbps:.2f} Mbps")
    print(f"   Disk: {state.resources.disk_io_mbps:.2f} Mbps")
    
    print(f"\n📐 Gate Metrics:")
    print(f"   fit_error: {state.fit_error:.4f}")
    print(f"   gluing_delta: {state.gluing_delta:.4f}")
    print(f"   lambda1: {state.lambda1:.4f}")
    print(f"   lambda_int: {state.lambda_int:.2f}")
    print(f"   isolation: {state.isolation_metric:.4f}")


def demo_kernel_with_real_substrate():
    """Show capacity admission using real measurements."""
    print("\n" + "=" * 60)
    print("DEMO 2: Capacity Admission with Real Substrate")
    print("=" * 60)
    
    # Create kernel with real substrate enabled
    kernel = CapacityKernel(use_real_substrate=True)
    
    print("\n🎫 Requesting capacity tokens...")
    
    # Request 1: Low capacity - should pass on most systems
    token1 = kernel.request_capacity(
        workload_id="low-priority-service",
        C=CapacityVector(C_geo=0.3, C_int=0.3),
        duration=timedelta(seconds=30)
    )
    
    if token1:
        print(f"   ✅ ADMITTED (token: {token1.token_id})")
    else:
        print(f"   ❌ REJECTED - gates failed")
    
    # Show gate state
    if kernel.gate_state:
        print(f"\n🔍 Gate State: {kernel.gate_state}")
        for gate in [kernel.gate_state.fit, kernel.gate_state.gluing,
                     kernel.gate_state.uv, kernel.gate_state.isolation]:
            print(f"   {gate}")
    
    # Request 2: Medium capacity
    token2 = kernel.request_capacity(
        workload_id="medium-service",
        C=CapacityVector(C_geo=0.5, C_int=0.5),
        duration=timedelta(seconds=30)
    )
    
    if token2:
        print(f"   ✅ ADMITTED (token: {token2.token_id})")
    else:
        print(f"   ❌ REJECTED")
    
    # Request 3: High capacity - may fail under load
    token3 = kernel.request_capacity(
        workload_id="high-intensity",
        C=CapacityVector(C_geo=0.8, C_int=0.8),
        duration=timedelta(seconds=30)
    )
    
    if token3:
        print(f"   ✅ ADMITTED (token: {token3.token_id})")
    else:
        print(f"   ❌ REJECTED (current load too high)")


def demo_continuous_monitoring():
    """Show continuous substrate monitoring background thread."""
    print("\n" + "=" * 60)
    print("DEMO 3: Continuous Substrate Monitoring")
    print("=" * 60)
    
    monitor = SubstrateMonitor(substrate_id="host", poll_interval=0.5)
    
    # Add callback to print updates
    def on_measure(state):
        print(f"   [{state.timestamp:%H:%M:%S}] CPU: {state.resources.cpu_percent:5.1f}% | "
              f"λ₁: {state.lambda1:.3f} | fit: {state.fit_error:.4f}")
    
    monitor.on_measurement(on_measure)
    monitor.start()
    
    print("\n📈 Monitoring for 3 seconds (updates every 0.5s)...")
    print("   (Run some CPU-intensive work in another terminal)")
    time.sleep(3)
    
    monitor.stop()
    
    # Get final state
    final = monitor.current_state
    if final:
        print(f"\n📊 Final measurements:")
        print(f"   Samples collected: {len(monitor.window.cpu_samples)}")
        print(f"   Dominant frequency (λ₁ analog): {monitor.window.dominant_frequency:.3f}")
        print(f"   Interaction strength (λ_int analog): {monitor.window.interaction_strength:.2f}")


def demo_admission_during_load():
    """Show admission decisions change based on real substrate load."""
    print("\n" + "=" * 60)
    print("DEMO 4: Load-Aware Admission Control")
    print("=" * 60)
    
    # Start continuous monitoring
    monitor = SubstrateMonitor(substrate_id="admission-test", poll_interval=0.5)
    monitor.start()
    
    # Create kernel with the monitor
    kernel = CapacityKernel(use_real_substrate=True, monitor=monitor)
    
    print("\n📊 Current system load:")
    time.sleep(1)  # Let monitor collect samples
    
    state = monitor.current_state
    if state:
        print(f"   CPU: {state.resources.cpu_percent:.1f}%")
        print(f"   Memory: {state.resources.memory_percent:.1f}%")
    
    print("\n🎫 Admission requests (same C_geo=0.6, C_int=0.6):")
    
    # Try admitting multiple workloads
    tokens = []
    for i in range(3):
        token = kernel.request_capacity(
            workload_id=f"workload-{i+1}",
            C=CapacityVector(C_geo=0.6, C_int=0.6),
            duration=timedelta(seconds=60)
        )
        
        if token:
            tokens.append(token)
            print(f"   #{i+1}: ✅ ADMITTED (token: {token.token_id})")
        else:
            print(f"   #{i+1}: ❌ REJECTED - system at capacity")
            # Show why
            if kernel.gate_state:
                failing = kernel.gate_state.failing
                print(f"        Failing gates: {failing}")
    
    monitor.stop()
    
    print(f"\n📋 Summary: {len(tokens)}/3 workloads admitted")


if __name__ == "__main__":
    print("🚀 Capacity Platform - Real Substrate Integration Demo")
    print("   P0 Deliverable: Sample data → Live measurement")
    
    try:
        import psutil
        print("   psutil available: YES (real measurements enabled)")
    except ImportError:
        print("   psutil available: NO (fallback to simulated data)")
        print("   Install: pip install psutil")
    
    print()
    
    demo_one_shot_measurement()
    demo_kernel_with_real_substrate()
    demo_continuous_monitoring()
    demo_admission_during_load()
    
    print("\n" + "=" * 60)
    print("✅ P0 Complete: Real substrate measurement integrated")
    print("=" * 60)
    print("\nNext: P1 - Step 3 Truth Infrastructure")
