#!/usr/bin/env python3
"""
Demo: Multiple substrates with different resource geometry.

Shows capacity governance across:
1. Host substrate - continuous (CPU, memory)
2. DB substrate - discrete (connection pool, query patterns)

Each has different N_geo and gate sensitivities.
"""

import sys
import time
from datetime import timedelta

sys.path.insert(0, '/tmp/openclaws/Repos/capacity-platform')

from capacity_kernel import CapacityKernel, CapacityVector, SubstrateMonitor
from capacity_kernel.db_substrate import DatabaseSubstrate, DBSubstrateState


def demo_host_substrate():
    """Host substrate - continuous geometry."""
    print("=" * 60)
    print("SUBSTRATE 1: Host (Continuous Geometry)")
    print("=" * 60)
    print("N_geo = N^3 continuous field (262,144 modes)")
    print("Resources: CPU%, Memory%, I/O bandwidth")
    print()
    
    monitor = SubstrateMonitor(poll_interval=1.0)
    monitor.start()
    time.sleep(3)  # Collect samples
    
    state = monitor.current_state
    if state:
        print(f"📊 Current State:")
        print(f"   CPU: {state.resources.cpu_percent:.1f}%")
        print(f"   Memory: {state.resources.memory_percent:.1f}%")
        print(f"   n_geo: {state.n_geo:,} (continuous)")
        print()
        print(f"🔍 Gates:")
        print(f"   fit_error: {state.fit_error:.4f}")
        print(f"   gluing: {state.gluing_delta:.4f}")
        print(f"   λ₁: {state.lambda1:.4f}")
        print(f"   λ_int: {state.lambda_int:.2f}")
    
    kernel = CapacityKernel(use_real_substrate=True, monitor=monitor)
    token = kernel.request_capacity(
        "compute-job",
        CapacityVector(C_geo=0.5, C_int=0.4),
        duration=timedelta(seconds=30)
    )
    
    print(f"\n🎫 Admission:")
    if token:
        print(f"   ✅ ADMITTED (token: {token.token_id})")
    else:
        print(f"   ❌ REJECTED")
    
    monitor.stop()
    return kernel if token else None


def demo_db_substrate():
    """Database substrate - discrete geometry."""
    print("\n" + "=" * 60)
    print("SUBSTRATE 2: Database (Discrete Geometry)")
    print("=" * 60)
    print("N_geo = pool_size^2 discrete lattice (400 points)")
    print("Resources: Connections, Query queue, Lock contention")
    print()
    
    db = DatabaseSubstrate(pool_size=20, substrate_id="postgres-main")
    db.start()
    
    # Simulate some query load
    print("Simulating query load...")
    for i in range(30):
        # Mix of fast and slow queries
        duration = 5.0 + (i % 10) * 10  # Pattern: 5, 15, 25... ms
        rows = 1 + (i % 5) * 10
        acquire_lock = (i % 7 == 0)  # Some queries need locks
        db.execute_query(duration, rows, acquire_lock)
    
    time.sleep(2)
    
    state = db.measure_now()
    print(f"\n📊 Current State:")
    print(f"   Pool: {state.pool_metrics.active_connections}/{state.pool_metrics.pool_size} active")
    print(f"   Waiting: {state.pool_metrics.waiting_requests} queued")
    print(f"   Utilization: {state.pool_metrics.utilization:.1%}")
    print(f"   n_geo: {state.n_geo} (discrete lattice)")
    print()
    print(f"🔍 Gates:")
    print(f"   fit_error: {state.fit_error:.4f}")
    print(f"   gluing: {state.gluing_delta:.4f}")
    print(f"   λ₁: {state.lambda1:.4f}")
    print(f"   λ_int: {state.lambda_int:.2f}")
    
    db.stop()
    return state


def demo_compare_geometries():
    """Compare the two substrate geometries."""
    print("\n" + "=" * 60)
    print("GEOMETRY COMPARISON")
    print("=" * 60)
    
    monitor = SubstrateMonitor(poll_interval=1.0)
    monitor.start()
    time.sleep(2)
    
    host_state = monitor.current_state
    monitor.stop()
    
    db = DatabaseSubstrate(pool_size=20)
    # Load DB with queries
    for i in range(20):
        db.execute_query(duration_ms=10 + i*2, rows=i*5)
    db_state = db.measure_now()
    
    print("\n| Metric | Host (Continuous) | DB (Discrete) |")
    print("|--------|-------------------|---------------|")
    print(f"| N_geo | {host_state.n_geo:,} | {db_state.n_geo} |")
    print(f"| λ₁ | {host_state.lambda1:.4f} | {db_state.lambda1:.4f} |")
    print(f"| λ_int | {host_state.lambda_int:.2f} | {db_state.lambda_int:.2f} |")
    print(f"| fit_error | {host_state.fit_error:.4f} | {db_state.fit_error:.4f} |")
    print(f"| gluing | {host_state.gluing_delta:.4f} | {db_state.gluing_delta:.4f} |")
    print(f"| isolation | {host_state.isolation_metric:.4f} | {db_state.isolation_metric:.4f} |")
    
    print("\n📝 Key Differences:")
    print("   • Host: N_geo is huge (262k), gates sensitive to continuous load patterns")
    print("   • DB: N_geo is small (400), gates measure discrete occupancy + query clustering")
    print("   • Host gluing = CPU/memory coherence")
    print("   • DB gluing = query duration/lock correlation")


def demo_multi_kernel():
    """Run multiple kernels with different substrates."""
    print("\n" + "=" * 60)
    print("MULTI-SUBSTRATE CAPACITY GOVERNANCE")
    print("=" * 60)
    
    # Kernel 1: Host resources
    host_monitor = SubstrateMonitor(poll_interval=1.0)
    host_monitor.start()
    time.sleep(2)
    host_kernel = CapacityKernel(use_real_substrate=True, monitor=host_monitor)
    
    # Kernel 2: DB resources (simulated)
    db_substrate = DatabaseSubstrate(pool_size=20)
    # Pre-load with queries
    for i in range(15):
        db_substrate.execute_query(duration_ms=20, rows=50)
    
    print("\n🎫 Requesting capacity across both substrates...")
    print()
    
    # Request 1: Low geometric capacity (fits both)
    print("Request 1: C_geo=0.3, C_int=0.3 (low capacity)")
    host_token = host_kernel.request_capacity("low-job", CapacityVector(C_geo=0.3, C_int=0.3))
    
    # Simulate DB admission check (manual for demo)
    db_state = db_substrate.measure_now()
    db_can_admit = (db_state.pool_metrics.utilization < 0.3 and 
                   db_state.gluing_delta < 0.004)
    
    print(f"   Host: {'✅ ADMITTED' if host_token else '❌ REJECTED'}")
    print(f"   DB:   {'✅ ADMITTED' if db_can_admit else '❌ REJECTED'} (util={db_state.pool_metrics.utilization:.1%})")
    
    if host_token and db_can_admit:
        print(f"   → 🎉 Can run cross-substrate workload")
    else:
        print(f"   → Cannot satisfy both substrates")
    
    print()
    
    # Request 2: High capacity
    print("Request 2: C_geo=0.8, C_int=0.8 (high capacity)")
    host_token2 = host_kernel.request_capacity("high-job", CapacityVector(C_geo=0.8, C_int=0.8))
    
    db_state2 = db_substrate.measure_now()
    db_can_admit2 = (db_state2.pool_metrics.utilization < 0.8 and
                    db_state2.gluing_delta < 0.004)
    
    print(f"   Host: {'✅ ADMITTED' if host_token2 else '❌ REJECTED'}")
    print(f"   DB:   {'✅ ADMITTED' if db_can_admit2 else '❌ REJECTED'} (util={db_state2.pool_metrics.utilization:.1%})")
    
    host_monitor.stop()
    
    print("\n✅ Multi-substrate capacity governance working")


if __name__ == "__main__":
    print("🚀 Multi-Substrate Capacity Platform Demo")
    print("   Showing different resource geometries")
    
    try:
        import psutil
        print("   psutil: YES")
    except:
        print("   psutil: NO (install for host measurements)")
    
    print()
    
    demo_host_substrate()
    demo_db_substrate()
    demo_compare_geometries()
    demo_multi_kernel()
    
    print("\n" + "=" * 60)
    print("✅ Different substrate geometries demonstrated")
    print("=" * 60)
