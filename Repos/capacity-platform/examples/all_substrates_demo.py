#!/usr/bin/env python3
"""
Demo: All three substrate geometries.

1. Host - continuous field (262,144 modes)
2. Database - discrete 2D lattice (400 points)  
3. GPU Tensor Core - discrete 3D lattice (10,240 points)

Same framework, different geometries.
"""

import sys
import time

sys.path.insert(0, '/tmp/openclaws/Repos/capacity-platform')

from capacity_kernel import SubstrateMonitor
from capacity_kernel.db_substrate import DatabaseSubstrate
from capacity_kernel.gpu_substrate import TensorCoreSubstrate


def demo_all():
    print("=" * 70)
    print("THREE SUBSTRATE GEOMETRIES")
    print("=" * 70)
    
    # 1. Host Substrate
    print("\n1️⃣  HOST SUBSTRATE (Continuous)")
    print("-" * 40)
    monitor = SubstrateMonitor(poll_interval=1.0)
    monitor.start()
    time.sleep(2)
    
    host_state = monitor.current_state
    if host_state:
        print(f"N_geo: {host_state.n_geo:,} (N³ continuous)")
        print(f"CPU: {host_state.resources.cpu_percent:.1f}%")
        print(f"λ₁: {host_state.lambda1:.4f}, fit: {host_state.fit_error:.4f}")
    monitor.stop()
    
    # 2. Database Substrate
    print("\n2️⃣  DATABASE SUBSTRATE (Discrete 2D)")
    print("-" * 40)
    db = DatabaseSubstrate(pool_size=20)
    for i in range(30):
        db.execute_query(duration_ms=10+i*2, rows=i*5, 
                       acquire_lock=(i%7==0))
    db_state = db.measure_now()
    
    print(f"N_geo: {db_state.n_geo} (pool² × discrete)")
    print(f"Pool: {db_state.pool_metrics.active_connections}/20")
    print(f"λ₁: {db_state.lambda1:.4f}, fit: {db_state.fit_error:.4f}")
    
    # 3. GPU Tensor Core Substrate
    print("\n3️⃣  GPU TENSOR CORE (Discrete 3D)")
    print("-" * 40)
    gpu = TensorCoreSubstrate(sm_count=80, tensor_cores_per_sm=4)
    for i in range(100):
        grid = (2, 2, 1)
        block = (32, 8, 1)
        gpu.launch_kernel(grid, block, is_tensor_op=(i<80), 
                        sm_occupancy=0.7+(i%4)*0.05)
    gpu_state = gpu.measure_now()
    
    print(f"N_geo: {gpu_state.n_geo:,} (SMs × TC × warps)")
    print(f"SMs: {gpu_state.gpu_metrics.active_kernels}/80")
    print(f"Tensor util: {gpu_state.gpu_metrics.tensor_core_util:.0%}")
    print(f"λ₁: {gpu_state.lambda1:.4f}, fit: {gpu_state.fit_error:.4f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("GEOMETRY COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Substrate':<20} {'Geometry':<15} {'N_geo':<12} {'Mode'}")
    print("-" * 60)
    print(f"{'Host':<20} {'Continuous':<15} {host_state.n_geo:<12,} {'Field'}")
    print(f"{'Database':<20} {'Discrete 2D':<15} {db_state.n_geo:<12} {'Lattice'}")
    print(f"{'GPU Tensor Core':<20} {'Discrete 3D':<15} {gpu_state.n_geo:<12,} {'Lattice'}")
    print()
    print("Same gates, same physics analogs, different substrate structures.")


if __name__ == "__main__":
    demo_all()
