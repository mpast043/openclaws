#!/usr/bin/env python3
"""Demo: Capacity-Governed Platform

Shows how a workload requests capacity and the kernel enforces gates.
"""

from capacity_kernel import CapacityKernel, CapacityVector
from datetime import timedelta


def main():
    print("=" * 60)
    print("Capacity-Governed Systems Platform - Demo")
    print("=" * 60)
    
    # Initialize kernel for a 4D topology cluster
    kernel = CapacityKernel(d_max=4, k_gluing=2.0)
    print(f"\n[1] Initialized kernel: d_max={kernel.d_max}, k_gluing={kernel.k_gluing}")
    
    # Workload 1: Payment service (high reliability needs)
    print("\n[2] Workload 1: Payment service requesting capacity...")
    payment_c = CapacityVector(C_geo=0.8, C_int=0.7, C_ptr=0.9, C_obs=0.5)
    token1 = kernel.request_capacity(
        workload_id="payment-service",
        C=payment_c,
        duration=timedelta(minutes=5)
    )
    
    if token1:
        print(f"   ✅ ADMITTED (token: {token1.token_id})")
        print(f"   Capacity: C_geo={payment_c.C_geo}, C_int={payment_c.C_int}")
    else:
        print("   ❌ REJECTED")
    
    # Workload 2: Batch job (lower reliability, burst capacity)
    print("\n[3] Workload 2: Batch analytics requesting capacity...")
    batch_c = CapacityVector(C_geo=0.5, C_int=0.3, C_ptr=0.4, C_obs=0.2)
    token2 = kernel.request_capacity(
        workload_id="batch-analytics",
        C=batch_c,
        duration=timedelta(hours=1)
    )
    
    if token2:
        print(f"   ✅ ADMITTED (token: {token2.token_id})")
        print(f"   Capacity: C_geo={batch_c.C_geo}, C_int={batch_c.C_int}")
    else:
        print("   ❌ REJECTED")
    
    # Check capacity during operation
    print("\n[4] Checking capacity for payment service...")
    valid = kernel.check_capacity(token1)
    print(f"   {'✅ VALID' if valid else '❌ INVALID'}")
    
    if kernel.gate_state:
        print(f"\n   Gate State: {kernel.gate_state}")
        print(f"   - fit: {kernel.gate_state.fit}")
        print(f"   - gluing: {kernel.gate_state.gluing}")
        print(f"   - uv: {kernel.gate_state.uv}")
        print(f"   - isolation: {kernel.gate_state.isolation}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
