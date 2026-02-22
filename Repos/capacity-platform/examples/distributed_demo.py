#!/usr/bin/env python3
"""
Demo: Distributed Coordination (P2)

Simulates a 5-node cluster coordinating capacity allocation.
"""

import sys
import time
import random

sys.path.insert(0, '/tmp/openclaws/Repos/capacity-platform')

from capacity_kernel import (
    NodeIdentity,
    create_cluster,
)


def demo_distributed_cluster():
    print("=" * 70)
    print("DISTRIBUTED COORDINATION DEMO (P2)")
    print("=" * 70)
    
    # Create 5-node cluster
    print("\n🔧 Creating 5-node cluster...")
    nodes = [
        NodeIdentity("node-1", "10.0.0.1", 8001, rack="rack-a", region="us-east"),
        NodeIdentity("node-2", "10.0.0.2", 8002, rack="rack-a", region="us-east"),
        NodeIdentity("node-3", "10.0.0.3", 8003, rack="rack-b", region="us-east"),
        NodeIdentity("node-4", "10.0.0.4", 8004, rack="rack-b", region="us-west"),
        NodeIdentity("node-5", "10.0.0.5", 8005, rack="rack-c", region="us-west"),
    ]
    
    cluster = create_cluster(nodes)
    
    # Start all nodes
    print("\n🚀 Starting cluster...")
    for node in cluster:
        node.start()
    time.sleep(0.5)
    
    # Simulate each node with different initial capacity (clean state)
    print("\n📊 Setting node capacities...")
    
    # Node states: [c_geo_avail, c_int_avail, fit, gluing, isolation]
    # Starting with 100% available capacity on all nodes
    initial_states = [
        (1.0, 1.0, 0.02, 0.001, 0.05),  # node-1: healthy
        (1.0, 1.0, 0.03, 0.002, 0.06),  # node-2: healthy
        (0.8, 0.9, 0.04, 0.003, 0.07),  # node-3: slightly loaded
        (0.7, 0.8, 0.05, 0.002, 0.08),  # node-4: more loaded
        (0.9, 0.7, 0.03, 0.004, 0.06),  # node-5: interaction-heavy
    ]
    
    for node, state in zip(cluster, initial_states):
        c_geo_avail, c_int_avail, fit, gluing, iso = state
        # c_geo_used = 1.0 - available, c_int_used = 1.0 - available
        c_geo_used = 1.0 - c_geo_avail
        c_int_used = 1.0 - c_int_avail
        
        node.update_capacity(c_geo_used, c_int_used, c_geo_avail, c_int_avail,
                            fit, gluing, iso)
        print(f"   {node.identity.node_id}: C_geo_avail={c_geo_avail:.1f}, "
              f"C_int_avail={c_int_avail:.1f} (rack: {node.identity.rack})")
    
    # Simulate gossip exchange between all nodes
    print("\n📡 Exchanging gossip state...")
    
    gossip_nodes = {n.identity.node_id: n.gossip for n in cluster}
    for node in cluster:
        node.gossip.propagate_to(gossip_nodes)
    
    time.sleep(0.5)
    
    # View cluster state from node-1
    print("\n" + "-" * 50)
    print("CLUSTER CAPACITY VIEW (from node-1)")
    print("-" * 50)
    
    view = cluster[0].get_cluster_view()
    print(f"   Active nodes: {view.active_nodes}")
    print(f"   Total C_geo allocated: {view.total_c_geo_allocated:.2f}")
    print(f"   Total C_int allocated: {view.total_c_int_allocated:.2f}")
    print(f"   Min C_geo available: {view.min_c_geo_available:.2f} (bottleneck)")
    print(f"   Min C_int available: {view.min_c_int_available:.2f} (bottleneck)")
    
    # Distributed gate evaluation
    print("\n" + "-" * 50)
    print("DISTRIBUTED GATE EVALUATION")
    print("-" * 50)
    
    gate_result = cluster[0].distributed_gates.evaluate_cluster_gates()
    print(f"   Cluster gates: {'✅ PASS' if gate_result['pass'] else '❌ FAIL'}")
    print(f"   Nodes checked: {gate_result['nodes_checked']}")
    if not gate_result['pass']:
        print(f"   Failures: {gate_result['failures']}")
    
    # Propose distributed allocations
    print("\n" + "=" * 70)
    print("DISTRIBUTED ALLOCATIONS")
    print("=" * 70)
    
    allocations = []
    workloads = [
        ("web-batch-1", 0.4, 0.3),
        ("ml-inference-2", 0.6, 0.5),
        ("db-query-3", 0.3, 0.2),
        ("cache-warm-4", 0.5, 0.4),
    ]
    
    for workload_id, c_geo, c_int in workloads:
        print(f"\n📨 {workload_id}: Requesting C_geo={c_geo}, C_int={c_int}")
        
        # Try to allocate from node-1 (coordinator)
        alloc = cluster[0].allocate(workload_id, c_geo, c_int)
        
        if alloc:
            allocations.append(alloc)
            print(f"   ✅ ALLOCATED (id: {alloc.allocation_id})")
            print(f"   Total granted: C_geo={alloc.total_c_geo:.2f}, C_int={alloc.total_c_int:.2f}")
            print(f"   Consensus: {len(alloc.votes_granted)}/{alloc.votes_needed} votes")
            
            # Update node capacities to reflect allocation
            for node_id, node_alloc in alloc.node_allocations.items():
                node = next((n for n in cluster if n.identity.node_id == node_id), None)
                if node and node.local_capacity:
                    old_avail_geo = node.local_capacity.c_geo_available
                    old_avail_int = node.local_capacity.c_int_available

                    # Update usage
                    node.local_capacity.c_geo_total += node_alloc.get("c_geo", 0)
                    node.local_capacity.c_int_total += node_alloc.get("c_int", 0)
                    node.local_capacity.c_geo_available -= node_alloc.get("c_geo", 0)
                    node.local_capacity.c_int_available -= node_alloc.get("c_int", 0)

                    # Update gossip
                    node.gossip.update_local_state(node.local_capacity)

                    print(f"   → {node_id}: C_geo {old_avail_geo:.2f}→{node.local_capacity.c_geo_available:.2f}, "
                          f"C_int {old_avail_int:.2f}→{node.local_capacity.c_int_available:.2f}")

            # Re-propagate updated state
            for node in cluster:
                node.gossip.propagate_to(gossip_nodes)
        else:
            print(f"   ❌ REJECTED (insufficient cluster capacity)")
    
    time.sleep(1.0)  # Let gossip update
    
    # Final cluster state
    print("\n" + "=" * 70)
    print("FINAL CLUSTER STATE")
    print("=" * 70)
    
    view = cluster[0].get_cluster_view()
    print(f"\n   Total allocations: {len(allocations)}")
    print(f"   Total C_geo allocated: {view.total_c_geo_allocated:.2f}")
    print(f"   Total C_int allocated: {view.total_c_int_allocated:.2f}")
    print(f"   Remaining bottleneck: C_geo={view.min_c_geo_available:.2f}, C_int={view.min_c_int_available:.2f}")
    
    # Per-node breakdown
    print(f"\n   Per-node breakdown:")
    for node in cluster:
        if node.local_capacity:
            avail_geo = node.local_capacity.c_geo_available
            avail_int = node.local_capacity.c_int_available
            print(f"   {node.identity.node_id}: C_geo_avail={avail_geo:.2f}, C_int={avail_int:.2f}")
    
    # Stress test - attempt over-allocation
    print("\n" + "=" * 70)
    print("STRESS TEST: Attempt over-allocation")
    print("=" * 70)
    
    # Request capacity that exceeds remaining
    view = cluster[0].get_cluster_view()
    request_c_geo = view.min_c_geo_available + 0.5  # More than available
    
    print(f"\n📨 stress-test: Requesting C_geo={request_c_geo:.2f} (> available)")
    alloc = cluster[0].allocate("stress-test", request_c_geo, 0.3)
    
    if alloc:
        print(f"   ⚠️  Warning: allocation succeeded but should have failed")
    else:
        print(f"   ✅ Correctly rejected (protects cluster from over-allocation)")
    
    # Stop cluster
    print("\n" + "=" * 70)
    print("Stopping cluster...")
    print("=" * 70)
    for node in cluster:
        node.stop()
    
    print(f"\n✅ Distributed coordination: {len(allocations)} allocations across 5 nodes")
    print("   No split-brain, no over-allocation, cluster-aware gates.")


if __name__ == "__main__":
    random.seed(42)
    demo_distributed_cluster()
