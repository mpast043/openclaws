"""Distributed Coordination (P2) - Multi-node capacity governance.

Prevents split-brain capacity allocation across a cluster of nodes.
Uses gossip for substrate sharing and consensus for capacity allocation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
from collections import deque, defaultdict
import threading
import time
import uuid
import hashlib
import json

import numpy as np


class NodeState(Enum):
    """State of a capacity node."""
    JOINING = "joining"
    ACTIVE = "active"
    SUSPECT = "suspect"  # Failed heartbeats
    LEFT = "left"


@dataclass
class NodeIdentity:
    """Identity of a capacity node in the cluster."""
    node_id: str
    host: str
    port: int
    rack: str = "default"
    region: str = "default"
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class NodeCapacity:
    """Capacity allocation on a specific node."""
    node_id: str
    timestamp: datetime
    
    # Allocated capacity on this node
    c_geo_total: float = 0.0  # Sum of all granted C_geo
    c_int_total: float = 0.0 # Sum of all granted C_int
    
    # Headroom
    c_geo_available: float = 1.0
    c_int_available: float = 1.0
    
    # Substrate health
    fit_error: float = 0.0
    gluing_delta: float = 0.0
    isolation: float = 0.0


@dataclass
class ClusterCapacityView:
    """Aggregated capacity view across cluster."""
    timestamp: datetime
    
    # Sum across all nodes
    total_c_geo_allocated: float = 0.0
    total_c_int_allocated: float = 0.0
    
    # Minimum available (bottleneck)
    min_c_geo_available: float = 0.0
    min_c_int_available: float = 0.0
    
    # Node count
    active_nodes: int = 0
    suspect_nodes: int = 0
    
    def get_cluster_capacity(self) -> Dict[str, float]:
        """Get effective cluster-wide capacity."""
        return {
            "c_geo_available": self.min_c_geo_available,
            "c_int_available": self.min_c_int_available,
            "active_nodes": self.active_nodes,
        }


@dataclass
class DistributedAllocation:
    """A capacity allocation spanning multiple nodes."""
    allocation_id: str
    workload_id: str
    primary_node: str  # Node that initiated
    
    # Capacity granted per node
    node_allocations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # node_id -> {"c_geo": x, "c_int": y}
    
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Consensus state
    votes_granted: Set[str] = field(default_factory=set)
    votes_needed: int = 0
    
    @property
    def total_c_geo(self) -> float:
        return sum(a.get("c_geo", 0) for a in self.node_allocations.values())
    
    @property
    def total_c_int(self) -> float:
        return sum(a.get("c_int", 0) for a in self.node_allocations.values())
    
    @property
    def consensus_reached(self) -> bool:
        return len(self.votes_granted) >= self.votes_needed


class GossipProtocol:
    """
    Epidemic gossip for sharing substrate states.
    
    Each node periodically shares its state with random peers.
    Converges to consistent cluster view in O(log N) rounds.
    """
    
    def __init__(
        self,
        node_id: str,
        peers: List[NodeIdentity],
        gossip_interval: float = 1.0,
        fanout: int = 3
    ):
        self.node_id = node_id
        self.peers = {p.node_id: p for p in peers}
        self.gossip_interval = gossip_interval
        self.fanout = min(fanout, len(peers))
        
        # Local state
        self.local_capacity: Optional[NodeCapacity] = None
        
        # Received states
        self.peer_states: Dict[str, NodeCapacity] = {}
        self._state_timestamps: Dict[str, datetime] = {}
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self.on_state_received: Optional[Callable[[str, NodeCapacity], None]] = None
        self.on_cluster_view_updated: Optional[Callable[[ClusterCapacityView], None]] = None
    
    def start(self) -> None:
        """Start gossip protocol."""
        self._running = True
        self._thread = threading.Thread(target=self._gossip_loop, daemon=True)
        self._thread.start()
        print(f"GossipProtocol[{self.node_id}] started (fanout={self.fanout})")
    
    def stop(self) -> None:
        """Stop gossip."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"GossipProtocol[{self.node_id}] stopped")
    
    def update_local_state(self, capacity: NodeCapacity) -> None:
        """Update local capacity state to gossip."""
        with self._lock:
            self.local_capacity = capacity
    
    def get_cluster_view(self) -> ClusterCapacityView:
        """Compute aggregated cluster capacity view."""
        with self._lock:
            view = ClusterCapacityView(timestamp=datetime.now())
            
            # Add local state
            if self.local_capacity:
                view.total_c_geo_allocated += self.local_capacity.c_geo_total
                view.total_c_int_allocated += self.local_capacity.c_int_total
                view.active_nodes += 1
                
                # Initialize mins
                view.min_c_geo_available = self.local_capacity.c_geo_available
                view.min_c_int_available = self.local_capacity.c_int_available
            
            # Aggregate peer states
            for node_id, state in self.peer_states.items():
                # Check freshness
                last_update = self._state_timestamps.get(node_id, datetime.min)
                age = (datetime.now() - last_update).total_seconds()
                
                if age < 10.0:  # Recent
                    view.total_c_geo_allocated += state.c_geo_total
                    view.total_c_int_allocated += state.c_int_total
                    view.active_nodes += 1
                    
                    view.min_c_geo_available = min(
                        view.min_c_geo_available, state.c_geo_available
                    )
                    view.min_c_int_available = min(
                        view.min_c_int_available, state.c_int_available
                    )
            
            return view
    
    def _gossip_loop(self) -> None:
        """Main gossip loop."""
        while self._running:
            self._gossip_round()
            time.sleep(self.gossip_interval)
    
    def _gossip_round(self) -> None:
        """Send local state to random peers."""
        if not self.local_capacity:
            return
        
        # Pick random peers
        peer_ids = list(self.peers.keys())
        if not peer_ids:
            # No peers configured
            return
        
        if len(peer_ids) <= self.fanout:
            targets = peer_ids
        else:
            import random
            targets = random.sample(peer_ids, self.fanout)
        
        for target_id in targets:
            if target_id == self.node_id:
                continue
            
            # Serialize and "send" (in this demo, simulate direct delivery)
            state_data = self._serialize_state(self.local_capacity)
            
            # In a real system, this would be a network RPC
            # For demo: find the target node and call _receive_state directly
            # This simulates the network layer
            
    def simulate_gossip_exchange(self, all_nodes: Dict[str, 'GossipProtocol']) -> None:
        """
        Simulate gossip exchange with all nodes.
        In a real system, this would be network RPCs.
        For demo: directly exchange states.
        """
        if not self.local_capacity:
            return
        
        state_data = self._serialize_state(self.local_capacity)
        
        # Direct exchange with all peers
        for peer_id in self.peers.keys():
            if peer_id in all_nodes and peer_id != self.node_id:
                all_nodes[peer_id]._receive_state(self.node_id, state_data)
        
    def propagate_to(self, all_nodes: Dict[str, 'GossipProtocol']) -> None:
        """Propagate state to all peer nodes."""
        if not self.local_capacity:
            return
            
        state_data = self._serialize_state(self.local_capacity)
        
        for peer_id in self.peers.keys():
            if peer_id in all_nodes and peer_id != self.node_id:
                all_nodes[peer_id]._receive_state(self.node_id, state_data)
    
    def _serialize_state(self, capacity: NodeCapacity) -> dict:
        """Serialize capacity state."""
        return {
            "node_id": capacity.node_id,
            "timestamp": capacity.timestamp.isoformat(),
            "c_geo_total": capacity.c_geo_total,
            "c_int_total": capacity.c_int_total,
            "c_geo_available": capacity.c_geo_available,
            "c_int_available": capacity.c_int_available,
            "fit_error": capacity.fit_error,
            "gluing_delta": capacity.gluing_delta,
            "isolation": capacity.isolation,
        }
    
    def _receive_state(self, from_node: str, state_data: dict) -> None:
        """Receive gossip from peer."""
        capacity = NodeCapacity(
            node_id=state_data["node_id"],
            timestamp=datetime.fromisoformat(state_data["timestamp"]),
            c_geo_total=state_data.get("c_geo_total", 0.0),
            c_int_total=state_data.get("c_int_total", 0.0),
            c_geo_available=state_data.get("c_geo_available", 1.0),
            c_int_available=state_data.get("c_int_available", 1.0),
            fit_error=state_data.get("fit_error", 0.0),
            gluing_delta=state_data.get("gluing_delta", 0.0),
            isolation=state_data.get("isolation", 0.0),
        )
        
        with self._lock:
            self.peer_states[from_node] = capacity
            self._state_timestamps[from_node] = datetime.now()
        
        if self.on_state_received:
            self.on_state_received(from_node, capacity)
        
        if self.on_cluster_view_updated:
            self.on_cluster_view_updated(self.get_cluster_view())


class ConsensusAllocator:
    """
    Distributed consensus for capacity allocation.
    
    Prevents split-brain by requiring majority of nodes to agree
    on capacity allocation. Uses simple majority voting.
    """
    
    def __init__(
        self,
        node_id: str,
        cluster_size: int,
        gossip: GossipProtocol
    ):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.gossip = gossip
        
        # Quorum = majority
        self.quorum = (cluster_size // 2) + 1
        
        # Pending allocations awaiting consensus
        self._pending: Dict[str, DistributedAllocation] = {}
        
        # Committed allocations
        self._committed: Dict[str, DistributedAllocation] = {}
        
        # Callbacks
        self.on_consensus_reached: Optional[Callable[[DistributedAllocation], None]] = None
        self.on_consensus_failed: Optional[Callable[[DistributedAllocation], None]] = None
    
    def propose_allocation(
        self,
        workload_id: str,
        c_geo: float,
        c_int: float,
        preferred_nodes: Optional[List[str]] = None
    ) -> Optional[DistributedAllocation]:
        """
        Propose a distributed capacity allocation.
        
        Returns allocation object if consensus reached, None if rejected.
        """
        allocation_id = str(uuid.uuid4())[:12]
        
        # Check cluster capacity first
        cluster_view = self.gossip.get_cluster_view()
        cluster_cap = cluster_view.get_cluster_capacity()
        
        if c_geo > cluster_cap["c_geo_available"]:
            print(f"Insufficient cluster C_geo: need {c_geo:.2f}, have {cluster_cap['c_geo_available']:.2f}")
            return None
        
        # Create allocation
        allocation = DistributedAllocation(
            allocation_id=allocation_id,
            workload_id=workload_id,
            primary_node=self.node_id,
            votes_needed=self.quorum,
            expires_at=datetime.now() + timedelta(seconds=10)
        )
        
        # Decide node allocation
        nodes = preferred_nodes or self._select_nodes(c_geo, c_int)
        
        for node in nodes:
            # Get available capacity on node
            node_cap = self._get_node_capacity(node)
            
            # Allocate what we can
            node_c_geo = min(c_geo, node_cap.c_geo_available)
            node_c_int = min(c_int, node_cap.c_int_available)
            
            if node_c_geo > 0 or node_c_int > 0:
                allocation.node_allocations[node] = {
                    "c_geo": node_c_geo,
                    "c_int": node_c_int
                }
                
                c_geo -= node_c_geo
                c_int -= node_c_int
                
                if c_geo <= 0.001 and c_int <= 0.001:
                    break
        
        if c_geo > 0.001 or c_int > 0.001:
            print(f"Could not allocate full capacity, remaining: C_geo={c_geo:.3f}, C_int={c_int:.3f}")
            return None
        
        # Store pending
        self._pending[allocation_id] = allocation
        
        # Request votes - for now, simulated that quorum nodes agree if gates pass
        self._request_votes(allocation)
        
        # Simulate synchronous consensus for demo
        allocation.votes_granted.add(self.node_id)
        
        # For demo: other nodes vote if cluster gates pass
        # In real impl: would send RPCs and collect responses
        cluster_view = self.gossip.get_cluster_view()
        nodes_alive = cluster_view.active_nodes
        
        # Add votes from nodes we're allocating on + votes needed from active nodes
        for node_id in allocation.node_allocations.keys():
            if node_id != self.node_id:
                allocation.votes_granted.add(node_id)
        
        # If still not enough votes, add from alive nodes that aren't participating
        remaining_nodes = [n for n in self.gossip.peer_states.keys() 
                          if n not in allocation.node_allocations and n != self.node_id]
        
        while len(allocation.votes_granted) < allocation.votes_needed and remaining_nodes:
            node = remaining_nodes.pop(0)
            allocation.votes_granted.add(node)
        
        # Check if consensus reached
        if allocation.consensus_reached:
            self._committed[allocation_id] = allocation
            del self._pending[allocation_id]
            
            if self.on_consensus_reached:
                self.on_consensus_reached(allocation)
            
            print(f"Consensus reached for {allocation_id[:8]}... (votes: {len(allocation.votes_granted)}/{allocation.votes_needed})")
            return allocation
        else:
            if self.on_consensus_failed:
                self.on_consensus_failed(allocation)
            print(f"Consensus failed for {allocation_id[:8]}...")
            return None
    
    def _get_node_capacity(self, node_id: str) -> NodeCapacity:
        """Get capacity for a specific node."""
        if node_id == self.node_id:
            if self.gossip.local_capacity:
                return self.gossip.local_capacity
        
        return self.gossip.peer_states.get(node_id, NodeCapacity(
            node_id=node_id,
            timestamp=datetime.now(),
            c_geo_available=0.0,
            c_int_available=0.0
        ))
    
    def _select_nodes(self, c_geo: float, c_int: float) -> List[str]:
        """Select best nodes for allocation."""
        # Simple strategy: prefer nodes with most available capacity
        candidates = []
        
        with self.gossip._lock:
            # Include self
            if self.gossip.local_capacity:
                candidates.append((self.node_id, self.gossip.local_capacity.c_geo_available))
            
            # Add peers
            for node_id, cap in self.gossip.peer_states.items():
                candidates.append((node_id, cap.c_geo_available))
        
        # Sort by available capacity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [n for n, _ in candidates]
    
    def _request_votes(self, allocation: DistributedAllocation) -> None:
        """Request votes from other nodes (simulated)."""
        # In real implementation: send RPCs, collect responses
        pass


class DistributedGateMonitor:
    """
    Monitor gates across the entire cluster.
    
    A gate fails if ANY node fails it (strictest interpretation)
    or if cluster-wide aggregate fails.
    """
    
    def __init__(self, gossip: GossipProtocol):
        self.gossip = gossip
        
        # Gate thresholds
        self.fit_threshold = 0.165
        self.gluing_threshold = 0.004
        self.isolation_threshold = 0.15
    
    def evaluate_cluster_gates(self) -> Dict[str, Any]:
        """Evaluate gates across all nodes."""
        view = self.gossip.get_cluster_view()
        
        # Gather node states
        node_states = []
        with self.gossip._lock:
            if self.gossip.local_capacity:
                node_states.append(self.gossip.local_capacity)
            node_states.extend(self.gossip.peer_states.values())
        
        if not node_states:
            return {"pass": False, "reason": "no nodes available"}
        
        # Strictest interpretation: fail if ANY node fails
        failures = []
        
        for state in node_states:
            if state.fit_error > self.fit_threshold:
                failures.append(f"{state.node_id}: fit={state.fit_error:.4f}")
            if state.gluing_delta > self.gluing_threshold:
                failures.append(f"{state.node_id}: gluing={state.gluing_delta:.4f}")
            if state.isolation > self.isolation_threshold:
                failures.append(f"{state.node_id}: isolation={state.isolation:.4f}")
        
        return {
            "pass": len(failures) == 0,
            "failures": failures,
            "nodes_checked": len(node_states),
            "active_nodes": view.active_nodes
        }


class CapacityNode:
    """
    A single node in a distributed capacity cluster.
    
    Combines local kernel with distributed coordination.
    """
    
    def __init__(
        self,
        identity: NodeIdentity,
        peers: List[NodeIdentity],
        gossip_interval: float = 1.0
    ):
        self.identity = identity
        self.state = NodeState.JOINING
        
        # Network
        self.peers = {p.node_id: p for p in peers if p.node_id != identity.node_id}
        
        # Distributed components
        self.gossip = GossipProtocol(
            node_id=identity.node_id,
            peers=list(self.peers.values()),
            gossip_interval=gossip_interval
        )
        
        self.consensus = ConsensusAllocator(
            node_id=identity.node_id,
            cluster_size=len(peers) + 1,
            gossip=self.gossip
        )
        
        self.distributed_gates = DistributedGateMonitor(self.gossip)
        
        # Local capacity state
        self.local_capacity: Optional[NodeCapacity] = None
        
        # Allocations on this node
        self._allocations: Dict[str, DistributedAllocation] = {}
    
    def start(self) -> None:
        """Start the node."""
        print(f"Node {self.identity.node_id} starting...")
        self.gossip.start()
        self.state = NodeState.ACTIVE
        print(f"Node {self.identity.node_id} active")
    
    def stop(self) -> None:
        """Stop the node."""
        self.gossip.stop()
        self.state = NodeState.LEFT
        print(f"Node {self.identity.node_id} stopped")
    
    def update_capacity(self, c_geo_used: float, c_int_used: float,
                       c_geo_available: float, c_int_available: float,
                       fit_error: float, gluing: float, isolation: float) -> None:
        """Update local capacity state."""
        self.local_capacity = NodeCapacity(
            node_id=self.identity.node_id,
            timestamp=datetime.now(),
            c_geo_total=c_geo_used,
            c_int_total=c_int_used,
            c_geo_available=c_geo_available,
            c_int_available=c_int_available,
            fit_error=fit_error,
            gluing_delta=gluing,
            isolation=isolation
        )
        
        self.gossip.update_local_state(self.local_capacity)
    
    def allocate(self, workload_id: str, c_geo: float, c_int: float
                ) -> Optional[DistributedAllocation]:
        """Allocate capacity with distributed consensus."""
        if self.state != NodeState.ACTIVE:
            print(f"Node {self.identity.node_id} not active")
            return None
        
        # Check distributed gates
        gate_result = self.distributed_gates.evaluate_cluster_gates()
        if not gate_result["pass"]:
            print(f"Distributed gates failed: {gate_result['failures']}")
            return None
        
        # Propose allocation
        return self.consensus.propose_allocation(workload_id, c_geo, c_int)
    
    def get_cluster_view(self) -> ClusterCapacityView:
        """Get aggregated cluster capacity."""
        return self.gossip.get_cluster_view()


def create_cluster(nodes: List[NodeIdentity]) -> List[CapacityNode]:
    """Factory: Create a cluster of capacity nodes."""
    capacity_nodes = []
    
    for identity in nodes:
        # Each node knows about all other nodes
        peers = [n for n in nodes if n.node_id != identity.node_id]
        node = CapacityNode(identity, peers)
        capacity_nodes.append(node)
    
    return capacity_nodes
