# Capacity-Governed Systems Platform

## Vision

A structural-stability runtime that enforces capacity constraints on distributed systems, ensuring fit, gluing coherence, UV bounds, and isolation gates are satisfied at all times.

## Core Principle

> **"A system is stable if and only if its operational state satisfies all four Step 2 gates."**

This transforms Framework v4.5 from a mathematical tool into an operational governance layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Capacity-Governed Platform                    │
├─────────────────────────────────────────────────────────────────┤
│  API Layer       │  Governance Layer      │  Resource Layer     │
│  ─────────       │  ───────────────       │  ──────────────       │
│  REST/gRPC       │  Gate Monitor          │  Compute Nodes      │
│  Event Stream    │  Stability Controller  │  Network Links      │
│  Admin CLI       │  Capacity Allocator    │  Storage Pools      │
├──────────────────┼────────────────────────┼───────────────────────┤
│  Applications    │  Service Mesh          │  Infrastructure       │
│  (workloads)     │  (traffic shaping)     │  (bare metal/cloud)   │
└──────────────────┴────────────────────────┴───────────────────────┘
```

---

## Components

### 1. Capacity Kernel

The core runtime that maintains the capacity vector:

```python
class CapacityKernel:
    """
    Runtime instance maintaining ⃗C for a subsystem.
    """
    C: CapacityVector          # Current capacity allocation
    gates: GateState           # Real-time gate status
    substrate: SubstrateState  # Measured system state

    def tick(self):
        # 1. Measure substrate state
        state = self.substrate.measure()
        
        # 2. Compute gate metrics
        metrics = compute_capacity_metrics(state, self.C)
        self.gates.update(metrics)
        
        # 3. Check stability
        if not self.gates.all_pass():
            self.stabilize()
    
    def stabilize(self):
        # Engage stability controller to restore gates
        ...
```

### 2. Structural Governance Layer

#### 2.1 Gate Monitor

Continuously evaluates four gates:

| Gate | What it measures | System analog |
|------|------------------|---------------|
| **fit** | Predicted vs actual behavior | Model error, SLA drift |
| **gluing** | Local coherence (no sharp cuts) | Transaction consistency, cache coherence |
| **UV** | High-frequency bounds | Rate limits, DoS protection |
| **isolation** | Cross-system contamination | Noisy neighbor detection, side-channel resistance |

```python
@dataclass
class GateMonitor:
    fit_threshold: float       # Max allowed model error
    gluing_k: float           # Coherence parameter (k/√N)
    uv_lambda1: float         # Min resolvable frequency
    uv_lambda_max: float      # Max allowed frequency
    isolation_eps: float      # Max cross-talk
    
    def evaluate(self, state: SubstrateState, C: CapacityVector) -> GateResult:
        ...
```

#### 2.2 Stability Controller

When gates fail, the controller acts:

1. **Immediate:** Rate limit, shed load, shed cache
2. **Short-term:** Redistribute capacity across axes
3. **Long-term:** Request infrastructure scaling

Control actions:

```python
class StabilityController:
    def on_fit_violation(self):
        # Reduce aggressive prediction, fallback to conservative
        ...
    
    def on_gluing_violation(self):
        # Add coherence points, reduce partition tolerance
        ...
    
    def on_uv_violation(self):
        # Apply rate limits, increase batch sizes
        ...
    
    def on_isolation_violation(self):
        # Reschedule, migrate, or fence workloads
        ...
```

### 3. Capacity Allocators (Per Axis)

Each capacity axis gets its own allocator:

| Axis | Allocator | Resource Type |
|------|-----------|---------------|
| C_geo | DimensionAllocator | Compute topology (threads → nodes → racks) |
| C_int | CouplingAllocator | Connection bandwidth, message rates |
| C_gauge | SymmetryAllocator | Consensus rounds, quorum size |
| C_ptr | PointerAllocator | Memory allocation, GC pressure, coherence |
| C_obs | ObserverAllocator | Metric collection, logging, tracing |

#### Example: Dimension Allocator (C_geo)

```python
class DimensionAllocator:
    """
    Manages geometric capacity: how many dimensions of compute topology
    are active (threads → cores → numa → nodes → racks → regions).
    """
    def allocate(self, C_geo: float, workload: WorkloadSpec) -> TopologyAssignment:
        d_nom = C_geo * D_max  # D_max = datacenter dimensions
        active_dims = int(np.ceil(d_nom))
        
        # Assign workload to active topology levels
        return TopologyAssignment(
            threads_per_core=C_geo >= 0.2,
            cores_per_numa=C_geo >= 0.4,
            numa_per_node=C_geo >= 0.6,
            nodes_per_rack=C_geo >= 0.8,
            racks_active=C_geo >= 1.0
        )
```

### 4. Substrate State Measurement

What we measure:

```python
@dataclass
class SubstrateState:
    # Time-series: load, latency, errors per service
    service_metrics: Dict[str, MetricStream]
    
    # Resource utilization: CPU, memory, network, disk
    resource_metrics: ResourceMetrics
    
    # Correlation structure: cross-service coupling
    correlator_g: NDArray  # Current G_C vs G_EFT
    
    # Spectral features (eigenvalue analogs)
    eigenvalue_spectrum: Spectrum  # Principal component-like spectrum
    
    timestamp: datetime
```

---

## API Design

### Admission Control

```python
# Request capacity budget
response = platform.request_capacity(
    workload_id="payment-service",
    capacity_vector=CapacityVector(
        C_geo=0.7,    # Use up to rack-level topology
        C_int=0.5,    # Moderate coupling budget
        C_gauge=None, # No special consensus needs
        C_ptr=0.8,    # Handle many pointers/connections
        C_obs=0.3     # Light observability
    ),
    duration=timedelta(hours=1)
)

if response.admitted:
    token = response.capacity_token
    # Use token for operations
else:
    print(f"Rejected: gates_failed = {response.gates_failed}")
```

### Runtime Checks

```python
# During operation, system checks capacity bounds
result = platform.check_operation(
    token=token,
    operation_type="cross_service_call",
    estimated_load=LoadEstimate(cpu=10, memory_mb=512)
)

if not result.allowed:
    # Handle capacity exceeded
    ...
```

### Event Streaming

```protobuf
message CapacityEvent {
  string workload_id = 1;
  CapacityVector allocated = 2;
  CapacityVector consumed = 3;
  GateState gates = 4;
  StabilityAction action_taken = 5;
  google.protobuf.Timestamp timestamp = 6;
}
```

---

## Use Cases

### 1. Multi-Tenant Cloud (C_geo + C_obs)

- Tenants get isolation guarantees via C_geo dimension separation
- Observability budget (C_obs) limits metric cardinality per tenant
- Gate monitor detects noisy neighbors (isolation violation)

### 2. Microservice Mesh (C_int + C_ptr)

- Service coupling managed via C_int (allowed RPS between services)
- Pointer/state budget via C_ptr (connection pools, goroutines, memory)
- Fit gate catches cascading failures early

### 3. Consensus Protocols (C_gauge)

- Raft/Paxos quorums governed by C_gauge symmetry constraints
- Gluing gate ensures no split-brain partitions
- UV bounds prevent spam attacks

### 4. Edge Computing (All axes)

- Resource-constrained devices use tight C_geo (few active dimensions)
- C_int controls sync frequency to cloud
- All gates must pass despite limited resources

---

## Implementation Phases

### Phase 0: Core Library (✅ Done)
- ✅ Multi-axis capacity math
- ✅ Gate evaluation
- ✅ Sweep runners

### Phase 1: Standalone Kernel (In Progress)
- Capacity token service
- In-memory gate monitor
- Local resource allocator

### Phase 2: Distributed Platform
- Cluster-wide capacity consensus
- Distributed gate monitoring
- Cross-node stability coordination

### Phase 3: Ecosystem
- Kubernetes operators
- Envoy filters
- Prometheus exporters

---

## Relation to Framework v4.5

| Framework Concept | Platform Implementation |
|-------------------|------------------------|
| Substrate | Compute + network + storage fabric |
| Capacity vector ⃗C | Resource allocation request |
| Gates (fit, gluing, UV, isolation) | Admission + runtime checks |
| Correlator G | Cross-service dependency graph |
| Spectral dimension d_s | System complexity measure |
| Step 3 selection | Autoscaler decisions |

---

## Next Steps

1. **Refine gate mapping** from spectral math to system metrics
2. **Build Phase 1** standalone kernel prototype
3. **Integrate with capacity-demo** for validation
4. **Design Kubernetes operator** for Phase 2

This document is a living design. Update as implementation proceeds.
