"""Capacity-Governed Systems Platform - Phase 1 Kernel

A lightweight runtime for enforcing capacity constraints on local resources.
"""

from .kernel import CapacityKernel, CapacityVector
from .gates import GateMonitor, GateResult, GateState
from .allocators import (
    DimensionAllocator,
    CouplingAllocator,
    PointerAllocator,
    ObserverAllocator,
)
from .substrate import SubstrateState, ResourceMetrics
from .substrate_monitor import SubstrateMonitor, create_real_substrate
from .step3_truth import (
    TruthLedger,
    Step3Gates,
    OutcomeTracker,
    PredictionOutcome,
    create_truth_infrastructure,
)

__version__ = "0.2.0"
__all__ = [
    "CapacityKernel",
    "CapacityVector",
    "GateMonitor",
    "GateResult",
    "GateState",
    "DimensionAllocator",
    "CouplingAllocator",
    "PointerAllocator",
    "ObserverAllocator",
    "SubstrateState",
    "ResourceMetrics",
    "SubstrateMonitor",
    "create_real_substrate",
    "TruthLedger",
    "Step3Gates",
    "OutcomeTracker",
    "PredictionOutcome",
    "create_truth_infrastructure",
]
