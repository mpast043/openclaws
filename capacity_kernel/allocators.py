"""Capacity allocators for each axis."""

from typing import Optional
import numpy as np
from .kernel import CapacityVector


class DimensionAllocator:
    """
    Allocates geometric capacity (C_geo) across topology dimensions.
    
    Maps C_geo ∈ [0,1] to active compute topology levels:
    - 0.0-0.2: Single-threaded
    - 0.2-0.4: Multi-threaded (cores)
    - 0.4-0.6: NUMA nodes
    - 0.6-0.8: Multi-node
    - 0.8-1.0: Multi-rack/datacenter
    """
    
    def __init__(self, d_max: int = 5):
        self.d_max = d_max  # Max topology dimensions
    
    def allocate(self, C_geo: float) -> dict:
        """Allocate resources across topology levels."""
        d_nom = C_geo * self.d_max
        active_levels = int(min(d_nom + 1, self.d_max))
        
        return {
            "threads": C_geo >= 0.1,
            "cores": C_geo >= 0.2,
            "numa": C_geo >= 0.4,
            "nodes": C_geo >= 0.6,
            "racks": C_geo >= 0.8,
            "active_levels": active_levels,
            "d_nominal": d_nom,
        }


class CouplingAllocator:
    """
    Allocates interaction capacity (C_int) for service coupling.
    
    Controls:
    - Max RPS between services
    - Connection pool sizes
    - Retry/backoff parameters
    """
    
    def allocate(self, C_int: float, baseline_rps: int = 1000) -> dict:
        """Allocate coupling budget."""
        return {
            "max_rps": int(C_int * baseline_rps),
            "connection_pool": int(10 + C_int * 90),  # 10-100 connections
            "retry_budget": C_int,  # Fraction of requests that can retry
            "circuit_breaker_threshold": max(0.1, 1.0 - C_int),  # Lower intolerance
        }


class PointerAllocator:
    """
    Allocates pointer state capacity (C_ptr).
    
    Controls:
    - Memory allocation limits
    - Object/reference count budgets
    - GC pressure targets
    """
    
    def allocate(self, C_ptr: float, baseline_memory_gb: float = 4.0) -> dict:
        """Allocate pointer state budget."""
        return {
            "memory_limit_gb": C_ptr * baseline_memory_gb,
            "max_objects": int(1e6 * C_ptr),
            "gc_target_pause_ms": int(100 * (1.0 - C_ptr)),  # Lower C_ptr = longer pauses
            "pointer_chasing_budget": int(100 * C_ptr),  # Levels of indirection
        }


class ObserverAllocator:
    """
    Allocates observer capacity (C_obs) for observability.
    
    Controls:
    - Metric cardinality
    - Log volume
    - Trace sampling rate
    """
    
    def allocate(self, C_obs: float) -> dict:
        """Allocate observer budget."""
        return {
            "metric_cardinality": int(100 * C_obs),
            "log_lines_per_sec": int(1000 * C_obs),
            "trace_sampling_rate": C_obs,
            "diagnostic_depth": int(10 * C_obs),  # How many levels to capture
        }


class GaugeAllocator:
    """
    Allocates symmetry capacity (C_gauge) for consensus.
    
    Controls:
    - Quorum size requirements
    - Consensus rounds
    - Tolerance for divergence
    """
    
    def allocate(self, C_gauge: float, baseline_nodes: int = 5) -> dict:
        """Allocate gauge budget for consensus."""
        return {
            "quorum_size": int(np.ceil(C_gauge * baseline_nodes)),
            "consistency_level": "strong" if C_gauge > 0.7 else "eventual",
            "tolerance_eps": (1.0 - C_gauge) * 0.1,
        }
