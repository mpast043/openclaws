"""Substrate state measurement."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


@dataclass
class ResourceMetrics:
    """Resource utilization snapshot."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    
    # Spectral analogs
    cpu_load_spectrum: NDArray = field(default_factory=lambda: np.array([]))
    memory_pressure_spectrum: NDArray = field(default_factory=lambda: np.array([]))


@dataclass  
class ServiceMetrics:
    """Per-service metrics."""
    request_rate: float = 0.0  # RPS
    latency_p50: float = 0.0   # ms
    latency_p99: float = 0.0   # ms
    error_rate: float = 0.0    # Fraction
    

@dataclass
class SubstrateState:
    """
    Current state of the substrate system.
    
    This is the analog of the substrate in Framework v4.5.
    """
    # Identity
    substrate_id: str
    timestamp: datetime
    
    # Resource utilization
    resources: ResourceMetrics
    
    # Service-level metrics
    services: Dict[str, ServiceMetrics] = field(default_factory=dict)
    
    # Gate metrics (the "observables" in Framework terms)
    fit_error: float = 0.0           # ||G_EFT - G_C||₂
    gluing_delta: float = 0.0        # Overlap delta
    lambda1: float = 0.0             # Smallest eigenvalue
    lambda_int: float = 0.0          # Spectral interaction strength
    isolation_metric: float = 0.0    # Cross-axis contamination
    
    # Derived
    n_geo: int = 262144  # Default N=64^3
    
    @classmethod
    def sample(cls, substrate_id: str = "default") -> "SubstrateState":
        """Generate a sample substrate state (for testing)."""
        return cls(
            substrate_id=substrate_id,
            timestamp=datetime.now(),
            resources=ResourceMetrics(
                cpu_percent=45.0,
                memory_percent=60.0,
                network_io_mbps=100.0,
                disk_io_mbps=50.0,
            ),
            services={},
            fit_error=0.017,
            gluing_delta=0.002,
            lambda1=0.01,
            lambda_int=7.4,
            isolation_metric=0.005,
            n_geo=64**3,
        )
    
    def to_spectral(self) -> Dict[str, NDArray]:
        """
        Convert substrate state to spectral representation.
        
        This is the analog of eigenvalue decomposition.
        """
        # In practice, this would do PCA/SVD on metrics
        return {
            "cpu_spectrum": np.array([self.resources.cpu_percent]),
            "memory_spectrum": np.array([self.resources.memory_percent]),
            "service_spectrum": np.array([
                s.request_rate for s in self.services.values()
            ]) if self.services else np.array([0.0]),
        }
