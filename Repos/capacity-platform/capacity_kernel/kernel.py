"""Core capacity kernel implementation."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import numpy as np
from numpy.typing import NDArray

from .substrate import SubstrateState
from .substrate_monitor import SubstrateMonitor


@dataclass(frozen=True)
class CapacityVector:
    """Typed capacity vector matching Framework v4.5."""
    C_geo: Optional[float] = None   # Geometric dimension budget
    C_int: Optional[float] = None   # Interaction coupling budget
    C_gauge: Optional[float] = None # Symmetry/consensus budget
    C_ptr: Optional[float] = None   # Pointer state budget
    C_obs: Optional[float] = None   # Observer complexity budget
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "C_geo": self.C_geo,
            "C_int": self.C_int,
            "C_gauge": self.C_gauge,
            "C_ptr": self.C_ptr,
            "C_obs": self.C_obs,
        }


@dataclass
class GateResult:
    """Result of gate evaluation."""
    name: str
    passed: bool
    value: float
    threshold: float
    margin: float  # How much room until threshold
    
    def __repr__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.name}: {self.value:.4f} / {self.threshold:.4f} (margin: {self.margin:+.4f})"


@dataclass
class GateState:
    """Current state of all four Step 2 gates."""
    fit: GateResult
    gluing: GateResult
    uv: GateResult
    isolation: GateResult
    timestamp: datetime
    
    @property
    def all_pass(self) -> bool:
        return all([
            self.fit.passed,
            self.gluing.passed,
            self.uv.passed,
            self.isolation.passed
        ])
    
    @property
    def failing(self) -> List[str]:
        """List of gate names that are failing."""
        return [
            g.name for g in [self.fit, self.gluing, self.uv, self.isolation]
            if not g.passed
        ]
    
    def __repr__(self) -> str:
        status = "STABLE" if self.all_pass else "UNSTABLE"
        return f"GateState[{status}] at {self.timestamp:%H:%M:%S}"


@dataclass
class CapacityToken:
    """Token representing an admitted capacity budget."""
    token_id: str
    capacity: CapacityVector
    expires_at: datetime
    workload_id: str


class CapacityKernel:
    """
    Runtime kernel maintaining capacity constraints for a subsystem.
    
    Usage:
        kernel = CapacityKernel(d_max=4)
        token = kernel.request_capacity(workload_id="svc-1", C_geo=0.7, C_int=0.5)
        
        while kernel.check_capacity(token):
            # Do work
            pass
    """
    
    def __init__(
        self,
        d_max: int = 4,
        k_gluing: float = 2.0,
        use_real_substrate: bool = False,
        monitor: Optional[SubstrateMonitor] = None
    ):
        self.d_max = d_max
        self.k_gluing = k_gluing
        self._tokens: Dict[str, CapacityToken] = {}
        self._last_gate_state: Optional[GateState] = None
        
        # Substrate measurement
        self._use_real_substrate = use_real_substrate
        self._monitor = monitor
        self._substrate_state: Optional[SubstrateState] = None
        
    def request_capacity(
        self,
        workload_id: str,
        C: CapacityVector,
        duration: timedelta = timedelta(minutes=5)
    ) -> Optional[CapacityToken]:
        """
        Request capacity allocation for a workload.
        
        Returns token if gates pass, None if rejected.
        """
        # Check against current substrate state
        gates = self._evaluate_gates(C)
        
        if not gates.all_pass:
            return None
        
        # Allocate token
        import uuid
        token = CapacityToken(
            token_id=str(uuid.uuid4())[:8],
            capacity=C,
            expires_at=datetime.now() + duration,
            workload_id=workload_id
        )
        self._tokens[token.token_id] = token
        return token
    
    def check_capacity(self, token: CapacityToken) -> bool:
        """Check if token is still valid and gates still pass."""
        # Check expiration
        if datetime.now() > token.expires_at:
            del self._tokens[token.token_id]
            return False
        
        # Re-evaluate gates
        gates = self._evaluate_gates(token.capacity)
        self._last_gate_state = gates
        
        if not gates.all_pass:
            # Trigger stabilization
            self._stabilize(gates)
            return False
        
        return True
    
    def _evaluate_gates(self, C: CapacityVector) -> GateState:
        """Evaluate all four Step 2 gates using real substrate measurements."""
        now = datetime.now()
        
        # Get substrate state - real or sample
        if self._use_real_substrate:
            if self._monitor:
                substrate = self._monitor.current_state or self._monitor.measure_now()
            else:
                # Create one-shot measurement
                from .substrate_monitor import create_real_substrate
                substrate = create_real_substrate()
        else:
            # Fallback to sample data
            substrate = SubstrateState.sample()
        
        self._substrate_state = substrate
        
        # Compute thresholds based on capacity request and substrate
        # Thresholds tighten as capacity request increases
        C_geo_val = C.C_geo or 0.5
        C_int_val = C.C_int or 0.5
        
        # Fit threshold: tighter for high geometric capacity
        fit_thresh = 0.25 * (1.0 - C_geo_val * 0.5)
        
        # Gluing threshold: system-dependent, scales with 1/sqrt(N_geo)
        gluing_thresh = self.k_gluing / np.sqrt(substrate.n_geo)
        
        # UV threshold: depends on capacity budget
        # High C_int means more interaction weight, so lower threshold
        lambda1_thresh = 3.84 * (1.0 - C_int_val * 0.5)
        lambda_int_thresh = 20.0 * lambda1_thresh  # Proportional to λ₁
        
        # Isolation threshold: tighter for high capacity
        isolation_thresh = 0.15 * (1.0 - max(C_geo_val, C_int_val) * 0.3)
        
        # Evaluate each gate
        fit = GateResult(
            "fit",
            substrate.fit_error < fit_thresh,
            substrate.fit_error,
            fit_thresh,
            fit_thresh - substrate.fit_error
        )
        
        gluing = GateResult(
            "gluing",
            substrate.gluing_delta < gluing_thresh,
            substrate.gluing_delta,
            gluing_thresh,
            gluing_thresh - substrate.gluing_delta
        )
        
        # UV gate passes if both eigenvalue checks pass
        uv_pass = substrate.lambda1 < lambda1_thresh and substrate.lambda_int < lambda_int_thresh
        uv = GateResult(
            "uv",
            uv_pass,
            substrate.lambda1,
            lambda1_thresh,
            lambda1_thresh - substrate.lambda1
        )
        
        isolation = GateResult(
            "isolation",
            substrate.isolation_metric < isolation_thresh,
            substrate.isolation_metric,
            isolation_thresh,
            isolation_thresh - substrate.isolation_metric
        )
        
        return GateState(fit, gluing, uv, isolation, now)
    
    def _stabilize(self, gates: GateState) -> None:
        """Take corrective action when gates fail."""
        failing = gates.failing
        print(f"Stabilizing: gates failing = {failing}")
        
        # Phase 1 stabilization: basic load shedding
        if "fit" in failing:
            print("  → Reducing prediction aggression")
        if "gluing" in failing:
            print("  → Adding coherence points")
        if "uv" in failing:
            print("  → Rate limiting")
        if "isolation" in failing:
            print("  → Rescheduling workloads")
    
    @property
    def substrate_state(self) -> Optional[SubstrateState]:
        """Last measured substrate state (available when using real substrate)."""
        return self._substrate_state
    
    @property
    def gate_state(self) -> Optional[GateState]:
        """Last evaluated gate state."""
        return self._last_gate_state
