"""Core capacity kernel implementation."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import numpy as np
from numpy.typing import NDArray


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
    
    def __init__(self, d_max: int = 4, k_gluing: float = 2.0):
        self.d_max = d_max
        self.k_gluing = k_gluing
        self._tokens: Dict[str, CapacityToken] = {}
        self._last_gate_state: Optional[GateState] = None
        
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
        """Evaluate all four Step 2 gates."""
        # TODO: Actually measure substrate state
        # For now, return "all pass" as placeholder
        now = datetime.now()
        
        return GateState(
            fit=GateResult("fit", True, 0.017, 0.25, 0.233),
            gluing=GateResult("gluing", True, 0.002, 0.004, 0.002),
            uv=GateResult("uv", True, 0.01, 3.8, 3.79),
            isolation=GateResult("isolation", True, 0.005, 0.15, 0.145),
            timestamp=now
        )
    
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
    def gate_state(self) -> Optional[GateState]:
        """Last evaluated gate state."""
        return self._last_gate_state
