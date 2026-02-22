"""Gate evaluation logic for capacity-governed systems."""

from .kernel import GateResult, GateState
from .substrate import SubstrateState
from .kernel import CapacityVector
from datetime import datetime
import numpy as np


class GateMonitor:
    """
    Monitors four Step 2 gates:
    - fit: Model prediction accuracy
    - gluing: Local coherence (no sharp thresholds)
    - uv: UV bounds on eigenvalues
    - isolation: Cross-subsystem contamination
    """
    
    def __init__(
        self,
        fit_threshold: float = 0.25,
        k_gluing: float = 2.0,
        uv_lambda1_max: float = 3.8,
        uv_lambda_int_max: float = 148.0,
        isolation_threshold: float = 0.15
    ):
        self.fit_threshold = fit_threshold
        self.k_gluing = k_gluing
        self.uv_lambda1_max = uv_lambda1_max
        self.uv_lambda_int_max = uv_lambda_int_max
        self.isolation_threshold = isolation_threshold
    
    def evaluate(
        self,
        state: SubstrateState,
        C: CapacityVector,
        N_geo: int = None
    ) -> GateState:
        """
        Evaluate all gates against current substrate state.
        
        Returns GateState with PASS/FAIL for each gate.
        """
        now = datetime.now()
        
        # Compute N_geo if not provided
        if N_geo is None:
            N_geo = state.n_geo
        
        gluing_threshold = self.k_gluing / np.sqrt(N_geo)
        
        # Evaluate each gate
        fit = self._check_fit(state)
        gluing = self._check_gluing(state, gluing_threshold)
        uv = self._check_uv(state)
        isolation = self._check_isolation(state)
        
        return GateState(fit, gluing, uv, isolation, now)
    
    def _check_fit(self, state: SubstrateState) -> GateResult:
        """Check fit gate: ||G_EFT - G_C||₂ < threshold."""
        error = state.fit_error  # From substrate measurement
        margin = self.fit_threshold - error
        passed = error < self.fit_threshold
        return GateResult("fit", passed, error, self.fit_threshold, margin)
    
    def _check_gluing(self, state: SubstrateState, threshold: float) -> GateResult:
        """Check gluing gate: overlap Δ < k/√N_geo."""
        delta = state.gluing_delta
        margin = threshold - delta
        passed = delta < threshold
        return GateResult("gluing", passed, delta, threshold, margin)
    
    def _check_uv(self, state: SubstrateState) -> GateResult:
        """Check UV bounds: λ₁ < λ₁_max, λ_int < λ_int_max."""
        # UV gate passes if both eigenvalues are within bounds
        uv_ok = (
            state.lambda1 < self.uv_lambda1_max and
            state.lambda_int < self.uv_lambda_int_max
        )
        # Use ratio of max eigenvalue to bound as metric
        uv_value = max(
            state.lambda1 / self.uv_lambda1_max,
            state.lambda_int / self.uv_lambda_int_max
        )
        margin = 1.0 - uv_value
        return GateResult("uv", uv_ok, uv_value, 1.0, margin)
    
    def _check_isolation(self, state: SubstrateState) -> GateResult:
        """Check isolation: cross-axis contamination minimal."""
        metric = state.isolation_metric
        margin = self.isolation_threshold - metric
        passed = metric < self.isolation_threshold
        return GateResult("isolation", passed, metric, self.isolation_threshold, margin)
