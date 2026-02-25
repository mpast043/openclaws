"""
cgf_schemas_v03.py - Capacity Governance Framework v0.3 Schemas
Platform contract with backward compatibility for v0.2.x

Changes v0.2 → v0.3:
- SCHEMA_VERSION = "0.3.0"
- Schema compatibility: accepts 0.2.x and 0.3.0 with additive changes only
- Data-driven policy configuration (PolicyConfig)
- Enhanced validation for cross-host compatibility
- New: LangGraph action parameters support

Backward Compatibility Rules:
- Accept payloads with schema_version "0.2.0" → map v0.2 types to v0.3
- Reject payloads with schema_version < "0.2.0"
- Unknown fields are allowed (additive changes)
- Missing required fields are rejected
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Set
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import re

# ============== SCHEMA VERSION ==============

SCHEMA_VERSION = "0.3.0"
MIN_COMPATIBLE_VERSION = "0.2.0"
COMPATIBLE_VERSIONS = {"0.2.0", "0.3.0"}

# ============== VERSION COMPATIBILITY ==============

def is_compatible_version(version: str) -> bool:
    """Check if schema version is compatible.
    
    Rules:
    - Exact match: compatible
    - 0.2.x: compatible (backward compatible)
    - < 0.2.0: incompatible
    """
    if version in COMPATIBLE_VERSIONS:
        return True
    # Accept 0.2.x patch versions
    if re.match(r"^0\.2\.\d+$", version):
        return True
    return False

def normalize_for_processing(version: str) -> str:
    """Normalize version to major.minor for processing.
    
    0.2.x → 0.2.0 (treat as 0.2.0)
    0.3.0 → 0.3.0
    """
    if re.match(r"^0\.2\.\d+$", version):
        return "0.2.0"
    return version

# ============== ENUMS ==============

class ActionType(str, Enum):
    """Canonical action types for CGF governance."""
    TOOL_CALL = "tool_call"
    MESSAGE_SEND = "message_send"
    MEMORY_WRITE = "memory_write"
    WORKFLOW_STEP = "workflow_step"

class DecisionType(str, Enum):
    """Canonical decision types."""
    ALLOW = "ALLOW"
    CONSTRAIN = "CONSTRAIN"
    AUDIT = "AUDIT"
    DEFER = "DEFER"
    BLOCK = "BLOCK"

class RiskTier(str, Enum):
    """Risk tiers for fail mode configuration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FailMode(str, Enum):
    """Fail mode behavior when CGF unavailable."""
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"
    DEFER = "defer"

class HostEventType(str, Enum):
    """Canonical EventType enum - single source of truth for 19 required events."""
    # Adapter lifecycle
    ADAPTER_REGISTERED = "adapter_registered"
    ADAPTER_DISCONNECTED = "adapter_disconnected"
    
    # Proposal lifecycle
    PROPOSAL_RECEIVED = "proposal_received"
    PROPOSAL_ENACTED = "proposal_enacted"
    PROPOSAL_EXPIRED = "proposal_expired"
    PROPOSAL_REVOKED = "proposal_revoked"
    
    # Decision lifecycle  
    DECISION_MADE = "decision_made"
    DECISION_REJECTED = "decision_rejected"
    
    # Enforcement
    ACTION_ALLOWED = "action_allowed"
    ACTION_BLOCKED = "action_blocked"
    ACTION_CONSTRAINED = "action_constrained"
    ACTION_DEFERRED = "action_deferred"
    ACTION_AUDITED = "action_audited"
    
    ERRORS = "errors"
    
    # Failure modes
    CONSTRAINT_FAILED = "constraint_failed"
    CGF_UNREACHABLE = "cgf_unreachable"
    EVALUATE_TIMEOUT = "evaluate_timeout"
    
    # Outcomes
    OUTCOME_LOGGED = "outcome_logged"
    SIDE_EFFECT_REPORTED = "side_effect_reported"

class HostType(str, Enum):
    """Known host types (extensible)."""
    OPENCLAW = "openclaw"
    LANGGRAPH = "langgraph"
    CUSTOM = "custom"

# ============== SCHEMA COMPATIBILITY WRAPPER ==============

class SchemaCompatibilityResult:
    """Result of schema compatibility check."""
    def __init__(self, is_compatible: bool, normalized_version: str, 
                 needs_migration: bool, errors: List[str] = None):
        self.is_compatible = is_compatible
        self.normalized_version = normalized_version
        self.needs_migration = needs_migration
        self.errors = errors or []

def check_schema_compatibility(payload: Dict[str, Any]) -> SchemaCompatibilityResult:
    """Check if payload schema is compatible with current version."""
    version = payload.get("schema_version", "0.2.0")  # Default to 0.2.0 for legacy
    
    if not is_compatible_version(version):
        return SchemaCompatibilityResult(
            is_compatible=False,
            normalized_version=version,
            needs_migration=False,
            errors=[f"Schema version {version} is incompatible. Minimum: {MIN_COMPATIBLE_VERSION}"]
        )
    
    normalized = normalize_for_processing(version)
    needs_migration = normalized != SCHEMA_VERSION
    
    return SchemaCompatibilityResult(
        is_compatible=True,
        normalized_version=normalized,
        needs_migration=needs_migration,
        errors=[]
    )

# ============== BASE SCHEMAS ==============

class BaseCGFSchema(BaseModel):
    """Base schema with schema versioning and compatibility check."""
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    
    @model_validator(mode='after')
    def check_version(self):
        if not is_compatible_version(self.schema_version):
            raise ValueError(f"Incompatible schema_version: {self.schema_version}")
        return self

class CapacityProfile(BaseModel):
    """5-axis capacity vector (P0-P5 aligned)."""
    C_geo: float = Field(default=0.0, ge=0.0, le=1.0)
    C_int: float = Field(default=0.0, ge=0.0, le=1.0)
    C_gauge: float = Field(default=0.0, ge=0.0, le=1.0)
    C_ptr: float = Field(default=0.0, ge=0.0, le=1.0)
    C_obs: float = Field(default=0.0, ge=0.0, le=1.0)

# ============== DATA-DRIVEN POLICY CONFIG ==============

class PolicyRuleConfig(BaseModel):
    """Single policy rule configuration."""
    condition: str  # e.g., "risk_tier == 'high' and confidence < 0.8"
    decision: DecisionType
    confidence: float  # Confidence to assign if rule matches
    justification: str
    reason_code: str

class FailModeConfig(BaseModel):
    """Fail mode configuration per (action_type, risk_tier)."""
    action_type: ActionType
    risk_tier: RiskTier
    fail_mode: FailMode
    timeout_ms: int = Field(default=500, ge=100, le=5000)
    rationale: str = "Configured fail mode for host-agnostic policy"

class PolicyConfig(BaseModel):
    """Complete data-driven policy configuration (v0.3).
    
    Loaded from JSON config file by CGF server.
    Host-agnostic: contains no OpenClaw or LangGraph-specific logic.
    """
    schema_version: str = Field(default=SCHEMA_VERSION)
    policy_version: str = Field(default="0.3.0")
    
    # Denylist: tools to always block
    tool_denylist: Set[str] = Field(default_factory=set)
    
    # Size thresholds for memory write
    memory_size_threshold_bytes: int = Field(default=10_000_000)  # 10MB
    
    # Confidence thresholds per risk tier
    confidence_thresholds: Dict[RiskTier, float] = Field(default_factory=lambda: {
        RiskTier.HIGH: 0.8,
        RiskTier.MEDIUM: 0.6,
        RiskTier.LOW: 0.3
    })
    
    # Fail mode table (data-driven, host-agnostic)
    fail_modes: List[FailModeConfig] = Field(default_factory=lambda: [
        FailModeConfig(action_type=ActionType.TOOL_CALL, risk_tier=RiskTier.HIGH, 
                      fail_mode=FailMode.FAIL_CLOSED, timeout_ms=500),
        FailModeConfig(action_type=ActionType.TOOL_CALL, risk_tier=RiskTier.MEDIUM,
                      fail_mode=FailMode.DEFER, timeout_ms=500),
        FailModeConfig(action_type=ActionType.TOOL_CALL, risk_tier=RiskTier.LOW,
                      fail_mode=FailMode.FAIL_OPEN, timeout_ms=500),
        FailModeConfig(action_type=ActionType.MEMORY_WRITE, risk_tier=RiskTier.HIGH,
                      fail_mode=FailMode.FAIL_CLOSED, timeout_ms=500),
        FailModeConfig(action_type=ActionType.MEMORY_WRITE, risk_tier=RiskTier.MEDIUM,
                      fail_mode=FailMode.FAIL_CLOSED, timeout_ms=500),
        FailModeConfig(action_type=ActionType.MEMORY_WRITE, risk_tier=RiskTier.LOW,
                      fail_mode=FailMode.FAIL_OPEN, timeout_ms=500),
    ])
    
    # Risk tier inference rules (schema-driven)
    risk_tier_rules: Dict[str, Any] = Field(default_factory=lambda: {
        "from_side_effects": {
            "write": RiskTier.HIGH,
            "read": RiskTier.LOW,
            "network": RiskTier.MEDIUM,
        }
    })
    
    policy_rules: List[PolicyRuleConfig] = Field(default_factory=lambda: [
        # Tool call denylist rule
        PolicyRuleConfig(
            condition="tool_name in denylist",
            decision=DecisionType.BLOCK,
            confidence=1.0,
            justification="Tool is in denylist",
            reason_code="DENYLISTED_TOOL"
        ),
        # Memory size rule
        PolicyRuleConfig(
            condition="size_bytes > threshold",
            decision=DecisionType.CONSTRAIN,
            confidence=0.9,
            justification="Large memory write requires quarantine",
            reason_code="LARGE_WRITE_THRESHOLD"
        ),
    ])
    
    def get_fail_mode(self, action_type: ActionType, risk_tier: RiskTier) -> FailModeConfig:
        """Get fail mode for (action_type, risk_tier) pair."""
        for fm in self.fail_modes:
            if fm.action_type == action_type and fm.risk_tier == risk_tier:
                return fm
        # Default fallback
        return FailModeConfig(
            action_type=action_type,
            risk_tier=risk_tier,
            fail_mode=FailMode.FAIL_CLOSED,
            timeout_ms=500,
            rationale="No explicit rule - defaulting to fail_closed"
        )

# ============== HOST CONFIG ==============

class HostConfig(BaseModel):
    """Host system configuration."""
    host_type: Union[str, HostType] = Field(..., description="Host type identifier")
    namespace: str = Field(default="default")
    capabilities: List[str] = Field(default_factory=list)
    version: str = Field(default="0.0.0")
    
    # v0.3: Optional host-specific metadata (for debugging, not policy logic)
    host_metadata: Optional[Dict[str, Any]] = Field(default=None)

class HostAdapterRegistration(BaseCGFSchema):
    """Adapter registration request."""
    adapter_type: str
    host_config: HostConfig
    features: List[str] = Field(default_factory=list)
    risk_tiers: Dict[str, FailMode] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

class HostAdapterRegistrationResponse(BaseCGFSchema):
    """Adapter registration response."""
    adapter_id: str
    registered_at: float
    expires_at: Optional[float] = None
    endpoints: List[str] = Field(default_factory=lambda: [
        "POST /v1/evaluate",
        "POST /v1/outcomes/report",
        "POST /v1/capacity/signals"
    ])

# ============== PROPOSAL PARAMS ==============

class ToolCallParams(BaseModel):
    """Tool call action parameters (v0.3 - host-agnostic)."""
    tool_name: str
    tool_args_hash: str
    side_effects_hint: List[str] = Field(default_factory=list)
    idempotent_hint: bool = False
    resource_hints: List[str] = Field(default_factory=list)
    
    # v0.3: Adapter computes risk_tier from side_effects_hint (data-driven)
    # This is NOT used by CGF policy directly - policy uses risk_tier from proposal

class MemoryWriteParams(BaseModel):
    """Memory write action parameters (v0.3)."""
    namespace: str
    size_bytes: int = Field(ge=0)
    ttl: Optional[int] = Field(default=None)
    sensitivity_hint: Literal["low", "medium", "high"] = "medium"
    content_hash: str
    context_refs: List[str] = Field(default_factory=list)
    operation: Literal["create", "update", "delete", "append"] = "update"

class LangGraphToolParams(BaseModel):
    """LangGraph-specific tool params (optional extension)."""
    node_id: Optional[str] = None
    thread_id: Optional[str] = None
    checkpoint_ns: Optional[str] = None

ActionParams = Union[ToolCallParams, MemoryWriteParams, LangGraphToolParams]

class HostProposal(BaseCGFSchema):
    """Action proposal for evaluation."""
    proposal_id: str
    timestamp: float
    action_type: ActionType
    action_params: Dict[str, Any]
    context_refs: List[str]
    estimated_cost: Dict[str, Any] = Field(default_factory=dict)
    risk_tier: RiskTier = RiskTier.MEDIUM
    priority: int = Field(default=0, ge=0, le=100)

# ============== CONTEXT & SIGNALS ==============

class HostContext(BaseCGFSchema):
    """Runtime context at proposal time."""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: int = 0
    recent_errors: int = 0
    memory_growth_rate: float = 0.0

class CapacitySignals(BaseCGFSchema):
    """Capacity signals from P0-P5 infrastructure."""
    token_rate: float = Field(default=0.0, ge=0)
    tool_call_rate: float = Field(default=0.0, ge=0)
    error_rate: float = Field(default=0.0, ge=0)
    memory_growth: float = Field(default=0.0)

# ============== EVALUATION ==============

class ConstraintConfig(BaseModel):
    """Constraint configuration."""
    type: str  # e.g., "quarantine_namespace", "rate_limit", "delay"
    params: Dict[str, Any] = Field(default_factory=dict)

class CGFDecision(BaseCGFSchema):
    """CGF governance decision."""
    decision_id: str
    proposal_id: str
    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    reason_code: Optional[str] = None
    constraint: Optional[ConstraintConfig] = None
    excised_features: Optional[List[str]] = None

class HostEvaluationRequest(BaseCGFSchema):
    """Evaluation request from host."""
    adapter_id: Optional[str] = None
    host_config: HostConfig
    proposal: HostProposal
    context: HostContext
    capacity_signals: CapacitySignals

class HostEvaluationResponse(BaseCGFSchema):
    """Evaluation response with decision."""
    decision: CGFDecision

# ============== OUTCOMES ==============

class HostOutcomeReport(BaseCGFSchema):
    """Outcome report from host after execution."""
    adapter_id: str
    proposal_id: str
    decision_id: str
    executed: bool
    executed_at: float
    duration_ms: float
    success: bool
    committed: bool = False
    quarantined: bool = False
    errors: List[str] = Field(default_factory=list)
    result_summary: Optional[str] = None

# ============== EVENTS ==============

EVENT_REQUIRED_FIELDS: Dict[HostEventType, Dict[str, type]] = {
    HostEventType.ADAPTER_REGISTERED: {
        "adapter_id": str,
        "host_type": str,
        "version": str
    },
    HostEventType.ADAPTER_DISCONNECTED: {
        "adapter_id": str,
        "reason": str
    },
    HostEventType.PROPOSAL_RECEIVED: {
        "proposal_id": str,
        "action_type": str,
        "action_params_hash": str,
        "risk_tier": str
    },
    HostEventType.PROPOSAL_ENACTED: {
        "proposal_id": str,
        "decision_id": str
    },
    HostEventType.PROPOSAL_EXPIRED: {
        "proposal_id": str,
        "expired_at": (int, float)
    },
    HostEventType.PROPOSAL_REVOKED: {
        "proposal_id": str,
        "revoked_by": str
    },
    HostEventType.DECISION_MADE: {
        "decision_id": str,
        "proposal_id": str,
        "decision_type": str,
        "confidence": (int, float),
        "justification": str
    },
    HostEventType.DECISION_REJECTED: {
        "decision_id": str,
        "proposal_id": str,
        "rejection_reason": str
    },
    HostEventType.ACTION_ALLOWED: {
        "decision_id": str,
        "proposal_id": str
    },
    HostEventType.ACTION_BLOCKED: {
        "decision_id": str,
        "proposal_id": str,
        "justification": str,
        "reason_code": str
    },
    HostEventType.ACTION_CONSTRAINED: {
        "decision_id": str,
        "proposal_id": str,
        "constraint_type": str
    },
    HostEventType.ACTION_DEFERRED: {
        "decision_id": str,
        "proposal_id": str,
        "deferred_until": (int, float)
    },
    HostEventType.ACTION_AUDITED: {
        "decision_id": str,
        "proposal_id": str,
        "audit_level": str
    },
    HostEventType.ERRORS: {
        "error_type": str,
        "message": str
    },
    HostEventType.CONSTRAINT_FAILED: {
        "constraint_type": str,
        "error": str
    },
    HostEventType.CGF_UNREACHABLE: {
        "proposal_id": str,
        "endpoint": str,
        "error_type": str
    },
    HostEventType.EVALUATE_TIMEOUT: {
        "proposal_id": str,
        "timeout_ms": (int, float)
    },
    HostEventType.OUTCOME_LOGGED: {
        "proposal_id": str,
        "decision_id": str
    },
    HostEventType.SIDE_EFFECT_REPORTED: {
        "proposal_id": str,
        "side_effect_type": str
    }
}

def get_event_required_fields(event_type: HostEventType) -> Dict[str, type]:
    """Get required fields for event type validation."""
    return EVENT_REQUIRED_FIELDS.get(event_type, {})

def validate_event_payload(event_type: HostEventType, payload: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate event payload has all required fields."""
    required = get_event_required_fields(event_type)
    errors = []
    
    for field, expected_type in required.items():
        if field not in payload:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(payload[field], expected_type):
            errors.append(f"Field {field}: expected {expected_type}, got {type(payload[field])}")
    
    return len(errors) == 0, errors

class HostEvent(BaseCGFSchema):
    """Canonical event for audit trail."""
    event_type: HostEventType
    adapter_id: str
    timestamp: float
    proposal_id: Optional[str] = None
    decision_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

# ============== REPLAY ==============

class ReplayPack(BaseCGFSchema):
    """Complete governance timeline for replay."""
    replay_id: str
    created_at: float
    completeness: Literal["full", "decision-only", "partial"]
    proposal: Optional[HostProposal] = None
    decision: Optional[CGFDecision] = None
    outcome: Optional[HostOutcomeReport] = None
    events: List[HostEvent] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

# ============== EVENT STORE ==============

class EventStoreSummary(BaseCGFSchema):
    """Summary of event store state."""
    total_events: int = 0
    events_by_type: Dict[str, int] = Field(default_factory=dict)
    events_by_adapter: Dict[str, int] = Field(default_factory=dict)
    time_range: Optional[tuple] = None

# ============== CROSS-HOST COMPATIBILITY ==============

class CrossHostCompatibilityReport(BaseCGFSchema):
    """Report comparing adapter behavior across hosts."""
    report_id: str
    created_at: float
    hosts_tested: List[str]
    scenarios: List[str]
    results: Dict[str, Dict[str, str]]  # host -> scenario -> "pass|fail"
    replay_comparisons: Dict[str, Dict[str, Any]]  # scenario -> comparison metrics
    compatible: bool

if __name__ == "__main__":
    # Demo: Policy config
    policy = PolicyConfig()
    print(f"Policy v{policy.policy_version}")
    print(f"Denylist: {policy.tool_denylist}")
    print(f"Memory threshold: {policy.memory_size_threshold_bytes / 1_000_000}MB")
    
    fm = policy.get_fail_mode(ActionType.TOOL_CALL, RiskTier.HIGH)
    print(f"Fail mode (TOOL_CALL, HIGH): {fm.fail_mode.value}")
