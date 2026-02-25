"""
cgf_schemas_v02.py - Capacity Governance Framework v0.2 Schemas
Platform contract with versioned schemas and canonical EventType enum.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

# ============== SCHEMA VERSION ==============

SCHEMA_VERSION = "0.2.0"

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
    """Canonical EventType enum - single source of truth for 19 required events.
    
    Each event type has required payload fields documented below.
    """
    # Adapter lifecycle
    ADAPTER_REGISTERED = "adapter_registered"
    """Payload: {adapter_id, host_type, version}"""
    
    ADAPTER_DISCONNECTED = "adapter_disconnected"
    """Payload: {adapter_id, reason, duration_seconds}"""
    
    # Proposal lifecycle
    PROPOSAL_RECEIVED = "proposal_received"
    """Payload: {proposal_id, action_type, action_params_hash, risk_tier}"""
    
    PROPOSAL_ENACTED = "proposal_enacted"
    """Payload: {proposal_id, decision_id, enacted_at}"""
    
    PROPOSAL_EXPIRED = "proposal_expired"
    """Payload: {proposal_id, expired_at, ttl_seconds}"""
    
    PROPOSAL_REVOKED = "proposal_revoked"
    """Payload: {proposal_id, revoked_by, reason}"""
    
    # Decision lifecycle  
    DECISION_MADE = "decision_made"
    """Payload: {decision_id, proposal_id, decision_type, confidence, justification}"""
    
    DECISION_REJECTED = "decision_rejected"
    """Payload: {decision_id, proposal_id, rejection_reason}"""
    
    # Enforcement
    ACTION_ALLOWED = "action_allowed"
    """Payload: {decision_id, proposal_id, executed_at}"""
    
    ACTION_BLOCKED = "action_blocked"
    """Payload: {decision_id, proposal_id, justification, reason_code}"""
    
    ACTION_CONSTRAINED = "action_constrained"
    """Payload: {decision_id, proposal_id, constraint_type, constraint_params}"""
    
    ACTION_DEFERRED = "action_deferred"
    """Payload: {decision_id, proposal_id, deferred_until, queue_position}"""
    
    ACTION_AUDITED = "action_audited"
    """Payload: {decision_id, proposal_id, audit_level, audit_tags}"""
    
    ERRORS = "errors"
    """Payload: {error_type, message, stack_trace?}"""
    
    # Failure modes
    CONSTRAINT_FAILED = "constraint_failed"
    """Payload: {decision_id, proposal_id, constraint_type, error, fallback_decision}"""
    
    CGF_UNREACHABLE = "cgf_unreachable"
    """Payload: {proposal_id, endpoint, error_type, fail_mode_applied}"""
    
    EVALUATE_TIMEOUT = "evaluate_timeout"
    """Payload: {proposal_id, timeout_ms, elapsed_ms}"""
    
    # Outcomes
    OUTCOME_LOGGED = "outcome_logged"
    """Payload: {proposal_id, decision_id, success, duration_ms}"""
    
    SIDE_EFFECT_REPORTED = "side_effect_reported"
    """Payload: {proposal_id, side_effect_type, resource_modified, bytes_affected}"""

# ============== BASE SCHEMAS ==============

class BaseCGFSchema(BaseModel):
    """Base schema with schema versioning."""
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version for compatibility")
    schema_name: str = Field(default="base", description="Schema name for validation")

class CapacityProfile(BaseModel):
    """5-axis capacity vector (P0-P5 aligned)."""
    C_geo: float = Field(default=0.0, ge=0.0, le=1.0, description="Geometric capacity")
    C_int: float = Field(default=0.0, ge=0.0, le=1.0, description="Interaction capacity")  
    C_gauge: float = Field(default=0.0, ge=0.0, le=1.0, description="Gauge field capacity")
    C_ptr: float = Field(default=0.0, ge=0.0, le=1.0, description="Pointer capacity")
    C_obs: float = Field(default=0.0, ge=0.0, le=1.0, description="Observer capacity")

# ============== HOST CONFIG ==============

class HostConfig(BaseModel):
    """Host system configuration."""
    host_type: str = Field(..., description="Host type: openclaw, langgraph, openai_api, etc.")
    namespace: str = Field(default="default", description="Namespace for isolation")
    capabilities: List[str] = Field(default_factory=list, description="Supported action types")
    version: str = Field(default="0.0.0", description="Host adapter version")

class HostAdapterRegistration(BaseModel):
    """Adapter registration request."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    adapter_type: str
    host_config: HostConfig
    features: List[str] = Field(default_factory=list)
    risk_tiers: Dict[str, FailMode] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

class HostAdapterRegistrationResponse(BaseModel):
    """Adapter registration response."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    adapter_id: str
    registered_at: float
    expires_at: Optional[float] = None
    endpoints: List[str] = Field(default_factory=lambda: [
        "POST /v1/evaluate",
        "POST /v1/outcomes/report",
        "POST /v1/capacity/signals"
    ])

# ============== PROPOSALS ==============

class ToolCallParams(BaseModel):
    """Tool call action parameters."""
    tool_name: str
    tool_args_hash: str
    side_effects_hint: List[str] = Field(default_factory=list)
    idempotent_hint: bool = False
    resource_hints: List[str] = Field(default_factory=list)

class MemoryWriteParams(BaseModel):
    """Memory write action parameters (v0.2)."""
    namespace: str
    size_bytes: int = Field(ge=0, description="Estimated size of write in bytes")
    ttl: Optional[int] = Field(default=None, description="Time-to-live in seconds if applicable")
    sensitivity_hint: Literal["low", "medium", "high"] = "medium"
    content_hash: str = Field(description="SHA-256 hash of content (content not sent)")
    context_refs: List[str] = Field(default_factory=list, description="Stable IDs for context references")
    operation: Literal["create", "update", "delete", "append"] = "update"

ActionParams = Union[ToolCallParams, MemoryWriteParams]

class HostProposal(BaseModel):
    """Action proposal for evaluation."""
    proposal_id: str
    timestamp: float
    action_type: ActionType
    action_params: Dict[str, Any]  # Union[ToolCallParams, MemoryWriteParams] serialized
    context_refs: List[str]
    estimated_cost: Dict[str, Any] = Field(default_factory=dict)
    risk_tier: RiskTier = RiskTier.MEDIUM
    priority: int = Field(default=0, ge=0, le=100)

# ============== CONTEXT & SIGNALS ==============

class HostContext(BaseModel):
    """Runtime context at proposal time."""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: int = 0
    recent_errors: int = 0
    memory_growth_rate: float = 0.0
    
class CapacitySignals(BaseModel):
    """Real-time capacity signals."""
    token_rate: float = 0.0
    tool_call_rate: float = 0.0
    error_rate: float = 0.0
    memory_growth: float = 0.0

# ============== EVALUATION ==============

class HostEvaluationRequest(BaseModel):
    """Complete evaluation request."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    adapter_id: str
    host_config: HostConfig
    proposal: HostProposal
    context: HostContext
    capacity_signals: CapacitySignals = Field(default_factory=CapacitySignals)

class ConstraintSpec(BaseModel):
    """Constraint specification."""
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: str

class CGFDecision(BaseModel):
    """CGF governance decision."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    decision_id: str
    proposal_id: str
    adapter_id: str
    decision: DecisionType
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    reason_code: Optional[str] = None
    constraint: Optional[ConstraintSpec] = None
    excised_features: List[str] = Field(default_factory=list)
    capacity_profile: CapacityProfile = Field(default_factory=CapacityProfile)
    expires_at: Optional[float] = None
    policy_version: str = Field(default="0.2.0")

class HostEvaluationResponse(BaseModel):
    """Evaluation response wrapper."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    decision: CGFDecision

# ============== OUTCOMES ==============

class HostOutcomeReport(BaseModel):
    """Execution outcome report."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    adapter_id: str
    proposal_id: str
    decision_id: str
    executed: bool
    executed_at: float
    duration_ms: float = 0.0
    success: bool
    result_summary: str = ""
    committed: Optional[bool] = None          # For memory ops: was write persisted?
    quarantined: Optional[bool] = None        # For constrained memory ops
    actual_cost: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    side_effects: List[Dict[str, Any]] = Field(default_factory=list)

# ============== EVENTS ==============

class HostEvent(BaseModel):
    """Canonical host event for persistence and replay."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    event_type: HostEventType
    adapter_id: str
    timestamp: float
    proposal_id: Optional[str] = None
    decision_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class ReplayPack(BaseModel):
    """Deterministic replay pack for governance timeline reconstruction."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    replay_id: str
    created_at: float
    
    # Provenance chain
    proposal: HostProposal
    decision: CGFDecision
    outcome: Optional[HostOutcomeReport] = None
    
    # Event timeline (chronological)
    events: List[HostEvent] = Field(default_factory=list)
    
    # Pointers to source files for full replay
    source_files: Dict[str, str] = Field(default_factory=dict)
    
    # Replay metadata
    replay_version: str = "1.0"
    completeness: Literal["full", "decision-only", "partial"] = "full"

# ============== CAPACITY SIGNALS ==============

class CapacitySignalUpdate(BaseModel):
    """Async capacity signal update from host."""
    schema_version: str = Field(default=SCHEMA_VERSION)
    adapter_id: str
    timestamp: float
    signals: CapacitySignals
    source: Optional[str] = None

# ============== HELPER FUNCTIONS ==============

def get_event_required_fields(event_type: HostEventType) -> Dict[str, type]:
    """Get required payload fields for each event type."""
    requirements = {
        HostEventType.ADAPTER_REGISTERED: {"adapter_id": str, "host_type": str, "version": str},
        HostEventType.ADAPTER_DISCONNECTED: {"adapter_id": str, "reason": str, "duration_seconds": (int, float)},
        HostEventType.PROPOSAL_RECEIVED: {"proposal_id": str, "action_type": str, "action_params_hash": str, "risk_tier": str},
        HostEventType.PROPOSAL_ENACTED: {"proposal_id": str, "decision_id": str, "enacted_at": (int, float)},
        HostEventType.PROPOSAL_EXPIRED: {"proposal_id": str, "expired_at": (int, float), "ttl_seconds": (int, float)},
        HostEventType.PROPOSAL_REVOKED: {"proposal_id": str, "revoked_by": str, "reason": str},
        HostEventType.DECISION_MADE: {"decision_id": str, "proposal_id": str, "decision_type": str, "confidence": (int, float), "justification": str},
        HostEventType.DECISION_REJECTED: {"decision_id": str, "proposal_id": str, "rejection_reason": str},
        HostEventType.ACTION_ALLOWED: {"decision_id": str, "proposal_id": str, "executed_at": (int, float)},
        HostEventType.ACTION_BLOCKED: {"decision_id": str, "proposal_id": str, "justification": str, "reason_code": str},
        HostEventType.ACTION_CONSTRAINED: {"decision_id": str, "proposal_id": str, "constraint_type": str, "constraint_params": dict},
        HostEventType.ACTION_DEFERRED: {"decision_id": str, "proposal_id": str, "deferred_until": (int, float), "queue_position": int},
        HostEventType.ACTION_AUDITED: {"decision_id": str, "proposal_id": str, "audit_level": str, "audit_tags": list},
        HostEventType.ERRORS: {"error_type": str, "message": str},
        HostEventType.CONSTRAINT_FAILED: {"decision_id": str, "proposal_id": str, "constraint_type": str, "error": str, "fallback_decision": str},
        HostEventType.CGF_UNREACHABLE: {"proposal_id": str, "endpoint": str, "error_type": str, "fail_mode_applied": str},
        HostEventType.EVALUATE_TIMEOUT: {"proposal_id": str, "timeout_ms": int, "elapsed_ms": (int, float)},
        HostEventType.OUTCOME_LOGGED: {"proposal_id": str, "decision_id": str, "success": bool, "duration_ms": (int, float)},
        HostEventType.SIDE_EFFECT_REPORTED: {"proposal_id": str, "side_effect_type": str, "resource_modified": str, "bytes_affected": int},
    }
    return requirements.get(event_type, {})

def validate_event_payload(event: HostEvent) -> List[str]:
    """Validate that an event has all required payload fields. Returns list of errors."""
    required = get_event_required_fields(event.event_type)
    errors = []
    
    for field, field_type in required.items():
        if field not in event.payload:
            errors.append(f"Missing required field: {field}")
            continue
        value = event.payload[field]
        if isinstance(field_type, tuple):
            if not any(isinstance(value, t) for t in field_type):
                errors.append(f"Field {field} has wrong type: expected one of {field_type}, got {type(value)}")
        elif not isinstance(value, field_type):
            errors.append(f"Field {field} has wrong type: expected {field_type}, got {type(value)}")
    
    return errors

# Export all schemas
__all__ = [
    "SCHEMA_VERSION",
    "ActionType",
    "DecisionType", 
    "RiskTier",
    "FailMode",
    "HostEventType",
    "BaseCGFSchema",
    "CapacityProfile",
    "HostConfig",
    "HostAdapterRegistration",
    "HostAdapterRegistrationResponse",
    "ToolCallParams",
    "MemoryWriteParams",
    "ActionParams",
    "HostProposal",
    "HostContext",
    "CapacitySignals",
    "HostEvaluationRequest",
    "ConstraintSpec",
    "CGFDecision",
    "HostEvaluationResponse", 
    "HostOutcomeReport",
    "HostEvent",
    "ReplayPack",
    "CapacitySignalUpdate",
    "get_event_required_fields",
    "validate_event_payload",
]
