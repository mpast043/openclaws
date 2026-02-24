# HostAdapter v1 Specification

**Status**: Draft  
**Version**: 1.0.0  
**Date**: 2026-02-24  
**Purpose**: Define the universal integration contract between host systems and the Capacity Governance Framework (CGF)

---

## 1. Design Principles

1. **Host-agnostic**: Same contract for agents, LLMs, simulations, data pipelines
2. **Minimal surface**: Adapter does not implement policy; it forwards to CGF
3. **Enforceable**: CGF decisions must be binding or explicitly escalated
4. **Observable**: All proposals, decisions, and outcomes must be auditable
5. **Non-blocking**: Adapter must not add significant latency to host execution

---

## 2. Core Concepts

### 2.1 Host System
Any system that generates actions, maintains state, or makes decisions:
- Agent runtimes (OpenClaw, LangGraph, CrewAI)
- LLM APIs (OpenAI, Anthropic, local models)
- Simulation pipelines
- Data processing workflows

### 2.2 HostAdapter
The boundary layer that:
- Observes host activity
- Calls CGF for governance decisions
- Enforces CGF constraints on the host
- Reports outcomes for audit

### 2.3 Governance Loop

```
┌─────────────────┐     observe      ┌──────────────┐
│   Host System   │ ───────────────> │  HostAdapter │
│  (proposes X)   │                  │              │
└─────────────────┘                  └──────┬───────┘
       │                                    │
       │            ┌──────────────┐        │ evaluate()
       │            │      CGF     │ <──────┘
       │            │  (decides)   │
       │            └──────┬───────┘
       │                   │
       │ enforce(decision) │
       └───────────────────┤
                           ▼
              ┌─────────────────────┐
              │   Host executes     │
              │  (constrained by    │
              │   CGF decision)     │
              └──────────┬──────────┘
                         │
              report(outcome)
                         │
                         ▼
              ┌─────────────────────┐
              │  CGF logs outcome   │
              │  updates capacity   │
              └─────────────────────┘
```

---

## 3. HostAdapter Interface

### 3.1 Interface Definition (Python)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum, auto
import time
import asyncio
import uuid
from datetime import datetime


class Decision(Enum):
    """CGF policy decisions returned to adapter"""
    ALLOW = auto()           # Proceed with full capability
    CONSTRAIN = auto()       # Proceed with modifications
    AUDIT = auto()           # Proceed but require audit trail
    DEFER = auto()           # Hold pending human/system review
    BLOCK = auto()           # Reject proposal


class ActionType(Enum):
    """Closed set of action types for MVP (host-agnostic)"""
    TOOL_CALL = "tool_call"           # Execute external tool/function
    MESSAGE_SEND = "message_send"     # Send message to user/system
    MEMORY_WRITE = "memory_write"     # Persist state to memory
    WORKFLOW_STEP = "workflow_step"   # Advance workflow/state machine


class HostEventType(Enum):
    """Canonical event types. All adapters MUST emit these."""
    ADAPTER_REGISTERED = "adapter_registered"
    PROPOSAL_RECEIVED = "proposal_received"
    DECISION_MADE = "decision_made"
    ENFORCEMENT_STARTED = "enforcement_started"
    ENFORCEMENT_FINISHED = "enforcement_finished"
    ACTION_EXECUTED = "action_executed"
    ACTION_BLOCKED = "action_blocked"
    ACTION_DEFERRED = "action_deferred"
    CONSTRAINT_APPLIED = "constraint_applied"
    CONSTRAINT_FAILED = "constraint_failed"
    AUDIT_REQUIRED = "audit_required"
    OUTCOME_REPORTED = "outcome_reported"
    OUTCOME_LOGGED = "outcome_logged"
    CGF_UNREACHABLE = "cgf_unreachable"
    EVALUATE_TIMEOUT = "evaluate_timeout"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    EXCISION_TRIGGERED = "excision_triggered"
    ADAPTER_DISCONNECTED = "adapter_disconnected"


class FailMode(Enum):
    """Behavior when CGF unavailable or evaluation times out"""
    FAIL_CLOSED = "fail_closed"   # Block action
    FAIL_OPEN = "fail_open"       # Allow action (degraded, logged)
    DEFER = "defer"               # Queue for review


@dataclass(frozen=True)
class HostProposal:
    """What the host proposes to do"""
    proposal_id: str                 # Unique ID for this proposal
    timestamp: float                 # Host timestamp (for ordering)
    action_type: ActionType          # Closed set: tool_call, message_send, memory_write, workflow_step
    action_params: Dict[str, Any]    # Standardized per action_type (see 3.2)
    context_refs: List[str]          # Referenced context/memory IDs
    estimated_cost: Optional[Dict[str, float]] = None  # Host's estimate of cost/complexity
    risk_tier: str = "medium"        # "low" | "medium" | "high" (informs fail_mode selection)
    

@dataclass(frozen=True)  
class ConstrainedAction:
    """Modified action returned with CONSTRAIN decision"""
    original_action: HostProposal
    modified_params: Dict[str, Any]  # Parameter overrides
    allowed_tools: Optional[List[str]] = None  # Subset of requested tools
    disallowed_params: Optional[List[str]] = None  # Parameters that must be removed
    reason: str = ""                 # Why constraint was applied


@dataclass
class CGFDecision:
    """Complete decision from CGF"""
    decision: Decision
    decision_id: str                 # CGF-generated decision ID (for audit)
    confidence: float                # CGF confidence in decision (0-1)
    
    # If CONSTRAIN: the constrained action
    constraint: Optional[ConstrainedAction] = None
    
    # If AUDIT: audit requirements
    audit_level: Optional[str] = None      # "basic", "deep", "human"
    audit_hooks: Optional[List[str]] = None  # Required audit callbacks
    
    # Capacity context at decision time
    capacity_profile: Optional[Dict[str, float]] = None
    
    # Excision flags: features/classes that must be suppressed
    excised_features: Optional[List[str]] = None
    
    # Human-readable reasoning
    justification: Optional[str] = None


@dataclass(frozen=True)
class ToolSideEffect:
    """Structured side effect for tool_call actions"""
    tool_name: str
    tool_args_hash: str              # Hash of canonicalized args
    resource_touched: Optional[str] = None  # file path, URL, contact ID, etc.
    resource_type: Optional[str] = None     # "file", "url", "api", "contact", etc.


@dataclass
class ExecutionOutcome:
    """Result of host execution (reported to CGF)"""
    proposal_id: str
    decision_id: str
    executed: bool                   # Whether host actually executed
    executed_at: Optional[float] = None     # Unix timestamp when execution started
    duration_ms: Optional[float] = None     # Execution time in milliseconds
    success: Optional[bool] = None   # Did it succeed?
    result_summary: Optional[str] = None
    actual_cost: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None
    side_effects: Optional[List[Union[str, ToolSideEffect]]] = None


# ============== REQUIRED HELPER COMPONENTS ==============

class CGFClient:
    """
    Required helper: HTTP client for CGF endpoints.
    
    Responsibilities:
    - POST /v1/adapters/register
    - POST /v1/evaluate (with timeout)
    - POST /v1/outcomes/report (async)
    - POST /v1/capacity/signals (async)
    - Handle retries, circuit breakers, timeout behaviors
    
    Configuration:
    - endpoint: str (base URL of CGF)
    - timeout_ms: int (default 500)
    - max_retries: int (default 3)
    - fail_mode: FailMode (default FAIL_CLOSED)
    """
    
    def __init__(self, endpoint: str, config: Dict[str, Any]):
        self.endpoint = endpoint
        self.config = config
        self.timeout_ms = config.get("timeout_ms", 500)
        self.fail_mode = FailMode(config.get("fail_mode", "fail_closed"))
    
    def evaluate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous evaluation call. Handles timeouts per fail_mode."""
        pass
    
    def register(self, metadata: Dict[str, Any]) -> str:
        """Register adapter, return adapter_id."""
        pass
    
    async def report_async(self, outcome: ExecutionOutcome) -> None:
        """Fire-and-forget outcome reporting."""
        pass
    
    async def send_capacity_signals(self, adapter_id: str, signals: Dict[str, float]) -> None:
        """Async capacity signal update."""
        pass


class HostAdapter(ABC):
    """
    Universal adapter contract between host systems and CGF.
    
    Implementations: OpenClawAdapter, LangGraphAdapter, OpenAIAdapter, etc.
    
    Required helpers (to be implemented by concrete adapter):
    - _generate_adapter_id(): Return unique adapter identifier
    - _register_with_cgf(): Call CGFClient.register(), store adapter_id
    - _parse_decision(): Parse CGF response into CGFDecision dataclass
    - _report_async(): Wrapper for CGFClient.report_async()
    """
    
    def __init__(self, cgf_endpoint: str, host_config: Dict[str, Any]):
        """
        Initialize adapter.
        
        host_config must include:
        - host_type: str (e.g., "openclaw", "langgraph")
        - namespace: str (e.g., "production", "dev")
        - capabilities: List[str]
        - fail_mode: str ("fail_closed" | "fail_open" | "defer")
        - risk_tiers: Dict[str, FailMode] (optional per-risk overrides)
        """
        self.cgf = CGFClient(cgf_endpoint, host_config)
        self.host_config = host_config
        self.adapter_id = self._generate_adapter_id()
        self._register_with_cgf()
        
        # Fail mode configuration
        self.default_fail_mode = FailMode(host_config.get("fail_mode", "fail_closed"))
        self.risk_tiers = host_config.get("risk_tiers", {
            "high": FailMode.FAIL_CLOSED,
            "medium": FailMode.DEFER,
            "low": FailMode.FAIL_OPEN
        })
    
    # ============== OBSERVATION ==============
    
    @abstractmethod
    def observe_proposal(self, host_context: Any) -> HostProposal:
        """
        Extract proposal from host system.
        
        Called when host is about to take action.
        Returns structured proposal for CGF evaluation.
        """
        pass
    
    @abstractmethod
    def observe_context(self, host_context: Any) -> Dict[str, Any]:
        """
        Extract relevant context from host.
        
        Includes: memory state, available tools, conversation history,
        substrate state (if available), etc.
        """
        pass
    
    @abstractmethod
    def observe_capacity_signals(self, host_context: Any) -> Dict[str, float]:
        """
        Extract capacity-related signals from host.
        
        Examples:
        - Token usage rate
        - Tool call frequency
        - Memory growth rate
        - Inference latency
        - Error rate
        """
        pass
    
    # ============== CGF INVOCATION ==============
    
    def evaluate(self, proposal: HostProposal, 
                 context: Dict[str, Any],
                 signals: Dict[str, float]) -> CGFDecision:
        """
        Call CGF for governance decision.
        
        This is the synchronous evaluation call.
        Must complete within configured timeout (default: 500ms).
        """
        request = {
            "adapter_id": self.adapter_id,
            "host_config": self.host_config,
            "proposal": proposal.__dict__,
            "context": context,
            "capacity_signals": signals,
            "timestamp": time.time()
        }
        
        response = self.cgf.evaluate(request)
        return self._parse_decision(response)
    
    # ============== ENFORCEMENT ==============
    
    @abstractmethod
    def enforce_allow(self, proposal: HostProposal, 
                      decision: CGFDecision) -> Any:
        """Execute action as proposed."""
        pass
    
    @abstractmethod
    def enforce_constrain(self, proposal: HostProposal,
                          decision: CGFDecision) -> Any:
        """
        Execute with modifications per constraint.
        
        Must apply all modifications in decision.constraint.
        Must fail closed if constraint cannot be applied.
        """
        pass
    
    @abstractmethod
    def enforce_audit(self, proposal: HostProposal,
                      decision: CGFDecision) -> Any:
        """
        Execute with audit requirements.
        
        Must invoke audit hooks during/after execution.
        Must log full context for audit trail.
        """
        pass
    
    @abstractmethod
    def enforce_defer(self, proposal: HostProposal,
                      decision: CGFDecision) -> Any:
        """
        Hold proposal pending review.
        
        Returns deferred status to host.
        Must provide mechanism to resume/escalate.
        """
        pass
    
    @abstractmethod
    def enforce_block(self, proposal: HostProposal,
                      decision: CGFDecision) -> Any:
        """
        Reject proposal.
        
        Must return clear rejection to host with justification.
        Host must not execute blocked actions.
        """
        pass
    
    # ============== REPORTING ==============
    
    @abstractmethod
    def observe_execution(self, host_result: Any) -> ExecutionOutcome:
        """
        Extract execution outcome from host.
        
        Called after host executes (or attempts) action.
        """
        pass
    
    def report(self, outcome: ExecutionOutcome) -> None:
        """
        Report outcome to CGF for audit and learning.
        
        Fire-and-forget (async). Failures logged locally.
        """
        asyncio.create_task(self._report_async(outcome))
    
    @abstractmethod
    def emit_event(self, event_type: str, 
                   payload: Dict[str, Any]) -> None:
        """
        Emit structured event to control plane.
        
        Events: proposal_received, decision_made, 
                action_executed, outcome_logged, capacity_exceeded, etc.
        """
        pass
    
    # ============== LIFECYCLE ==============
    
    def governance_hook(self, host_context: Any) -> Any:
        """
        Main entry point: observe → evaluate → enforce → report.
        
        This is the synchronous governance flow.
        Host calls this before executing proposed actions.
        
        INVARIANT: If CGF unavailable/evaluate times out, decision is made
        per fail_mode configuration (see 7.2).
        """
        # 1. OBSERVE
        proposal = self.observe_proposal(host_context)
        context = self.observe_context(host_context)
        signals = self.observe_capacity_signals(host_context)
        
        self.emit_event(HostEventType.PROPOSAL_RECEIVED, {
            "proposal_id": proposal.proposal_id,
            "action_type": proposal.action_type.value,
            "risk_tier": proposal.risk_tier
        })
        
        # 2. EVALUATE (with fail_mode handling)
        try:
            decision = self.evaluate(proposal, context, signals)
            self.emit_event(HostEventType.DECISION_MADE, {
                "proposal_id": proposal.proposal_id,
                "decision": decision.decision.name,
                "decision_id": decision.decision_id,
                "confidence": decision.confidence
            })
        except (TimeoutError, ConnectionError) as e:
            # CGF unreachable or timeout - apply fail_mode
            fail_mode = self.risk_tiers.get(proposal.risk_tier, self.default_fail_mode)
            event_type = HostEventType.EVALUATE_TIMEOUT if isinstance(e, TimeoutError) else HostEventType.CGF_UNREACHABLE
            self.emit_event(event_type, {
                "proposal_id": proposal.proposal_id,
                "fail_mode": fail_mode.value,
                "risk_tier": proposal.risk_tier
            })
            return self._apply_fail_mode(proposal, fail_mode, str(e))
        
        # 3. ENFORCE (with constraint failure handling)
        self.emit_event(HostEventType.ENFORCEMENT_STARTED, {
            "proposal_id": proposal.proposal_id,
            "decision": decision.decision.name
        })
        
        result = None
        enforcement_success = True
        
        try:
            if decision.decision == Decision.ALLOW:
                result = self.enforce_allow(proposal, decision)
            elif decision.decision == Decision.CONSTRAIN:
                result = self._enforce_constrain_with_fallback(proposal, decision)
            elif decision.decision == Decision.AUDIT:
                result = self._enforce_audit_with_fallback(proposal, decision)
            elif decision.decision == Decision.DEFER:
                result = self.enforce_defer(proposal, decision)
            elif decision.decision == Decision.BLOCK:
                result = self.enforce_block(proposal, decision)
            else:
                raise ValueError(f"Unknown decision: {decision.decision}")
        except Exception as e:
            enforcement_success = False
            self.emit_event(HostEventType.CONSTRAINT_FAILED if decision.decision == Decision.CONSTRAIN else HostEventType.ENFORCEMENT_FINISHED, {
                "proposal_id": proposal.proposal_id,
                "decision": decision.decision.name,
                "error": str(e),
                "fallback": "BLOCK"
            })
            # ENFORCEMENT INVARIANT: Fail closed
            result = self.enforce_block(proposal, CGFDecision(
                decision=Decision.BLOCK,
                decision_id=f"fallback-{uuid.uuid4().hex[:8]}",
                confidence=1.0,
                justification=f"Enforcement failed: {str(e)}"
            ))
        
        if enforcement_success:
            self.emit_event(HostEventType.ENFORCEMENT_FINISHED, {
                "proposal_id": proposal.proposal_id,
                "decision": decision.decision.name,
                "success": True
            })
        
        # 4. REPORT (if execution occurred)
        if decision.decision in (Decision.ALLOW, Decision.CONSTRAIN, Decision.AUDIT) and enforcement_success:
            outcome = self.observe_execution(result)
            self.report(outcome)
            
            self.emit_event(HostEventType.OUTCOME_LOGGED, {
                "proposal_id": proposal.proposal_id,
                "executed": outcome.executed,
                "success": outcome.success,
                "duration_ms": outcome.duration_ms
            })
        
        return result
    
    def _enforce_constrain_with_fallback(self, proposal: HostProposal, decision: CGFDecision) -> Any:
        """
        Execute with constraints. 
        
        INVARIANT: If constraint cannot be applied, fail closed (treat as BLOCK).
        Must emit constraint_failed event.
        """
        try:
            result = self.enforce_constrain(proposal, decision)
            self.emit_event(HostEventType.CONSTRAINT_APPLIED, {
                "proposal_id": proposal.proposal_id,
                "modified_fields": list(decision.constraint.modified_params.keys()) if decision.constraint else [],
                "reason": decision.constraint.reason if decision.constraint else ""
            })
            return result
        except Exception as e:
            self.emit_event(HostEventType.CONSTRAINT_FAILED, {
                "proposal_id": proposal.proposal_id,
                "error": str(e),
                "fallback": "BLOCK"
            })
            raise  # Will be caught and converted to BLOCK by governance_hook
    
    def _enforce_audit_with_fallback(self, proposal: HostProposal, decision: CGFDecision) -> Any:
        """
        Execute with audit requirements.
        
        INVARIANT: If required audit hooks cannot run, fallback to DEFER or BLOCK
        based on risk tier.
        """
        try:
            return self.enforce_audit(proposal, decision)
        except Exception as e:
            # Audit failure: defer or block based on risk
            fail_mode = self.risk_tiers.get(proposal.risk_tier, FailMode.BLOCK)
            self.emit_event(HostEventType.AUDIT_REQUIRED, {
                "proposal_id": proposal.proposal_id,
                "audit_level": decision.audit_level,
                "audit_failed": True,
                "fallback": fail_mode.value
            })
            if fail_mode == FailMode.DEFER:
                return self.enforce_defer(proposal, decision)
            else:
                raise  # Will be converted to BLOCK
    
    def _apply_fail_mode(self, proposal: HostProposal, fail_mode: FailMode, reason: str) -> Any:
        """Apply fail_mode when CGF unavailable."""
        fallback_decision = CGFDecision(
            decision=Decision.BLOCK if fail_mode == FailMode.FAIL_CLOSED else Decision.DEFER if fail_mode == FailMode.DEFER else Decision.ALLOW,
            decision_id=f"failmode-{uuid.uuid4().hex[:8]}",
            confidence=1.0,
            justification=f"CGF unavailable: {reason}. Fail mode: {fail_mode.value}"
        )
        
        if fail_mode == FailMode.FAIL_CLOSED:
            return self.enforce_block(proposal, fallback_decision)
        elif fail_mode == FailMode.DEFER:
            return self.enforce_defer(proposal, fallback_decision)
        else:  # FAIL_OPEN
            self.emit_event(HostEventType.ACTION_EXECUTED, {
                "proposal_id": proposal.proposal_id,
                "note": "FAIL_OPEN: CGF unavailable, action allowed without governance"
            })
            return self.enforce_allow(proposal, fallback_decision)

---

## 3.2 Standardized Action Parameters

To maintain host-agnostic policy, `action_params` MUST use this schema per `action_type`:

### `TOOL_CALL`

```python
{
    "tool_name": str,              # Required: identifier of tool
    "tool_args": Dict[str, Any],   # Required: canonicalized arguments
    "tool_args_hash": str,         # Required: hash for audit
    "resource_hints": [            # Optional: what resource types may be touched
        {"type": "file", "path": "/tmp/output"},
        {"type": "url", "domain": "api.example.com"}
    ],
    "timeout_ms": int,             # Optional: host's default timeout
    "retry_policy": str            # Optional: identifier of retry behavior
}
```

### `MESSAGE_SEND`

```python
{
    "recipient_type": str,         # "user" | "system" | "channel"
    "recipient_id": Optional[str], # Channel/user ID if applicable
    "content_preview": str,        # First 200 chars (for audit)
    "content_hash": str,           # Hash of full content
    "message_type": str,           # "text" | "structured" | "file"
    "has_attachments": bool,
    "attachment_types": [str]      # If has_attachments, list of mime types
}
```

### `MEMORY_WRITE`

```python
{
    "memory_namespace": str,       # e.g., "session", "long_term"
    "key": str,                    # Memory key/identifier
    "value_hash": str,             # Hash of value being written
    "value_size_bytes": int,       # Size for capacity tracking
    "ttl_seconds": Optional[int],  # Time-to-live if applicable
    "overwrite": bool              # Whether this overwrites existing
}
```

### `WORKFLOW_STEP`

```python
{
    "workflow_id": str,            # Workflow identifier
    "step_id": str,                # Current step
    "step_name": str,              # Human-readable step name
    "inputs_hash": str,            # Hash of step inputs
    "transition_to": str,          # Next step if this succeeds
    "is_terminal": bool            # Whether this ends the workflow
}
```

---

## 4. CGF Endpoints (What Adapter Calls)

### 4.1 Evaluation Endpoint

```
POST /v1/evaluate

Request:
{
    "adapter_id": "openclaw-adapter-001",
    "host_config": {
        "host_type": "agent_runtime",
        "runtime": "openclaw",
        "namespace": "production",
        "capabilities": ["tool_use", "memory", "messaging"]
    },
    "proposal": {
        "proposal_id": "prop-uuid-123",
        "timestamp": 1580000000.0,
        "action_type": "tool_call",
        "action_params": {
            "tool": "web_search",
            "query": "..."
        },
        "context_refs": ["mem-456", "ctx-789"],
        "estimated_cost": {
            "tokens": 150,
            "latency_ms": 500
        }
    },
    "context": {
        "conversation_id": "conv-abc",
        "turn_number": 5,
        "available_tools": ["web_search", "file_read", "code_exec"],
        "memory_size": 2048,
        "recent_errors": 0
    },
    "capacity_signals": {
        "token_rate": 45.2,        # tokens/second
        "tool_call_rate": 0.8,     # calls/second
        "memory_growth": 1.2,      # MB/second
        "error_rate": 0.0          # errors/minute
    }
}

Response:
{
    "decision_id": "dec-uuid-456",
    "decision": "CONSTRAIN",
    "confidence": 0.87,
    "capacity_profile": {
        "C_geo": 0.72,
        "C_int": 0.45,
        "C_gauge": 0.91,
        "C_ptr": 0.68,
        "C_obs": 0.55
    },
    "constraint": {
        "modified_params": {
            "tool": "web_search",
            "query": "...",
            "max_results": 5  # <- constrained
        },
        "disallowed_params": ["timeout", "recursive"],
        "reason": "C_obs_select below threshold for unbounded search"
    },
    "excised_features": ["recursive_search"],
    "justification": "Host capacity sufficient for basic search but not for recursive/deep traversal. Constraining to max 5 results."
}
```

### 4.2 Registration Endpoint

```
POST /v1/adapters/register

Request:
{
    "adapter_type": "openclaw",
    "host_metadata": {
        "runtime_version": "1.2.3",
        "supported_actions": ["tool_call", "memory_write", "message_send"],
        "substrate_info": {
            "host": "macbook-air-m2",
            "gpu": "mps",
            "memory_gb": 16
        }
    }
}

Response:
{
    "adapter_id": "openclaw-adapter-001",
    "registered_at": "2026-02-24T09:51:00Z",
    "policy_version": "v1.2"
}
```

### 4.3 Outcome Reporting Endpoint

```
POST /v1/outcomes/report

Request:
{
    "adapter_id": "openclaw-adapter-001",
    "proposal_id": "prop-uuid-123",
    "decision_id": "dec-uuid-456",
    "executed": true,
    "success": true,
    "actual_cost": {
        "tokens": 143,
        "latency_ms": 420
    },
    "result_summary": "Retrieved 5 search results",
    "side_effects": ["web_request", "json_parse"]
}
```

### 4.4 Capacity Update Endpoint (Async)

```
POST /v1/capacity/signals

Request:
{
    "adapter_id": "openclaw-adapter-001",
    "timestamp": 1580000000.0,
    "signals": {
        "token_rate": 45.2,
        "tool_call_rate": 0.8,
        "error_rate": 0.0,
        "inference_latency_ms": 120
    }
}
```

---

## 5. Event Contract

### 5.1 Canonical Event Types (Enum)

Adapters MUST use `HostEventType` enum. All 19 event types are canonically defined:

| Enum Value | Event String | When | Required Payload |
|------------|--------------|------|------------------|
| `ADAPTER_REGISTERED` | `"adapter_registered"` | On init | `adapter_id`, `host_type`, `timestamp` |
| `PROPOSAL_RECEIVED` | `"proposal_received"` | After observe_proposal | `proposal_id`, `action_type`, `timestamp`, `risk_tier` |
| `DECISION_MADE` | `"decision_made"` | After evaluate | `proposal_id`, `decision`, `confidence`, `decision_id` |
| `ENFORCEMENT_STARTED` | `"enforcement_started"` | Before enforcement | `proposal_id`, `decision` |
| `ENFORCEMENT_FINISHED` | `"enforcement_finished"` | After enforcement (success or fail) | `proposal_id`, `success`, `error?` |
| `ACTION_EXECUTED` | `"action_executed"` | After ALLOW/CONSTRAIN/AUDIT execute | `proposal_id`, `execution_time_ms` |
| `ACTION_BLOCKED` | `"action_blocked"` | After BLOCK enforce | `proposal_id`, `justification`, `excised_features?` |
| `ACTION_DEFERRED` | `"action_deferred"` | After DEFER enforce | `proposal_id`, `escalation_path` |
| `CONSTRAINT_APPLIED` | `"constraint_applied"` | If CONSTRAIN succeeds | `proposal_id`, `modified_fields`, `reason` |
| `CONSTRAINT_FAILED` | `"constraint_failed"` | If CONSTRAIN cannot be applied | `proposal_id`, `error`, `fallback` |
| `AUDIT_REQUIRED` | `"audit_required"` | If AUDIT decision or audit hooks fail | `proposal_id`, `audit_level`, `audit_failed?` |
| `OUTCOME_REPORTED` | `"outcome_reported"` | When report() called | `proposal_id`, `outcome_hash` |
| `OUTCOME_LOGGED` | `"outcome_logged"` | After async report succeeds | `proposal_id`, `executed`, `success`, `duration_ms` |
| `CGF_UNREACHABLE` | `"cgf_unreachable"` | ConnectionError during evaluate | `proposal_id`, `fail_mode`, `risk_tier` |
| `EVALUATE_TIMEOUT` | `"evaluate_timeout"` | TimeoutError during evaluate | `proposal_id`, `fail_mode`, `risk_tier`, `timeout_ms` |
| `CAPACITY_EXCEEDED` | `"capacity_exceeded"` | If host hits limits | `capacity_axis`, `threshold`, `current_value` |
| `EXCISION_TRIGGERED` | `"excision_triggered"` | If CGF excises features | `feature_ids`, `capacity_at_excision` |
| `ADAPTER_DISCONNECTED` | `"adapter_disconnected"` | On shutdown | `adapter_id`, `reason` |

### 5.2 Event Format

All events follow this schema:

```json
{
    "event_type": "decision_made",
    "event_type_enum": "DECISION_MADE",
    "adapter_id": "openclaw-adapter-001",
    "timestamp": 1580000000.0,
    "payload": { 
        "proposal_id": "prop-uuid-123",
        ...
    },
    "correlation_id": "prop-uuid-123"
}
```

---

## 6. Lifecycle Hooks

Adapters should integrate with host lifecycle:

### 6.1 Pre-execution (REQUIRED)
Host calls `governance_hook()` before executing proposed action.
CGF has opportunity to ALLOW/CONSTRAIN/BLOCK.

### 6.2 During execution (OPTIONAL)
For long-running actions, adapter may stream progress to CGF:
```python
adapter.stream_progress(proposal_id, {
    "status": "in_progress",
    "progress_pct": 45,
    "intermediate_results": [...]
})
```

### 6.3 Post-execution (REQUIRED)
Adapter calls `report()` with execution outcome.

### 6.4 Capacity drift detection (RECOMMENDED)
Adapter periodically sends capacity signals even when no proposal active.

---

## 7. Error Handling and Fail Modes

### 7.1 Error Scenarios

| Scenario | Adapter Behavior |
|----------|------------------|
| CGF timeout (per `timeout_ms`) | Apply `fail_mode` per risk tier (default: `FAIL_CLOSED`) |
| CGF unreachable (connection error) | Apply `fail_mode` per risk tier; emit `CGF_UNREACHABLE` |
| Evaluation parse error | Log locally; `BLOCK`; emit `CONSTRAINT_FAILED`; alert control plane |
| Host rejects constraint | Log divergence; emit `CONSTRAINT_FAILED`; escalate to `BLOCK` |
| Constraint application fails | ENFORCEMENT INVARIANT: Fail closed → `BLOCK` + emit `CONSTRAINT_FAILED` |
| Audit hooks fail | Fallback per risk tier (`DEFER` for medium, `BLOCK` for high) |

### 7.2 Fail Mode Configuration

**Required in `host_config`:**

```python
{
    "fail_mode": "fail_closed",  # Default for all actions
    "risk_tiers": {              # Per-risk overrides (optional)
        "high": "fail_closed",   # Tools with side effects, outbound messaging, memory commits
        "medium": "defer",       # Queue for review
        "low": "fail_open"       # Pure computations, no external effects
    }
}
```

**Recommended defaults for OpenClaw:**

| Action Type | Default Risk Tier | Recommended Fail Mode |
|-------------|-------------------|----------------------|
| `tool_call` with side effects (file, network, code exec) | `high` | `fail_closed` |
| `message_send` (user-facing) | `high` | `fail_closed` |
| `memory_write` (persistent state) | `high` | `fail_closed` |
| `tool_call` read-only | `medium` | `defer` |
| Pure computation | `low` | `fail_open` (with logging) |

### 7.3 Enforcement Invariants

1. **Constraint Failure**: If `CONSTRAIN` decision but constraint cannot be applied → treat as `BLOCK`
2. **Audit Failure**: If `AUDIT` decision but audit hooks cannot run → fallback per risk tier
3. **CGF Unavailable**: Apply `fail_mode` based on proposal's `risk_tier`
4. **Unexpected Errors**: Always fail closed (`BLOCK`) unless explicitly configured `fail_open`

---

## 8. Performance Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Evaluation latency | <100ms | 500ms |
| Enforcement overhead | <5ms | 20ms |
| Event emission | Async, <50ms | 100ms |
| Memory footprint | <10MB | 50MB |

---

## 9. First Implementation: OpenClawAdapter

Based on this spec, the first adapter will:

1. **Observe**: Intercept message sends / tool calls / memory writes
2. **Context**: Extract session history, tool registry, memory state
3. **Signals**: Track token usage rate, tool frequency, error rate
4. **Enforce**: Modify `tool.params`, block `message.send()`, gate `memory.write()`
5. **Report**: Log tool execution results, memory commit outcomes

---

## 10. Appendix: Comparison Table

This table shows how different host types map to the HostAdapter contract:

| Aspect | OpenClaw | LangGraph | OpenAI API | Data Pipeline |
|--------|----------|-----------|------------|---------------|
| **observe_proposal** | Message send, tool call | Node transition, tool invoke | Completion request | Transform step |
| **observe_context** | Session, tools, history | Graph state, messages | Conversation, models | Pipeline state |
| **observe_capacity** | Token rate, latency | Node execution time | API rate limits | Throughput, errors |
| **enforce_allow** | Execute message | Execute node | Return completion | Execute transform |
| **enforce_constrain** | Limit tool params | Limit node selection | Truncate response | Sample data |
| **enforce_block** | Drop message | Skip node | Return error | Skip batch |
| **observe_execution** | Tool result | Node output | API response | Transform result |

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-24 | Initial spec |

---

## 12. Next Steps

1. Review and ratify this spec
2. Implement CGF endpoints (evaluation, registration, reporting)
3. Implement OpenClawAdapter as reference implementation
4. Write integration tests for all 5 decisions
5. Build control plane event consumer
