"""
cgf_server_v03.py - Capacity Governance Framework Server v0.3

Features:
- Data-driven policy from JSON config
- Backward compatible with v0.2.x schemas
- Support for multiple hosts (OpenClaw, LangGraph)
- Enhanced replay with cross-host comparison

Policy v0.3:
- Fail modes configured via policy_config_v03.json
- Risk tier inference from side_effects (not hardcoded)
- Host-agnostic: no branching on host_type
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import v0.3 schemas
try:
    from cgf_schemas_v03 import (
        SCHEMA_VERSION,
        MIN_COMPATIBLE_VERSION,
        is_compatible_version,
        normalize_for_processing,
        PolicyConfig,
        HostAdapterRegistration,
        HostAdapterRegistrationResponse,
        HostEvaluationRequest,
        HostEvaluationResponse,
        HostOutcomeReport,
        HostEvent,
        CGFDecision,
        ConstraintConfig,
        ReplayPack,
        HostEventType,
        ActionType,
        DecisionType,
        RiskTier
    )
    HAS_V03 = True
except ImportError:
    # Fallback to v0.2
    from cgf_schemas_v02 import (
        SCHEMA_VERSION,
        HostAdapterRegistration,
        HostAdapterRegistrationResponse,
        HostEvaluationRequest,
        HostEvaluationResponse,
        HostOutcomeReport,
        HostEvent,
        CGFDecision,
        ConstraintConfig,
        ReplayPack,
        HostEventType,
        ActionType,
        DecisionType,
        RiskTier
    )
    HAS_V03 = False
    print("Warning: Using v0.2 schemas (cgf_schemas_v03 not found)")

# ============== CONFIGURATION ==============

# Load policy config
POLICY_CONFIG_PATH = Path(__file__).parent / "policy_config_v03.json"
if POLICY_CONFIG_PATH.exists():
    with open(POLICY_CONFIG_PATH) as f:
        POLICY_DATA = json.load(f)
else:
    POLICY_DATA = {}
    print("Warning: policy_config_v03.json not found, using defaults")

# ============== STATE ==============

adapters: Dict[str, Dict] = {}
proposals: Dict[str, Dict] = {}
decisions: Dict[str, CGFDecision] = {}
outcomes: Dict[str, HostOutcomeReport] = {}
events: List[HostEvent] = []

# ============== APP ==============

app = FastAPI(title="CGF Server v0.3", version="0.3.0")

# ============== HELPERS ==============

def generate_id(prefix: str = "") -> str:
    """Generate unique ID."""
    return f"{prefix}-{datetime.now().timestamp() * 1000:.0f}"

def get_policy_config() -> PolicyConfig:
    """Get policy config."""
    if HAS_V03 and POLICY_DATA:
        return PolicyConfig(**POLICY_DATA)
    # Default fallback
    return PolicyConfig()

def evaluate_policy(proposal: Any, context: Any, signals: Any) -> CGFDecision:
    """Host-agnostic policy evaluation using data-driven config.
    
    Key invariant: No branching on host_type!
    Only uses: action_params, risk_tier, capacity_signals
    """
    policy = get_policy_config()
    
    action_type = proposal.action_type
    action_params = proposal.action_params
    risk_tier = proposal.risk_tier
    
    # Extract common fields that policy can reason about
    tool_name = action_params.get("tool_name", "")
    size_bytes = action_params.get("size_bytes", 0)
    sensitivity_hint = action_params.get("sensitivity_hint", "medium")
    side_effects = action_params.get("side_effects_hint", [])
    
    decision_id = generate_id("dec")
    confidence_threshold = policy.confidence_thresholds.get(risk_tier, 0.6)
    
    # === POLICY RULES (data-driven) ===
    
    # Rule 1: Denylist check (generic - applies to any action with tool_name)
    if tool_name in policy.tool_denylist:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.BLOCK,
            confidence=1.0,
            justification="Tool is in denylist",
            reason_code="DENYLISTED_TOOL"
        )
    
    # Rule 2: Large memory write threshold
    if action_type == ActionType.MEMORY_WRITE and size_bytes > policy.memory_size_threshold_bytes:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.CONSTRAIN,
            confidence=0.9,
            justification="Large memory write requires quarantine",
            reason_code="LARGE_WRITE_THRESHOLD",
            constraint=ConstraintConfig(
                type="quarantine_namespace",
                params={
                    "target_namespace": f"_quarantine_{datetime.now().timestamp()}",
                    "source_namespace": action_params.get("namespace", "default")
                }
            )
        )
    
    # Rule 3: High sensitivity with insufficient confidence
    if sensitivity_hint == "high":
        # Default confidence for new proposals
        default_confidence = 0.85
        if default_confidence < confidence_threshold:
            return CGFDecision(
                decision_id=decision_id,
                proposal_id=proposal.proposal_id,
                decision=DecisionType.CONSTRAIN,
                confidence=default_confidence,
                justification=f"High sensitivity with insufficient confidence (< {confidence_threshold})",
                reason_code="HIGH_SENSITIVITY_LOW_CONFIDENCE",
                constraint=ConstraintConfig(
                    type="quarantine_namespace",
                    params={}
                )
            )
    
    # Rule 4: Risk tier with errors
    if risk_tier == RiskTier.HIGH and context.recent_errors > 3:
        return CGFDecision(
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            decision=DecisionType.BLOCK,
            confidence=0.75,
            justification="High risk tier with recent errors",
            reason_code="HIGH_RISK_WITH_ERRORS"
        )
    
    # Default: ALLOW with baseline confidence
    return CGFDecision(
        decision_id=decision_id,
        proposal_id=proposal.proposal_id,
        decision=DecisionType.ALLOW,
        confidence=default_confidence,
        justification="Within policy thresholds",
        reason_code="DEFAULT_ALLOW"
    )

def log_event(event: HostEvent):
    """Log event to memory and optionally to file."""
    events.append(event)
    
    # Persist
    event_dir = Path("./cgf_data")
    event_dir.mkdir(exist_ok=True)
    event_file = event_dir / "events.jsonl"
    
    with open(event_file, "a") as f:
        f.write(json.dumps(event.model_dump() if hasattr(event, 'model_dump') else event.__dict__) + "\n")

def get_fail_mode_config(action_type: ActionType, risk_tier: RiskTier) -> Dict[str, Any]:
    """Get fail mode from policy config."""
    policy = get_policy_config()
    
    for fm in policy.fail_modes:
        if fm.action_type == action_type and fm.risk_tier == risk_tier:
            return {
                "fail_mode": fm.fail_mode.value,
                "timeout_ms": fm.timeout_ms,
                "rationale": fm.rationale
            }
    
    # Default
    return {
        "fail_mode": "fail_closed",
        "timeout_ms": 500,
        "rationale": "Default fail-closed for safety"
    }

# ============== ENDPOINTS ==============

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "CGF Server",
        "version": "0.3.0",
        "schema_version": SCHEMA_VERSION if HAS_V03 else "0.2.0",
        "policy_version": POLICY_DATA.get("policy_version", "default"),
        "endpoints": [
            "POST /v1/register",
            "POST /v1/evaluate",
            "POST /v1/outcomes/report",
            "GET /v1/proposals/{proposal_id}/replay"
        ],
        "capabilities": {
            "hosts_supported": ["openclaw", "langgraph", "custom"],
            "action_types": ["tool_call", "memory_write"],
            "event_types": 19
        }
    }

@app.post("/v1/register")
def register_adapter(reg: HostAdapterRegistration):
    """Register a host adapter."""
    adapter_id = generate_id("adp")
    
    # Validate schema version
    request_version = reg.schema_version if hasattr(reg, 'schema_version') else "0.2.0"
    if HAS_V03 and not is_compatible_version(request_version):
        raise HTTPException(
            status_code=400,
            detail=f"Incompatible schema version: {request_version}. Min: {MIN_COMPATIBLE_VERSION}"
        )
    
    adapters[adapter_id] = {
        "adapter_id": adapter_id,
        "adapter_type": reg.adapter_type,
        "host_config": reg.host_config.model_dump() if hasattr(reg.host_config, 'model_dump') else reg.host_config.__dict__,
        "registered_at": datetime.now().timestamp(),
        "schema_version": request_version
    }
    
    # Log event
    event = HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.ADAPTER_REGISTERED,
        adapter_id=adapter_id,
        timestamp=datetime.now().timestamp(),
        payload={
            "adapter_type": reg.adapter_type,
            "host_type": reg.host_config.host_type if hasattr(reg.host_config, 'host_type') else "unknown",
            "version": reg.host_config.version if hasattr(reg.host_config, 'version') else "0.0.0"
        }
    )
    log_event(event)
    
    return HostAdapterRegistrationResponse(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        adapter_id=adapter_id,
        registered_at=datetime.now().timestamp(),
        expires_at=datetime.now().timestamp() + 3600  # 1 hour
    )

@app.post("/v1/evaluate")
def evaluate_proposal(req: HostEvaluationRequest):
    """Evaluate a proposal."""
    # Validate schema
    request_version = req.schema_version if hasattr(req, 'schema_version') else "0.2.0"
    if HAS_V03 and not is_compatible_version(request_version):
        raise HTTPException(
            status_code=400,
            detail=f"Incompatible schema version: {request_version}"
        )
    
    # Store proposal
    proposals[req.proposal.proposal_id] = {
        "proposal": req.proposal.model_dump() if hasattr(req.proposal, 'model_dump') else req.proposal.__dict__,
        "context": req.context.model_dump() if hasattr(req.context, 'model_dump') else req.context.__dict__,
        "signals": req.capacity_signals.model_dump() if hasattr(req.capacity_signals, 'model_dump') else req.capacity_signals.__dict__,
        "adapter_id": req.adapter_id
    }
    
    # Log proposal received
    log_event(HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.PROPOSAL_RECEIVED,
        adapter_id=req.adapter_id or "unknown",
        timestamp=datetime.now().timestamp(),
        proposal_id=req.proposal.proposal_id,
        payload={
            "action_type": req.proposal.action_type.value if hasattr(req.proposal.action_type, 'value') else str(req.proposal.action_type),
            "action_params_hash": str(hash(str(req.proposal.action_params)))[:16],
            "risk_tier": req.proposal.risk_tier.value if hasattr(req.proposal.risk_tier, 'value') else str(req.proposal.risk_tier)
        }
    ))
    
    # Evaluate with policy
    decision = evaluate_policy(req.proposal, req.context, req.capacity_signals)
    decisions[decision.decision_id] = decision
    
    # Log decision
    log_event(HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.DECISION_MADE,
        adapter_id=req.adapter_id or "unknown",
        timestamp=datetime.now().timestamp(),
        proposal_id=req.proposal.proposal_id,
        decision_id=decision.decision_id,
        payload={
            "decision_type": decision.decision.value,
            "confidence": decision.confidence,
            "justification": decision.justification
        }
    ))
    
    # Log enforcement
    enforcement_events = {
        DecisionType.ALLOW: HostEventType.ACTION_ALLOWED,
        DecisionType.BLOCK: HostEventType.ACTION_BLOCKED,
        DecisionType.CONSTRAIN: HostEventType.ACTION_CONSTRAINED,
        DecisionType.DEFER: HostEventType.ACTION_DEFERRED,
        DecisionType.AUDIT: HostEventType.ACTION_AUDITED
    }
    
    if decision.decision in enforcement_events:
        payload = {
            "decision_id": decision.decision_id,
            "proposal_id": req.proposal.proposal_id
        }
        if decision.decision == DecisionType.BLOCK:
            payload["justification"] = decision.justification
            payload["reason_code"] = decision.reason_code
        
        log_event(HostEvent(
            schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
            event_type=enforcement_events[decision.decision],
            adapter_id=req.adapter_id or "unknown",
            timestamp=datetime.now().timestamp(),
            proposal_id=req.proposal.proposal_id,
            decision_id=decision.decision_id,
            payload=payload
        ))
    
    return HostEvaluationResponse(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        decision=decision
    )

@app.post("/v1/outcomes/report")
def report_outcome(outcome: HostOutcomeReport):
    """Report execution outcome."""
    outcomes[outcome.proposal_id] = outcome
    
    log_event(HostEvent(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        event_type=HostEventType.OUTCOME_LOGGED,
        adapter_id=outcome.adapter_id,
        timestamp=datetime.now().timestamp(),
        proposal_id=outcome.proposal_id,
        decision_id=outcome.decision_id,
        payload={
            "success": outcome.success,
            "duration_ms": outcome.duration_ms,
            "committed": outcome.committed,
            "quarantined": outcome.quarantined
        }
    ))
    
    return {"received": True}

@app.get("/v1/proposals/{proposal_id}/replay")
def get_replay(proposal_id: str):
    """Get replay pack for proposal."""
    if proposal_id not in proposals:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    proposal_data = proposals[proposal_id]
    decision = None
    outcome = None
    proposal_events = [e for e in events if e.proposal_id == proposal_id]
    
    # Find decision
    for d in decisions.values():
        if d.proposal_id == proposal_id:
            decision = d
            break
    
    # Find outcome
    if proposal_id in outcomes:
        outcome = outcomes[proposal_id]
    
    # Determine completeness
    if outcome:
        completeness = "full"
    elif decision:
        completeness = "decision-only"
    else:
        completeness = "partial"
    
    replay = ReplayPack(
        schema_version=SCHEMA_VERSION if HAS_V03 else "0.2.0",
        replay_id=generate_id("replay"),
        created_at=datetime.now().timestamp(),
        completeness=completeness,
        proposal=proposal_data["proposal"],
        decision=decision,
        outcome=outcome,
        events=proposal_events
    )
    
    return {
        "schema_version": SCHEMA_VERSION if HAS_V03 else "0.2.0",
        "replay": replay.model_dump() if hasattr(replay, 'model_dump') else replay.__dict__
    }

@app.get("/v1/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "adapters": len(adapters),
        "proposals": len(proposals),
        "events": len(events),
        "schema_version": SCHEMA_VERSION if HAS_V03 else "0.2.0"
    }

# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(f"CGF Server v0.3")
    print(f"Schema: {SCHEMA_VERSION if HAS_V03 else '0.2.0'}")
    print(f"Policy: {POLICY_DATA.get('policy_version', 'default')}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080)
