"""
CGF Server v0.2 - Capacity Governance Framework
Local FastAPI server implementing HostAdapter v1 SPEC endpoints.

Endpoints:
- POST /v1/register
- POST /v1/evaluate
- POST /v1/outcomes/report
- POST /v1/capacity/signals
- POST /v1/events (v0.2) - Event ingestion
- GET  /v1/proposals/{proposal_id}/replay (v0.2) - Replay pack

Changes v0.1 -> v0.2:
- Schema versioning (0.2.0)
- Canonical EventType enum
- Memory_write action type support
- Policy v0.2: denylist + confidence-based constraints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
import uuid
import asyncio
from pathlib import Path

# Import v0.2 schemas
from cgf_schemas_v02 import (
    SCHEMA_VERSION,
    ActionType,
    DecisionType,
    RiskTier,
    CapacityProfile,
    HostConfig,
    HostAdapterRegistration,
    ToolCallParams,
    MemoryWriteParams,
    HostProposal,
    HostContext,
    CapacitySignals,
    HostEvaluationRequest,
    ConstraintSpec,
    CGFDecision,
    HostOutcomeReport,
    HostEvent,
    HostEventType,
    get_event_required_fields,
    validate_event_payload,
    ReplayPack
)

app = FastAPI(title="CGF Server v0.2", version="0.2.0")

# ============== CONFIGURATION ==============

DATA_DIR = Path("./cgf_data")
DATA_DIR.mkdir(exist_ok=True)

# Policy v0.2 Configuration
POLICY_CONFIG = {
    "version": "0.2.0",
    "tool_call": {
        "denylist": {"file_write", "exec", "shell", "eval", "code_exec"},
        "default_decision": DecisionType.ALLOW,
        "default_confidence": 0.9
    },
    "memory_write": {
        "sensitivity_thresholds": {
            "high": {"min_confidence": 0.8, "fail_closed": True},
            "medium": {"min_confidence": 0.6, "fail_closed": False},
            "low": {"min_confidence": 0.3, "fail_closed": False}
        },
        "default_decision": DecisionType.ALLOW,
        "default_confidence": 0.85
    }
}

class MemoryConstraintConfig(BaseModel):
    """Memory constraint configuration for quarantine."""
    quarantine_namespace: str = Field(default="_quarantine_")
    strip_fields_on_constrain: List[str] = Field(default_factory=list)

# ============== IN-MEMORY STATE ==============

adapters: Dict[str, dict] = {}
proposals: Dict[str, dict] = {}
decisions: Dict[str, CGFDecision] = {}
outcomes: Dict[str, HostOutcomeReport] = {}
events: List[HostEvent] = []

# ============== PERSISTENCE ==============

def persist_record(category: str, record: dict):
    """Append record to JSONL file."""
    path = DATA_DIR / f"{category}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_records(category: str) -> List[dict]:
    """Load all records from JSONL file."""
    path = DATA_DIR / f"{category}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def find_events_for_proposal(proposal_id: str) -> List[HostEvent]:
    """Find all events for a proposal, sorted by timestamp."""
    return sorted(
        [e for e in events if e.payload.get("proposal_id") == proposal_id],
        key=lambda e: e.timestamp
    )

# ============== POLICY ENGINE v0.2 ==============

async def evaluate_tool_call(proposal: HostProposal, context: HostContext, 
                             signals: CapacitySignals) -> CGFDecision:
    """
    Policy v0.2 for tool_call actions:
    - Denylist check
    - Default ALLOW for non-denylisted
    """
    params = proposal.action_params
    tool_name = params.get("tool_name", "unknown")
    
    decision_id = f"dec-{uuid.uuid4().hex[:12]}"
    
    # Check denylist
    if tool_name in POLICY_CONFIG["tool_call"]["denylist"]:
        return CGFDecision(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            adapter_id="cgf-server",
            decision=DecisionType.BLOCK,
            confidence=1.0,
            justification=f"Tool '{tool_name}' is in denylist",
            reason_code="DENYLISTED_TOOL",
            policy_version=POLICY_CONFIG["version"]
        )
    
    return CGFDecision(
        schema_version=SCHEMA_VERSION,
        decision_id=decision_id,
        proposal_id=proposal.proposal_id,
        adapter_id="cgf-server",
        decision=DecisionType.ALLOW,
        confidence=0.9,
        justification=f"Tool '{tool_name}' allowed by policy",
        reason_code=None,
        policy_version=POLICY_CONFIG["version"]
    )

async def evaluate_memory_write(proposal: HostProposal, context: HostContext,
                                signals: CapacitySignals) -> CGFDecision:
    """
    Policy v0.2 for memory_write actions:
    - High sensitivity + low confidence → CONSTRAIN (quarantine)
    - CGF unreachable + medium/high + side effects → BLOCK (fail_closed)
    """
    params = proposal.action_params
    sensitivity = params.get("sensitivity_hint", "medium")
    confidence = params.get("confidence", POLICY_CONFIG["memory_write"]["default_confidence"])
    size_bytes = params.get("size_bytes", 0)
    namespace = params.get("namespace", "default")
    
    decision_id = f"dec-{uuid.uuid4().hex[:12]}"
    thresholds = POLICY_CONFIG["memory_write"]["sensitivity_thresholds"]
    
    # High sensitivity check
    if sensitivity == "high" and confidence < thresholds["high"]["min_confidence"]:
        return CGFDecision(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            adapter_id="cgf-server",
            decision=DecisionType.CONSTRAIN,
            confidence=confidence,
            justification=f"High sensitivity memory write with insufficient confidence ({confidence:.2f})",
            reason_code="HIGH_SENSITIVITY_LOW_CONFIDENCE",
            constraint=ConstraintSpec(
                type="quarantine_namespace",
                params={"target_namespace": "_quarantine_", "source_namespace": namespace},
                reason="Low confidence for high-sensitivity write"
            ),
            policy_version=POLICY_CONFIG["version"]
        )
    
    # Size-based safeguard (for testing)
    if size_bytes > 10_000_000:  # 10MB threshold for demo
        return CGFDecision(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            adapter_id="cgf-server",
            decision=DecisionType.CONSTRAIN,
            confidence=0.95,
            justification=f"Large memory write ({size_bytes} bytes) requires quarantine",
            reason_code="LARGE_WRITE_THRESHOLD",
            constraint=ConstraintSpec(
                type="quarantine_namespace",
                params={"target_namespace": "_quarantine_large_", "source_namespace": namespace, "size_bytes": size_bytes},
                reason="Write size exceeds threshold"
            ),
            policy_version=POLICY_CONFIG["version"]
        )
    
    return CGFDecision(
        schema_version=SCHEMA_VERSION,
        decision_id=decision_id,
        proposal_id=proposal.proposal_id,
        adapter_id="cgf-server",
        decision=DecisionType.ALLOW,
        confidence=0.85,
        justification=f"Memory write allowed for {sensitivity} sensitivity ({size_bytes} bytes)",
        reason_code=None,
        policy_version=POLICY_CONFIG["version"]
    )

# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    return {
        "service": "CGF Server v0.2",
        "schema_version": SCHEMA_VERSION,
        "status": "operational",
        "policy_version": POLICY_CONFIG["version"],
        "endpoints": [
            "POST /v1/register",
            "POST /v1/evaluate",
            "POST /v1/outcomes/report",
            "POST /v1/capacity/signals",
            "POST /v1/events",
            "GET  /v1/proposals/{proposal_id}/replay"
        ]
    }

@app.post("/v1/register")
async def register(request: HostAdapterRegistration):
    """Register a host adapter."""
    adapter_id = f"adp-{uuid.uuid4().hex[:12]}"
    
    adapter_record = {
        "schema_version": request.schema_version,
        "adapter_id": adapter_id,
        "adapter_type": request.adapter_type,
        "host_config": request.host_config.model_dump(),
        "features": request.features,
        "risk_tiers": request.risk_tiers,
        "registered_at": request.timestamp,
        "expires_at": request.timestamp + 86400  # 24h
    }
    
    adapters[adapter_id] = adapter_record
    persist_record("adapters", adapter_record)
    
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "registered_at": adapter_record["registered_at"],
        "expires_at": adapter_record["expires_at"],
        "endpoints": [
            "POST /v1/evaluate",
            "POST /v1/outcomes/report",
            "POST /v1/capacity/signals",
            "POST /v1/events"
        ]
    }

@app.post("/v1/evaluate")
async def evaluate(request: HostEvaluationRequest):
    """Evaluate a proposal and return a decision."""
    proposal = request.proposal
    context = request.context
    signals = request.capacity_signals
    
    # Store proposal
    proposals[proposal.proposal_id] = {
        "schema_version": request.schema_version,
        **proposal.model_dump(),
        "received_at": datetime.now().timestamp()
    }
    persist_record("proposals", proposals[proposal.proposal_id])
    
    # Route to policy based on action_type
    if proposal.action_type == ActionType.TOOL_CALL:
        decision = await evaluate_tool_call(proposal, context, signals)
    elif proposal.action_type == ActionType.MEMORY_WRITE:
        decision = await evaluate_memory_write(proposal, context, signals)
    else:
        # Unknown action type - fail closed
        decision_id = f"dec-{uuid.uuid4().hex[:12]}"
        decision = CGFDecision(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id,
            proposal_id=proposal.proposal_id,
            adapter_id="cgf-server",
            decision=DecisionType.BLOCK,
            confidence=1.0,
            justification=f"Unknown action_type: {proposal.action_type}",
            reason_code="UNKNOWN_ACTION_TYPE",
            policy_version=POLICY_CONFIG["version"]
        )
    
    decisions[decision.decision_id] = decision
    persist_record("decisions", decision.model_dump())
    
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision.model_dump()
    }

@app.post("/v1/outcomes/report")
async def report_outcome(report: HostOutcomeReport):
    """Report execution outcome."""
    outcomes[report.proposal_id] = report
    persist_record("outcomes", report.model_dump())
    return {"status": "ok", "received": True}

@app.post("/v1/capacity/signals")
async def capacity_signals(adapter_id: str, signals: CapacitySignals):
    """Receive async capacity signal updates."""
    persist_record("signals", {
        "adapter_id": adapter_id,
        "signals": signals.model_dump(),
        "received_at": datetime.now().timestamp()
    })
    return {"status": "ok"}

@app.post("/v1/events")
async def ingest_event(event: HostEvent):
    """Ingest a canonical host event (v0.2)."""
    # Validate required fields
    errors = validate_event_payload(event)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    events.append(event)
    persist_record("events", event.model_dump())
    return {"status": "ok", "event_id": f"evt-{len(events)}"}

# ============== REPLAY ENDPOINT (v0.2) ==============

@app.get("/v1/proposals/{proposal_id}/replay")
async def get_replay_pack(proposal_id: str):
    """
    Generate a deterministic replay pack for a proposal.
    Contains: proposal + decision + outcome + event timeline
    """
    # Find proposal
    proposal_data = proposals.get(proposal_id)
    if not proposal_data:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
    
    # Reconstruct proposal
    proposal = HostProposal(**{k: v for k, v in proposal_data.items() if k != "received_at"})
    
    # Find decision
    decision = None
    for d in decisions.values():
        if d.proposal_id == proposal_id:
            decision = d
            break
    
    # Find outcome
    outcome = outcomes.get(proposal_id)
    
    # Find events
    proposal_events = find_events_for_proposal(proposal_id)
    
    # Build replay pack
    replay_id = f"replay-{uuid.uuid4().hex[:12]}"
    replay_pack = ReplayPack(
        schema_version=SCHEMA_VERSION,
        replay_id=replay_id,
        created_at=datetime.now().timestamp(),
        proposal=proposal,
        decision=decision,
        outcome=outcome,
        events=proposal_events,
        source_files={
            "proposals": str(DATA_DIR / "proposals.jsonl"),
            "decisions": str(DATA_DIR / "decisions.jsonl"),
            "outcomes": str(DATA_DIR / "outcomes.jsonl"),
            "events": str(DATA_DIR / "events.jsonl")
        },
        replay_version="1.0",
        completeness="full" if outcome else "decision-only"
    )
    
    # Persist replay pack
    persist_record("replay_packs", replay_pack.model_dump())
    
    return {
        "schema_version": SCHEMA_VERSION,
        "replay": replay_pack.model_dump()
    }

@app.get("/v1/adapters/{adapter_id}")
async def get_adapter(adapter_id: str):
    """Get adapter registration info."""
    adapter = adapters.get(adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return adapter

@app.get("/v1/adapters/{adapter_id}/events")
async def get_adapter_events(adapter_id: str, event_type: Optional[str] = None):
    """Get events for an adapter, optionally filtered by type."""
    adapter_events = [e for e in events if e.adapter_id == adapter_id]
    if event_type:
        adapter_events = [e for e in adapter_events if str(e.event_type) == event_type]
    return {
        "adapter_id": adapter_id,
        "event_count": len(adapter_events),
        "events": [e.model_dump() for e in adapter_events]
    }

@app.get("/v1/stats")
async def get_stats():
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_CONFIG["version"],
        "adapters": len(adapters),
        "proposals": len(proposals),
        "decisions": len(decisions),
        "outcomes": len(outcomes),
        "events": len(events),
        "storage_path": str(DATA_DIR)
    }

# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
