"""
CGF Server v0.1 - Capacity Governance Framework
Local FastAPI server implementing HostAdapter v1 SPEC endpoints.

Endpoints:
- POST /v1/register
- POST /v1/evaluate
- POST /v1/outcomes/report
- POST /v1/capacity/signals
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import json
import hashlib
import uuid
import asyncio
from pathlib import Path

app = FastAPI(title="CGF Server v0.1", version="1.0.0")

# ============== CONFIGURATION ==============

DENYLIST = {"file_write", "fs_write", "exec", "shell", "eval"}  # Test denylist
DATA_DIR = Path("./cgf_data")
DATA_DIR.mkdir(exist_ok=True)

# ============== PYDANTIC MODELS ==============

class CapacityProfile(BaseModel):
    """5-axis capacity profile"""
    C_geo: float = Field(ge=0.0, le=1.0, default=0.5)
    C_int: float = Field(ge=0.0, le=1.0, default=0.5)
    C_gauge: float = Field(ge=0.0, le=1.0, default=0.5)
    C_ptr: float = Field(ge=0.0, le=1.0, default=0.5)
    C_obs: float = Field(ge=0.0, le=1.0, default=0.5)

class HostConfig(BaseModel):
    """Host adapter configuration"""
    host_type: Literal["openclaw", "langgraph", "openai_api", "data_pipeline"]
    runtime: Optional[str] = None
    namespace: str = "default"
    capabilities: List[str] = []
    version: str = "0.1.0"

class RegisterRequest(BaseModel):
    """POST /v1/register request"""
    adapter_type: str
    host_metadata: HostConfig

class ToolParams(BaseModel):
    """Tool call parameters (generic)"""
    tool_name: str
    tool_args_hash: str  # SHA256 of canonicalized args
    side_effects_hint: List[str] = []  # e.g., ["file_write", "network"]
    idempotent_hint: bool = False
    resource_hints: List[Dict[str, str]] = []

class HostProposal(BaseModel):
    """Tool execution proposal"""
    proposal_id: str
    timestamp: float
    action_type: Literal["tool_call"] = "tool_call"  # v0.1: only tool_call
    action_params: ToolParams
    context_refs: List[str] = []  # session_id, agent_id, etc.
    estimated_cost: Optional[Dict[str, float]] = None
    risk_tier: Literal["high", "medium", "low"] = "medium"

class ExecutionContext(BaseModel):
    """Host execution context"""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: int = 0
    recent_errors: int = 0
    memory_growth_rate: float = 0.0

class CapacitySignals(BaseModel):
    """Real-time capacity signals"""
    token_rate: float = 0.0  # tokens/sec
    tool_call_rate: float = 0.0  # calls/sec
    error_rate: float = 0.0  # errors/sec
    memory_growth: float = 0.0  # bytes/sec

class EvaluateRequest(BaseModel):
    """POST /v1/evaluate request"""
    adapter_id: str
    host_config: Dict[str, Any]
    proposal: HostProposal
    context: ExecutionContext
    capacity_signals: CapacitySignals

class Constraint(BaseModel):
    """Constraint to apply if CONSTRAIN decision"""
    modified_args: Optional[Dict[str, Any]] = None  # Replace args with these
    denied_args: List[str] = []  # Arg keys to remove
    timeout_override_ms: Optional[int] = None
    reason: str = ""

class EvaluateResponse(BaseModel):
    """POST /v1/evaluate response"""
    decision_id: str
    decision: Literal["ALLOW", "CONSTRAIN", "AUDIT", "DEFER", "BLOCK"]
    confidence: float = Field(ge=0.0, le=1.0)
    capacity_profile: CapacityProfile
    constraint: Optional[Constraint] = None
    excised_features: List[str] = []
    justification: str

class SideEffect(BaseModel):
    """Observed side effect"""
    resource_type: str  # "file", "url", "memory", "network"
    resource_id: str
    operation: str  # "read", "write", "delete", "execute"

class ExecutionOutcome(BaseModel):
    """POST /v1/outcomes/report request"""
    adapter_id: str
    proposal_id: str
    decision_id: str
    executed: bool
    executed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    result_summary: Optional[str] = None
    actual_cost: Optional[Dict[str, float]] = None
    errors: List[str] = []
    side_effects: List[SideEffect] = []

class CapacityUpdate(BaseModel):
    """POST /v1/capacity/signals request"""
    adapter_id: str
    timestamp: float
    signals: CapacitySignals

# ============== JSONL STORAGE ==============

def append_jsonl(filename: str, obj: Dict) -> None:
    """Append JSON object to JSONL file"""
    path = DATA_DIR / f"{filename}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")

def load_jsonl(filename: str, limit: int = 1000) -> List[Dict]:
    """Load recent entries from JSONL"""
    path = DATA_DIR / f"{filename}.jsonl"
    if not path.exists():
        return []
    with open(path, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-limit:]]

# ============== POLICY ENGINE ==============

def evaluate_policy(req: EvaluateRequest) -> EvaluateResponse:
    """
    v0.1 Policy Engine.
    
    Rules (in order):
    1. Denylist match -> BLOCK
    2. High error rate (errors > 10/min) -> DEFER
    3. Everything else -> ALLOW
    """
    tool_name = req.proposal.action_params.tool_name
    signals = req.capacity_signals
    
    # Rule 1: Denylist
    if tool_name in DENYLIST:
        return EvaluateResponse(
            decision_id=f"dec-{uuid.uuid4().hex[:12]}",
            decision="BLOCK",
            confidence=1.0,
            capacity_profile=CapacityProfile(C_obs=0.2),
            justification=f"Tool '{tool_name}' in denylist"
        )
    
    # Rule 2: High error rate (10 errors/min = 0.167/sec)
    if signals.error_rate > 0.167:
        return EvaluateResponse(
            decision_id=f"dec-{uuid.uuid4().hex[:12]}",
            decision="DEFER",
            confidence=0.8,
            capacity_profile=CapacityProfile(C_obs=0.4),
            justification=f"High error rate: {signals.error_rate:.2f}/sec"
        )
    
    # Rule 3: ALLOW
    return EvaluateResponse(
        decision_id=f"dec-{uuid.uuid4().hex[:12]}",
        decision="ALLOW",
        confidence=0.95,
        capacity_profile=CapacityProfile(
            C_geo=0.8, C_int=0.75, C_gauge=0.9, C_ptr=0.7, C_obs=0.85
        ),
        justification="Within capacity limits"
    )

# ============== STATE ==============

registered_adapters: Dict[str, Dict] = {}

# ============== ENDPOINTS ==============

@app.post("/v1/register", response_model=Dict[str, str])
def register_adapter(req: RegisterRequest):
    """
    Register a new adapter.
    Returns adapter_id and registration metadata.
    """
    adapter_id = f"{req.adapter_type}-{uuid.uuid4().hex[:12]}"
    
    record = {
        "adapter_id": adapter_id,
        "adapter_type": req.adapter_type,
        "host_type": req.host_metadata.host_type,
        "namespace": req.host_metadata.namespace,
        "registered_at": datetime.now().isoformat(),
        "capabilities": req.host_metadata.capabilities,
        "version": req.host_metadata.version
    }
    
    registered_adapters[adapter_id] = record
    append_jsonl("adapters", record)
    
    return {
        "adapter_id": adapter_id,
        "registered_at": record["registered_at"],
        "policy_version": "v0.1"
    }

@app.post("/v1/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """
    Evaluate a proposal and return governance decision.
    """
    # Store proposal
    proposal_record = {
        "proposal_id": req.proposal.proposal_id,
        "adapter_id": req.adapter_id,
        "timestamp": req.proposal.timestamp,
        "action_type": req.proposal.action_type,
        "tool_name": req.proposal.action_params.tool_name,
        "risk_tier": req.proposal.risk_tier
    }
    append_jsonl("proposals", proposal_record)
    
    # Run policy
    decision = evaluate_policy(req)
    
    # Store decision
    decision_record = {
        "decision_id": decision.decision_id,
        "proposal_id": req.proposal.proposal_id,
        "adapter_id": req.adapter_id,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "justification": decision.justification,
        "created_at": datetime.now().isoformat()
    }
    append_jsonl("decisions", decision_record)
    
    return decision

@app.post("/v1/outcomes/report")
def report_outcome(outcome: ExecutionOutcome):
    """
    Report execution outcome for audit trail.
    """
    record = {
        **outcome.model_dump(),
        "reported_at": datetime.now().isoformat()
    }
    append_jsonl("outcomes", record)
    
    return {"status": "ok", "outcome_id": hashlib.sha256(
        f"{outcome.proposal_id}:{outcome.executed_at}".encode()
    ).hexdigest()[:16]}

@app.post("/v1/capacity/signals")
def update_capacity_signals(update: CapacityUpdate):
    """
    Receive async capacity signal updates.
    v0.1: Store only, not used in policy.
    """
    record = {
        "adapter_id": update.adapter_id,
        "timestamp": update.timestamp,
        "signals": update.signals.model_dump(),
        "received_at": datetime.now().isoformat()
    }
    append_jsonl("capacity_signals", record)
    
    return {"status": "ok"}

# ============== QUERY ENDPOINTS (Debugging) ==============

@app.get("/v1/adapters/{adapter_id}")
def get_adapter(adapter_id: str):
    """Get adapter info"""
    if adapter_id not in registered_adapters:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return registered_adapters[adapter_id]

@app.get("/v1/stats")
def get_stats():
    """Get aggregate statistics"""
    decisions = load_jsonl("decisions", limit=10000)
    outcomes = load_jsonl("outcomes", limit=10000)
    
    blocked = sum(1 for d in decisions if d.get("decision") == "BLOCK")
    allowed = sum(1 for d in decisions if d.get("decision") == "ALLOW")
    
    return {
        "adapters": len(registered_adapters),
        "proposals_evaluated": len(decisions),
        "blocked": blocked,
        "allowed": allowed,
        "block_rate": blocked / len(decisions) if decisions else 0,
        "outcomes_reported": len(outcomes)
    }

@app.get("/")
def root():
    return {
        "service": "CGF Server v0.1",
        "status": "operational",
        "endpoints": [
            "POST /v1/register",
            "POST /v1/evaluate",
            "POST /v1/outcomes/report",
            "POST /v1/capacity/signals",
            "GET  /v1/adapters/{id}",
            "GET  /v1/stats"
        ]
    }

# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    print(f"CGF Server v0.1 starting...")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Denylist: {DENYLIST}")
    uvicorn.run(app, host="127.0.0.1", port=8080)
