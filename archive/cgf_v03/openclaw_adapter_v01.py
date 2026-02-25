"""
OpenClawAdapter v0.1 - HostAdapter Implementation for OpenClaw

Implements the HostAdapter v1 SPEC contract for OpenClaw tool execution.
Integrates at the tool execution layer (NOT a gateway proxy).

Usage:
    adapter = OpenClawAdapter(
        cgf_endpoint="http://127.0.0.1:8080",
        adapter_type="openclaw",
        host_config={"host_type": "openclaw", "namespace": "main"}
    )
    
    result = adapter.governance_hook(host_context)
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp

# ============== DATA DIRECTORY ==============

DATA_DIR = Path("./openclaw_adapter_data")
DATA_DIR.mkdir(exist_ok=True)


# ============== ENUMS ==============

class Decision(Enum):
    ALLOW = "ALLOW"
    CONSTRAIN = "CONSTRAIN"
    AUDIT = "AUDIT"
    DEFER = "DEFER"
    BLOCK = "BLOCK"


class FailMode(Enum):
    FAIL_CLOSED = "fail_closed"  # Block on CGF down/timeout
    FAIL_OPEN = "fail_open"      # Allow on CGF down/timeout
    DEFER = "defer"              # Defer for review


# ============== EXCEPTIONS ==============

class CGFGovernanceError(Exception):
    """Raised when governance blocks or defers an action"""
    def __init__(self, decision: Decision, justification: str, proposal_id: str):
        self.decision = decision
        self.justification = justification
        self.proposal_id = proposal_id
        super().__init__(f"Governance {decision.value}: {justification}")


class CGFUnavailableError(Exception):
    """Raised when CGF is unreachable or times out"""
    pass


class ConstraintApplicationError(Exception):
    """Raised when constraint cannot be applied"""
    pass


# ============== DATA CLASSES ==============

@dataclass
class ToolParams:
    """Tool call parameters (minimal, generic)"""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_args_hash: str
    side_effects_hint: List[str] = field(default_factory=list)
    idempotent_hint: bool = False
    resource_hints: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class HostProposal:
    """Tool execution proposal"""
    proposal_id: str
    timestamp: float
    action_type: str  # v0.1: only "tool_call"
    action_params: ToolParams
    context_refs: List[str] = field(default_factory=list)
    estimated_cost: Dict[str, float] = field(default_factory=dict)
    risk_tier: str = "medium"


@dataclass
class ExecutionContext:
    """Host execution context"""
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_number: int = 0
    recent_errors: int = 0
    memory_growth_rate: float = 0.0


@dataclass
class CapacitySignals:
    """Real-time capacity signals"""
    token_rate: float = 0.0
    tool_call_rate: float = 0.0
    error_rate: float = 0.0
    memory_growth: float = 0.0


@dataclass
class ExecutionOutcome:
    """Execution outcome for reporting"""
    adapter_id: str
    proposal_id: str
    decision_id: str
    executed: bool
    executed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    result_summary: Optional[str] = None
    actual_cost: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    side_effects: List[Dict[str, str]] = field(default_factory=list)


# ============== MAIN ADAPTER CLASS ==============

class OpenClawAdapter:
    """
    OpenClaw HostAdapter v0.1.
    
    Governance flow: observe → evaluate → enforce → report
    - Intercepts tool calls before execution
    - Calls CGF /v1/evaluate for decision
    - Enforces ALLOW/BLOCK/CONSTRAIN
    - Reports outcomes to /v1/outcomes/report
    
    Fail mode behavior:
    - Side-effect tools: FAIL_CLOSED (block on CGF down)
    - Read-only tools: FAIL_OPEN (allow on CGF down with logging)
    """
    
    def __init__(
        self,
        cgf_endpoint: str,
        adapter_type: str = "openclaw",
        host_config: Optional[Dict] = None,
        timeout_ms: int = 500
    ):
        self.cgf_endpoint = cgf_endpoint.rstrip("/")
        self.adapter_type = adapter_type
        self.host_config = host_config or {}
        self.timeout_ms = timeout_ms / 1000  # Convert to seconds for aiohttp
        
        self.adapter_id: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self.proposals_count = 0
        self.allowed_count = 0
        self.blocked_count = 0
        self.error_count = 0
        
        # Tool categories for fail mode
        self.side_effect_tools = {
            "file_write", "fs_write", "write", "save",
            "exec", "shell", "eval", "code",
            "send", "post", "email", "message"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_ms)
            )
        return self._session
    
    # ============== OBSERVATION METHODS ==============
    
    def observe_proposal(self, host_context: Dict[str, Any]) -> HostProposal:
        """
        Extract proposal from host context (tool execution request).
        
        host_context fields (minimal, generic):
            - tool_name: str
            - tool_args: dict
            - agent_id: str (optional)
            - session_id: str (optional)
            - side_effects_hint: list[str] (optional)
            - idempotent_hint: bool (optional)
        """
        tool_args = host_context.get("tool_args", {})
        tool_args_canonical = json.dumps(tool_args, sort_keys=True, default=str)
        tool_args_hash = hashlib.sha256(tool_args_canonical.encode()).hexdigest()[:32]
        
        # Determine risk tier based on tool category
        tool_name = host_context.get("tool_name", "unknown")
        side_effects = host_context.get("side_effects_hint", [])
        
        if any(t in tool_name for t in self.side_effect_tools) or "write" in side_effects:
            risk_tier = "high"
        elif "read" in side_effects or tool_name in {"read", "cat", "ls", "list"}:
            risk_tier = "low"
        else:
            risk_tier = "medium"
        
        # Resource hints from context
        resource_hints = []
        if "filepath" in tool_args or "path" in tool_args:
            path = tool_args.get("filepath") or tool_args.get("path")
            resource_hints.append({"type": "file", "path": str(path)[:100]})
        
        return HostProposal(
            proposal_id=f"prop-{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            action_type="tool_call",
            action_params=ToolParams(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_args_hash=tool_args_hash,
                side_effects_hint=host_context.get("side_effects_hint", []),
                idempotent_hint=host_context.get("idempotent_hint", False),
                resource_hints=resource_hints,
                metadata={
                    "agent_id": host_context.get("agent_id"),
                    "session_id": host_context.get("session_id")
                }
            ),
            context_refs=[
                host_context.get("agent_id", "unknown"),
                host_context.get("session_id", "unknown")
            ],
            estimated_cost={
                "tokens": len(tool_args_canonical) // 4,
                "latency_ms": 500
            },
            risk_tier=risk_tier
        )
    
    def observe_context(self, host_context: Dict[str, Any]) -> ExecutionContext:
        """Extract minimal execution context"""
        return ExecutionContext(
            agent_id=host_context.get("agent_id"),
            session_id=host_context.get("session_id"),
            turn_number=host_context.get("turn_number", 0),
            recent_errors=self.error_count,
            memory_growth_rate=host_context.get("memory_growth_rate", 0.0)
        )
    
    def observe_capacity_signals(self, host_context: Dict[str, Any]) -> CapacitySignals:
        """Extract capacity signals (minimal v0.1)"""
        time_window = 60.0
        return CapacitySignals(
            token_rate=host_context.get("token_count", 0) / time_window,
            tool_call_rate=self.proposals_count / time_window if time_window > 0 else 0,
            error_rate=self.error_count / time_window if time_window > 0 else 0,
            memory_growth=host_context.get("memory_growth", 0.0)
        )
    
    # ============== REGISTRATION ==============
    
    async def register(self) -> str:
        """Register adapter with CGF"""
        session = await self._get_session()
        
        payload = {
            "adapter_type": self.adapter_type,
            "host_metadata": {
                "host_type": self.host_config.get("host_type", "openclaw"),
                "runtime": self.host_config.get("runtime"),
                "namespace": self.host_config.get("namespace", "default"),
                "capabilities": self.host_config.get("capabilities", ["tool_call"]),
                "version": "0.1.0"
            }
        }
        
        try:
            async with session.post(
                f"{self.cgf_endpoint}/v1/register",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.adapter_id = data["adapter_id"]
                    self.emit_event("adapter_registered", {
                        "adapter_id": self.adapter_id,
                        "cgf_endpoint": self.cgf_endpoint
                    })
                    return self.adapter_id
                else:
                    raise CGFUnavailableError(f"Registration failed: {resp.status}")
        except Exception as e:
            raise CGFUnavailableError(f"Registration error: {e}")
    
    # ============== EVALUATION ==============
    
    async def _evaluate(self, proposal: HostProposal, context: ExecutionContext, signals: CapacitySignals) -> Decision:
        """Call CGF /v1/evaluate"""
        session = await self._get_session()
        
        payload = {
            "adapter_id": self.adapter_id,
            "host_config": self.host_config,
            "proposal": {
                "proposal_id": proposal.proposal_id,
                "timestamp": proposal.timestamp,
                "action_type": proposal.action_type,
                "action_params": {
                    "tool_name": proposal.action_params.tool_name,
                    "tool_args_hash": proposal.action_params.tool_args_hash,
                    "side_effects_hint": proposal.action_params.side_effects_hint,
                    "idempotent_hint": proposal.action_params.idempotent_hint,
                    "resource_hints": proposal.action_params.resource_hints
                },
                "context_refs": proposal.context_refs,
                "estimated_cost": proposal.estimated_cost,
                "risk_tier": proposal.risk_tier
            },
            "context": {
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "turn_number": context.turn_number,
                "recent_errors": context.recent_errors,
                "memory_growth_rate": context.memory_growth_rate
            },
            "capacity_signals": {
                "token_rate": signals.token_rate,
                "tool_call_rate": signals.tool_call_rate,
                "error_rate": signals.error_rate,
                "memory_growth": signals.memory_growth
            }
        }
        
        self.emit_event("proposal_received", {
            "proposal_id": proposal.proposal_id,
            "tool_name": proposal.action_params.tool_name,
            "risk_tier": proposal.risk_tier
        })
        
        try:
            async with session.post(
                f"{self.cgf_endpoint}/v1/evaluate",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.emit_event("decision_made", {
                        "proposal_id": proposal.proposal_id,
                        "decision": data["decision"],
                        "confidence": data["confidence"]
                    })
                    return Decision(data["decision"])
                else:
                    text = await resp.text()
                    raise CGFUnavailableError(f"Evaluate failed: {resp.status} - {text}")
        except asyncio.TimeoutError:
            raise CGFUnavailableError("Timeout calling CGF")
        except aiohttp.ClientError as e:
            raise CGFUnavailableError(f"CGF unreachable: {e}")
    
    def _apply_fail_mode(self, risk_tier: str, tool_name: str) -> Decision:
        """
        Apply fail mode when CGF is unavailable.
        
        Side-effect tools -> BLOCK (fail closed)
        Read-only tools -> ALLOW (fail open)
        """
        has_side_effects = (
            tool_name in self.side_effect_tools or
            any(w in tool_name for w in ["write", "save", "exec", "send"])
        )
        
        if has_side_effects or risk_tier == "high":
            # FAIL_CLOSED: Block side-effect tools when CGF down
            return Decision.BLOCK
        else:
            # FAIL_OPEN: Allow read-only tools when CGF down
            return Decision.ALLOW
    
    # ============== ENFORCEMENT ==============
    
    async def enforce_allow(
        self,
        proposal: HostProposal,
        decision_id: str
    ) -> Dict[str, Any]:
        """
        Execute tool normally.
        Returns context for actual tool execution.
        """
        self.emit_event("action_allowed", {
            "proposal_id": proposal.proposal_id,
            "decision_id": decision_id,
            "tool_name": proposal.action_params.tool_name
        })
        
        return {
            "executed": True,
            "tool_name": proposal.action_params.tool_name,
            "tool_args": proposal.action_params.tool_args,
            "decision_id": decision_id,
            "proposal_id": proposal.proposal_id
        }
    
    async def enforce_block(
        self,
        proposal: HostProposal,
        decision_id: str,
        justification: str
    ) -> Dict[str, Any]:
        """
        Block tool execution.
        Raises CGFGovernanceError.
        """
        self.emit_event("action_blocked", {
            "proposal_id": proposal.proposal_id,
            "decision_id": decision_id,
            "tool_name": proposal.action_params.tool_name,
            "justification": justification
        })
        self.blocked_count += 1
        
        raise CGFGovernanceError(
            decision=Decision.BLOCK,
            justification=justification,
            proposal_id=proposal.proposal_id
        )
    
    async def enforce_constrain(
        self,
        proposal: HostProposal,
        decision_id: str,
        constraint: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Apply constraint to tool execution.
        v0.1 constraints:
        - denied_args: remove args keys
        - timeout_override_ms: set timeout
        
        If constraint cannot be applied, FAIL_CLOSED and treat as BLOCK.
        """
        try:
            modified_args = dict(proposal.action_params.tool_args)
            
            if constraint:
                # Remove denied args
                denied = constraint.get("denied_args", [])
                for key in denied:
                    modified_args.pop(key, None)
                
                # Apply modified args if specified
                if constraint.get("modified_args"):
                    modified_args.update(constraint["modified_args"])
                
                self.emit_event("constraint_applied", {
                    "proposal_id": proposal.proposal_id,
                    "decision_id": decision_id,
                    "denied_args": denied,
                    "reason": constraint.get("reason", "")
                })
            
            return {
                "executed": True,
                "tool_name": proposal.action_params.tool_name,
                "tool_args": modified_args,
                "constrained": True,
                "decision_id": decision_id,
                "proposal_id": proposal.proposal_id
            }
            
        except Exception as e:
            # ENFORCEMENT INVARIANT: Constraint fails -> BLOCK
            self.emit_event("constraint_failed", {
                "proposal_id": proposal.proposal_id,
                "decision_id": decision_id,
                "error": str(e),
                "fallback": "BLOCK"
            })
            raise CGFGovernanceError(
                decision=Decision.BLOCK,
                justification=f"Constraint application failed: {e}",
                proposal_id=proposal.proposal_id
            )
    
    # ============== REPORTING ==============
    
    def observe_execution(
        self,
        host_result: Dict[str, Any],
        proposal: HostProposal
    ) -> ExecutionOutcome:
        """Extract outcome from tool execution result"""
        return ExecutionOutcome(
            adapter_id=self.adapter_id or "unknown",
            proposal_id=proposal.proposal_id,
            decision_id=host_result.get("decision_id", "unknown"),
            executed=host_result.get("executed", False),
            executed_at=host_result.get("executed_at"),
            duration_ms=host_result.get("duration_ms"),
            success=host_result.get("success"),
            result_summary=host_result.get("result_summary", "")[:200],
            actual_cost=host_result.get("actual_cost", {}),
            errors=[host_result["error"]] if host_result.get("error") else [],
            side_effects=host_result.get("side_effects", [])
        )
    
    async def report(self, outcome: ExecutionOutcome) -> None:
        """
        Report execution outcome to CGF (async, fire-and-forget).
        On failure, write to local JSONL for later recovery.
        """
        session = await self._get_session()
        
        payload = {
            "adapter_id": outcome.adapter_id,
            "proposal_id": outcome.proposal_id,
            "decision_id": outcome.decision_id,
            "executed": outcome.executed,
            "executed_at": outcome.executed_at,
            "duration_ms": outcome.duration_ms,
            "success": outcome.success,
            "result_summary": outcome.result_summary,
            "actual_cost": outcome.actual_cost,
            "errors": outcome.errors,
            "side_effects": outcome.side_effects
        }
        
        try:
            async with session.post(
                f"{self.cgf_endpoint}/v1/outcomes/report",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5.0)  # Longer timeout for reporting
            ) as resp:
                self.emit_event("outcome_logged", {
                    "proposal_id": outcome.proposal_id,
                    "executed": outcome.executed,
                    "success": outcome.success
                })
        except Exception as e:
            # Fire-and-forget failed: write to local JSONL
            self.emit_event("outcome_report_failed", {
                "proposal_id": outcome.proposal_id,
                "error": str(e)
            })
            self._write_local_outcome(payload, e)
    
    def _write_local_outcome(self, payload: Dict, error: Exception) -> None:
        """Write outcome to local JSONL when CGF report fails"""
        record = {
            **payload,
            "report_failed_at": time.time(),
            "report_error": str(error),
            "pending_retry": True
        }
        path = DATA_DIR / "outcomes_local.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def emit_event(self, event_type: str, payload: Dict) -> None:
        """Emit event to local JSONL (control plane later)"""
        event = {
            "event_type": event_type,
            "adapter_id": self.adapter_id,
            "timestamp": time.time(),
            "payload": payload
        }
        path = DATA_DIR / "events.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    # ============== MAIN HOOK ==============
    
    async def governance_hook(self, host_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main governance flow: observe → evaluate → enforce
        
        Args:
            host_context: Tool call context with tool_name, tool_args, etc.
            
        Returns:
            Enforcement result (tool args to execute if allowed)
            
        Raises:
            CGFGovernanceError: If tool is blocked/deferred
            CGFUnavailableError: If CGF unreachable (handled via fail mode)
        """
        self.proposals_count += 1
        
        # 1. OBSERVE
        proposal = self.observe_proposal(host_context)
        context = self.observe_context(host_context)
        signals = self.observe_capacity_signals(host_context)
        
        # Ensure registered
        if self.adapter_id is None:
            await self.register()
        
        # 2. EVALUATE (with fail mode on error)
        decision: Decision
        try:
            decision = await self._evaluate(proposal, context, signals)
        except CGFUnavailableError as e:
            # Apply fail mode
            decision = self._apply_fail_mode(proposal.risk_tier, proposal.action_params.tool_name)
            self.emit_event("cgf_unreachable_fallback", {
                "proposal_id": proposal.proposal_id,
                "error": str(e),
                "fail_mode_decision": decision.value
            })
        
        # 3. ENFORCE
        if decision == Decision.ALLOW:
            return await self.enforce_allow(proposal, "allow-direct")
        
        elif decision == Decision.BLOCK:
            return await self.enforce_block(
                proposal, "block-direct", "Blocked by CGF policy"
            )
        
        elif decision == Decision.CONSTRAIN:
            # v0.1: Apply constraint (if implementable) else fail closed
            constraint = {}  # Would come from CGF response
            return await self.enforce_constrain(proposal, "constrain-direct", constraint)
        
        elif decision == Decision.DEFER:
            self.emit_event("action_deferred", {
                "proposal_id": proposal.proposal_id,
                "reason": "DEFER decision from CGF"
            })
            raise CGFGovernanceError(
                decision=Decision.DEFER,
                justification="Action deferred for review",
                proposal_id=proposal.proposal_id
            )
        
        elif decision == Decision.AUDIT:
            # v0.1: Treat AUDIT as ALLOW with audit flag
            result = await self.enforce_allow(proposal, "audit-direct")
            result["audit_required"] = True
            return result
        
        else:
            # Unknown decision: fail closed
            return await self.enforce_block(
                proposal, "unknown-decision", f"Unknown decision: {decision}"
            )
    
    async def report_execution(
        self,
        host_result: Dict[str, Any],
        host_context: Dict[str, Any]
    ) -> None:
        """
        Call after tool execution to report outcome.
        Separated from governance_hook to allow async execution.
        """
        proposal = self.observe_proposal(host_context)
        outcome = self.observe_execution(host_result, proposal)
        await self.report(outcome)
    
    async def close(self) -> None:
        """Close adapter and emit disconnect event"""
        self.emit_event("adapter_disconnected", {
            "adapter_id": self.adapter_id,
            "proposals_total": self.proposals_count,
            "blocked": self.blocked_count
        })
        if self._session and not self._session.closed:
            await self._session.close()


# ============== SYNC WRAPPER FOR EASIER INTEGRATION ==============

class OpenClawAdapterSync:
    """
    Synchronous wrapper for OpenClawAdapter.
    Used when integrating into non-async OpenClaw code.
    """
    
    def __init__(self, *args, **kwargs):
        self._adapter = OpenClawAdapter(*args, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def governance_hook(self, host_context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous governance hook"""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._adapter.governance_hook(host_context)
        )
    
    def report_execution(
        self,
        host_result: Dict[str, Any],
        host_context: Dict[str, Any]
    ) -> None:
        """Synchronous outcome reporting"""
        loop = self._get_loop()
        # Fire-and-forget; don't block on report
        asyncio.ensure_future(
            self._adapter.report_execution(host_result, host_context),
            loop=loop
        )
    
    def close(self) -> None:
        """Close adapter"""
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._adapter.close())
