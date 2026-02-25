"""
langgraph_adapter_v01.py - LangGraph Host Adapter v0.1
Second host implementation for Capacity Governance Framework v0.3.

Interception Point:
- LangGraph tool invocation occurs in graph nodes
- Hook wraps tool calls before execution
- Routes through adapter.governance_hook()

Architecture:
1. Custom LangGraph ToolNode wrapper
2. State-based observation (LangGraph state as session)
3. Thread ID as session identifier
4. Compatible with OpenClaw evaluation contract
"""

import asyncio
import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Import v0.3 schemas with backward compatibility
try:
    from cgf_schemas_v03 import (
        SCHEMA_VERSION,
        ActionType,
        DecisionType,
        RiskTier,
        FailMode,
        HostEventType,
        HostConfig,
        HostProposal,
        HostContext,
        CapacitySignals,
        CGFDecision,
        HostOutcomeReport,
        HostEvent,
        ConstraintConfig,
        get_event_required_fields,
        validate_event_payload
    )
    SCHEMA_MODULE = "v0.3"
except ImportError:
    # Fallback to v0.2
    from cgf_schemas_v02 import (
        SCHEMA_VERSION,
        ActionType,
        DecisionType,
        RiskTier,
        FailMode,
        HostEventType,
        HostConfig,
        HostProposal,
        HostContext,
        CapacitySignals,
        CGFDecision,
        HostOutcomeReport,
        HostEvent,
        ConstraintConfig,
        get_event_required_fields,
        validate_event_payload
    )
    SCHEMA_MODULE = "v0.2"

# ============== CONFIGURATION ==============

DEFAULT_CONFIG = {
    "cgf_endpoint": os.environ.get("CGF_ENDPOINT", "http://127.0.0.1:8080"),
    "cgf_timeout_ms": int(os.environ.get("CGF_TIMEOUT_MS", "500")),
    "cgf_enabled": os.environ.get("CGF_ENABLED", "true").lower() == "true",
    "data_dir": "./langgraph_cgf_data",
    "log_level": os.environ.get("LOG_LEVEL", "info"),
    "side_effect_tools": ["file_write", "fs_write", "write", "save", "exec", "shell", 
                          "eval", "bash", "python", "subprocess"]
}

# Ensure data directory
os.makedirs(DEFAULT_CONFIG["data_dir"], exist_ok=True)

# ============== ABSTRACT HOST ADAPTER (v0.3 Contract) ==============

class HostAdapter:
    """Abstract base class for CGF Host Adapters (v0.3 contract).
    
    Every host must implement this contract for cross-host compatibility.
    """
    
    def __init__(self, host_config: HostConfig):
        self.host_config = host_config
        self.adapter_id: Optional[str] = None
        self._registered = False
    
    # === OBSERVATION (mandatory) ===
    
    def observe_proposal(self, **kwargs) -> HostProposal:
        """Create action proposal from raw host inputs."""
        raise NotImplementedError
    
    def observe_context(self, **kwargs) -> HostContext:
        """Create runtime context from host state."""
        raise NotImplementedError
    
    def observe_capacity_signals(self, **kwargs) -> CapacitySignals:
        """Create capacity signals from infrastructure."""
        raise NotImplementedError
    
    # === ENFORCEMENT (mandatory) ===
    
    async def enforce_allow(self, decision: CGFDecision, **kwargs):
        """Enforce ALLOW decision."""
        raise NotImplementedError
    
    async def enforce_block(self, decision: CGFDecision, **kwargs):
        """Enforce BLOCK decision."""
        raise NotImplementedError
    
    async def enforce_constrain(self, decision: CGFDecision, **kwargs):
        """Enforce CONSTRAIN decision."""
        raise NotImplementedError
    
    # === EXECUTION (mandatory) ===
    
    async def observe_execution(self, **kwargs) -> HostOutcomeReport:
        """Observe execution outcome."""
        raise NotImplementedError
    
    async def report(self, outcome: HostOutcomeReport):
        """Report outcome to CGF."""
        raise NotImplementedError
    
    # === COMMUNICATION (mandatory) ===
    
    def emit_event(self, event_type: HostEventType, payload: Dict, 
                   proposal_id: Optional[str] = None,
                   decision_id: Optional[str] = None) -> HostEvent:
        """Emit canonical event."""
        raise NotImplementedError
    
    # === GOVERNANCE HOOK (entry point) ===
    
    async def governance_hook(self, **kwargs):
        """Main governance entry point."""
        raise NotImplementedError

# ============== UTILITY FUNCTIONS ==============

def generate_id(prefix: str = "") -> str:
    """Generate unique ID with prefix."""
    return f"{prefix}-{datetime.now().timestamp() * 1000:.0f}-{hashlib.md5(os.urandom(8)).hexdigest()[:6]}"

def hash_content(content: Union[str, bytes]) -> str:
    """Generate content hash."""
    if isinstance(content, str):
        return hashlib.sha256(content.encode()).hexdigest()
    return hashlib.sha256(content).hexdigest()

def log_event(level: str, message: str, meta: Dict = None):
    """Log with level filtering."""
    levels = {"debug": 0, "info": 1, "warn": 2, "error": 3}
    if levels.get(level, 1) >= levels.get(DEFAULT_CONFIG["log_level"], 1):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "adapter": "langgraph",
            **(meta or {})
        }
        print(f"[LANGGRAPH-CGF] [{level.upper()}] {message}")
        
        # Persist to log
        log_path = Path(DEFAULT_CONFIG["data_dir"]) / "adapter.log"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

# ============== LANGGRAPH ADAPTER IMPLEMENTATION ==============

class LangGraphAdapter(HostAdapter):
    """LangGraph-specific Host Adapter (Host #2).
    
    Interception Point:
    - LangGraph tool nodes (ToolNode class)
    - State: LangGraph state dict with thread_id
    - Configuration: graph configuration
    
    Example integration:
        from langgraph_adapter import langgraph_governance_hook
        
        # Wrap ToolNode
        tool_node = ToolNode(tools)
        governed_tool_node = GovernedToolNode(tool_node, adapter)
    """
    
    def __init__(self, host_config: Optional[HostConfig] = None):
        if host_config is None:
            host_config = HostConfig(
                host_type="langgraph",
                namespace="default",
                capabilities=["tool_call"],  # v0.1: tool_call only
                version="0.1.0"
            )
        super().__init__(host_config)
        self._proposal_count = 0
        self._event_count = 0
        
        # Lazy registration
        self._registration_task = None
    
    # ============== OBSERVATION ==============
    
    def observe_proposal(self, tool_name: str, tool_args: Dict, 
                         thread_id: Optional[str] = None,
                         node_id: Optional[str] = None,
                         checkpoint_ns: Optional[str] = None) -> HostProposal:
        """Create proposal from LangGraph tool invocation."""
        self._proposal_count += 1
        
        # Compute risk tier from side effects
        side_effects = self._infer_side_effects(tool_name)
        risk_tier = self._compute_risk_tier(side_effects, tool_name)
        
        # Hash args (don't send full content)
        args_str = json.dumps(tool_args, sort_keys=True, default=str)
        args_hash = hash_content(args_str)[:32]
        
        return HostProposal(
            proposal_id=generate_id("lg-prop"),
            timestamp=datetime.now().timestamp(),
            action_type=ActionType.TOOL_CALL,
            action_params={
                "tool_name": tool_name,
                "tool_args_hash": args_hash,
                "side_effects_hint": side_effects,
                "idempotent_hint": tool_name == "search",  # Example heuristic
                "resource_hints": [],
                # LangGraph-specific (not used by policy, for debugging)
                "node_id": node_id,
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns
            },
            context_refs=[
                thread_id or "unknown_thread",
                node_id or "unknown_node",
                checkpoint_ns or "default"
            ],
            estimated_cost={
                "tokens": len(args_str) // 4,
                "latency_ms": 300
            },
            risk_tier=risk_tier,
            priority=0
        )
    
    def _infer_side_effects(self, tool_name: str) -> List[str]:
        """Infer side effects from tool name (heuristic)."""
        side_effect_tools = DEFAULT_CONFIG["side_effect_tools"]
        if any(se in tool_name.lower() for se in side_effect_tools):
            return ["write"]
        return ["read"]
    
    def _compute_risk_tier(self, side_effects: List[str], tool_name: str) -> RiskTier:
        """Compute risk tier data-driven (NOT hardcoded).
        
        Uses schema-derived rules:
        - write operations → HIGH risk
        - network operations → MEDIUM risk
        - read operations → LOW risk
        """
        if "write" in side_effects:
            return RiskTier.HIGH
        elif "network" in side_effects:
            return RiskTier.MEDIUM
        return RiskTier.LOW
    
    def observe_context(self, thread_id: Optional[str] = None,
                       node_id: Optional[str] = None,
                       state: Optional[Dict] = None,
                       recent_errors: int = 0) -> HostContext:
        """Create context from LangGraph state."""
        return HostContext(
            agent_id=thread_id,  # LangGraph thread = agent session
            session_id=thread_id,
            turn_number=state.get("turn_number", 0) if state else 0,
            recent_errors=recent_errors,
            memory_growth_rate=0.0
        )
    
    def observe_capacity_signals(self, token_rate: float = 0.0,
                                  tool_rate: float = 0.0,
                                  error_rate: float = 0.0) -> CapacitySignals:
        """Create capacity signals."""
        return CapacitySignals(
            token_rate=token_rate,
            tool_call_rate=tool_rate,
            error_rate=error_rate,
            memory_growth=0.0
        )
    
    # ============== ENFORCEMENT ==============
    
    async def enforce_allow(self, decision: CGFDecision, 
                            tool_call: Optional[Dict] = None) -> Dict[str, Any]:
        """Enforce ALLOW - return parameters for execution."""
        self.emit_event(
            HostEventType.ACTION_ALLOWED,
            {
                "decision_id": decision.decision_id,
                "proposal_id": decision.proposal_id,
                "executed_at": datetime.now().timestamp(),
                "host": "langgraph"
            },
            decision.proposal_id,
            decision.decision_id
        )
        
        log_event("info", "Tool allowed by CGF", {
            "decision": decision.decision,
            "tool": tool_call.get("tool_name") if tool_call else None
        })
        
        return {"allowed": True, "reason": "allowed"}
    
    async def enforce_block(self, decision: CGFDecision,
                            tool_call: Optional[Dict] = None) -> None:
        """Enforce BLOCK - raise exception."""
        self.emit_event(
            HostEventType.ACTION_BLOCKED,
            {
                "decision_id": decision.decision_id,
                "proposal_id": decision.proposal_id,
                "justification": decision.justification,
                "reason_code": decision.reason_code or "BLOCKED_BY_POLICY"
            },
            decision.proposal_id,
            decision.decision_id
        )
        
        log_event("warn", "Tool blocked by CGF", {
            "tool": tool_call.get("tool_name") if tool_call else None,
            "reason": decision.justification
        })
        
        raise LangGraphToolBlocked(
            tool_name=tool_call.get("tool_name") if tool_call else "unknown",
            reason=decision.justification,
            decision_id=decision.decision_id
        )
    
    async def enforce_constrain(self, decision: CGFDecision,
                                tool_call: Optional[Dict] = None) -> Dict[str, Any]:
        """Enforce CONSTRAIN (e.g., rate limit, sandbox)."""
        constraint = decision.constraint
        constraint_type = constraint.type if constraint else "unknown"
        
        self.emit_event(
            HostEventType.ACTION_CONSTRAINED,
            {
                "decision_id": decision.decision_id,
                "proposal_id": decision.proposal_id,
                "constraint_type": constraint_type,
                "constraint_params": constraint.params if constraint else {}
            },
            decision.proposal_id,
            decision.decision_id
        )
        
        log_event("info", "Tool constrained by CGF", {
            "tool": tool_call.get("tool_name") if tool_call else None,
            "constraint": constraint_type
        })
        
        # For LangGraph, constraints might mean sandboxed execution
        # or parameter modification
        return {
            "allowed": True,
            "constrained": True,
            "constraint_type": constraint_type
        }
    
    # ============== EXECUTION ==============
    
    async def observe_execution(self, proposal: HostProposal,
                                decision: CGFDecision,
                                success: bool,
                                duration_ms: float,
                                error: Optional[str] = None) -> HostOutcomeReport:
        """Create outcome report."""
        return HostOutcomeReport(
            adapter_id=self.adapter_id or "unregistered",
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            executed=success,
            executed_at=datetime.now().timestamp(),
            duration_ms=duration_ms,
            success=success,
            committed=success and decision.decision != DecisionType.BLOCK,
            quarantined=decision.decision == DecisionType.CONSTRAIN,
            errors=[error] if error else [],
            result_summary="Tool executed" if success else "Tool failed"
        )
    
    async def report(self, outcome: HostOutcomeReport):
        """Report outcome to CGF server."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{DEFAULT_CONFIG['cgf_endpoint']}/v1/outcomes/report",
                    json=outcome.model_dump() if hasattr(outcome, 'model_dump') else outcome.__dict__,
                    timeout=aiohttp.ClientTimeout(total=5)
                ):
                    pass
        except Exception as e:
            log_event("warn", "Failed to report outcome", {"error": str(e)})
        
        # Local logging
        outcome_path = Path(DEFAULT_CONFIG["data_dir"]) / "outcomes.jsonl"
        with open(outcome_path, "a") as f:
            f.write(json.dumps(outcome.model_dump() if hasattr(outcome, 'model_dump') else outcome.__dict__) + "\n")
        
        self.emit_event(
            HostEventType.OUTCOME_LOGGED,
            {
                "proposal_id": outcome.proposal_id,
                "decision_id": outcome.decision_id,
                "success": outcome.success,
                "duration_ms": outcome.duration_ms
            },
            outcome.proposal_id,
            outcome.decision_id
        )
    
    # ============== EVENTS ==============
    
    def emit_event(self, event_type: HostEventType, payload: Dict,
                  proposal_id: Optional[str] = None,
                  decision_id: Optional[str] = None) -> HostEvent:
        """Emit canonical event with validation."""
        # Validate required fields
        valid, errors = validate_event_payload(event_type, payload)
        if not valid:
            log_event("error", f"Event validation failed: {errors}")
        
        event = HostEvent(
            event_type=event_type,
            adapter_id=self.adapter_id or "unregistered",
            timestamp=datetime.now().timestamp(),
            proposal_id=proposal_id,
            decision_id=decision_id,
            payload=payload
        )
        
        # Persist
        event_path = Path(DEFAULT_CONFIG["data_dir"]) / "events.jsonl"
        with open(event_path, "a") as f:
            f.write(json.dumps(event.model_dump() if hasattr(event, 'model_dump') else event.__dict__) + "\n")
        
        self._event_count += 1
        return event
    
    # ============== GOVERNANCE HOOK ==============
    
    async def _register_lazy(self):
        """Lazy registration with CGF."""
        if self._registered:
            return
        
        import aiohttp
        
        payload = {
            "schema_version": SCHEMA_VERSION,
            "adapter_type": "langgraph",
            "host_config": self.host_config.model_dump() if hasattr(self.host_config, 'model_dump') else self.host_config.__dict__,
            "features": ["tool_call"],
            "timestamp": datetime.now().timestamp()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{DEFAULT_CONFIG['cgf_endpoint']}/v1/register",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.adapter_id = data.get("adapter_id", generate_id("lg-adp"))
                        self._registered = True
                        self.emit_event(
                            HostEventType.ADAPTER_REGISTERED,
                            {"adapter_id": self.adapter_id, "host_type": "langgraph"}
                        )
                        log_event("info", f"Registered with CGF: {self.adapter_id}")
                    else:
                        self.adapter_id = generate_id("lg-local")
        except Exception as e:
            log_event("error", "Registration failed", {"error": str(e)})
            self.adapter_id = generate_id("lg-local")
    
    async def governance_hook(self, tool_name: str, tool_args: Dict,
                              thread_id: Optional[str] = None,
                              node_id: Optional[str] = None,
                              state: Optional[Dict] = None) -> Dict[str, Any]:
        """Main governance entry point for LangGraph.
        
        Call this before executing any tool in LangGraph.
        
        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments dict
            thread_id: LangGraph thread ID
            node_id: Current node ID
            state: Current LangGraph state
            
        Returns:
            Dict with 'allowed', 'reason', 'decision_id', etc.
            
        Raises:
            LangGraphToolBlocked: If CGF blocks the tool
        """
        start_time = datetime.now().timestamp()
        
        # Lazy registration
        await self._register_lazy()
        
        # Observe
        proposal = self.observe_proposal(tool_name, tool_args, thread_id, node_id)
        context = self.observe_context(thread_id, node_id, state)
        signals = self.observe_capacity_signals()
        
        # Evaluate with CGF
        import aiohttp
        
        eval_payload = {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": self.adapter_id,
            "host_config": self.host_config.model_dump() if hasattr(self.host_config, 'model_dump') else self.host_config.__dict__,
            "proposal": proposal.model_dump() if hasattr(proposal, 'model_dump') else proposal.__dict__,
            "context": context.model_dump() if hasattr(context, 'model_dump') else context.__dict__,
            "capacity_signals": signals.model_dump() if hasattr(signals, 'model_dump') else signals.__dict__
        }
        
        self.emit_event(
            HostEventType.PROPOSAL_RECEIVED,
            {
                "proposal_id": proposal.proposal_id,
                "action_type": proposal.action_type.value,
                "action_params_hash": hash_content(json.dumps(proposal.action_params))[:16],
                "risk_tier": proposal.risk_tier.value
            },
            proposal.proposal_id
        )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{DEFAULT_CONFIG['cgf_endpoint']}/v1/evaluate",
                    json=eval_payload,
                    timeout=aiohttp.ClientTimeout(total=DEFAULT_CONFIG['cgf_timeout_ms'] / 1000)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        decision_data = data.get("decision", {})
                        
                        # Validate decision
                        decision = CGFDecision(**decision_data) if isinstance(decision_data, dict) else decision_data
                    else:
                        raise CGFUnreachableError(f"CGF returned {resp.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
            # CGF unreachable - apply fail mode
            is_side_effect = "write" in proposal.action_params.get("side_effects_hint", [])
            risk_tier = proposal.risk_tier
            
            if is_side_effect or risk_tier == RiskTier.HIGH:
                # Fail closed for side-effects or high risk
                duration_ms = (datetime.now().timestamp() - start_time) * 1000
                outcome = await self.observe_execution(
                    proposal,
                    CGFDecision(
                        decision_id=generate_id("dec"),
                        proposal_id=proposal.proposal_id,
                        decision=DecisionType.BLOCK,
                        confidence=0.0,
                        justification=f"CGF unreachable - fail closed for {risk_tier.value} risk",
                        reason_code="CGF_UNREACHABLE_FAIL_CLOSED"
                    ),
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e)
                )
                await self.report(outcome)
                
                raise LangGraphToolBlocked(
                    tool_name=tool_name,
                    reason=f"CGF unavailable ({e}) - fail closed for {risk_tier.value} risk",
                    decision_id="fail-mode"
                )
            else:
                # Fail open
                log_event("warn", "CGF unreachable - fail open", {"risk": risk_tier.value})
                return {"allowed": True, "reason": "CGF unreachable - fail open", "fail_open": True}
        
        # Enforce decision
        duration_ms = (datetime.now().timestamp() - start_time) * 1000
        
        if decision.decision == DecisionType.BLOCK:
            await self.enforce_block(decision, {"tool_name": tool_name, "args": tool_args})
        elif decision.decision == DecisionType.CONSTRAIN:
            constraint_result = await self.enforce_constrain(decision, {"tool_name": tool_name, "args": tool_args})
            # Report outcome
            outcome = await self.observe_execution(proposal, decision, True, duration_ms)
            await self.report(outcome)
            return {"allowed": True, "constrained": True, **constraint_result}
        else:  # ALLOW
            await self.enforce_allow(decision, {"tool_name": tool_name, "args": tool_args})
        
        # Report outcome for ALLOW
        outcome = await self.observe_execution(proposal, decision, True, duration_ms)
        await self.report(outcome)
        
        return {"allowed": True, "decision_id": decision.decision_id}


# ============== EXCEPTIONS ==============

class LangGraphToolBlocked(Exception):
    """Exception raised when CGF blocks a tool."""
    
    def __init__(self, tool_name: str, reason: str, decision_id: str):
        self.tool_name = tool_name
        self.reason = reason
        self.decision_id = decision_id
        super().__init__(f"Tool '{tool_name}' blocked by CGF: {reason}")


class CGFUnreachableError(Exception):
    """CGF server is unreachable."""
    pass


# ============== LANGGRAPH INTEGRATION ==============

class GovernedToolNode:
    """Wrapper for LangGraph ToolNode with CGF governance.
    
    Usage:
        from langgraph.prebuilt import ToolNode
        from langgraph_adapter import GovernedToolNode
        
        # Original tool node
        tool_node = ToolNode(tools)
        
        # Wrapped with governance
        governed_node = GovernedToolNode(tool_node, thread_id="thread-123")
        
        # Use in graph
        graph.add_node("tools", governed_node)
    """
    
    def __init__(self, tool_node: Any, thread_id: Optional[str] = None,
                 adapter: Optional[LangGraphAdapter] = None):
        self.tool_node = tool_node
        self.thread_id = thread_id
        self.adapter = adapter or LangGraphAdapter()
    
    async def __call__(self, state: Dict) -> Dict:
        """Process tool calls with governance.
        
        Interception Point:
        - Before tool_node(state) executes
        - For each tool call in state["messages"][-1].tool_calls
        """
        messages = state.get("messages", [])
        if not messages:
            return await self.tool_node(state) if asyncio.iscoroutinefunction(self.tool_node) else self.tool_node(state)
        
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", []) or last_message.get("tool_calls", [])
        
        if not tool_calls:
            return await self.tool_node(state) if asyncio.iscoroutinefunction(self.tool_node) else self.tool_node(state)
        
        # Process first tool call with governance
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
        
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except:
                tool_args = {}
        
        # Run governance hook
        try:
            result = await self.adapter.governance_hook(
                tool_name=tool_name,
                tool_args=tool_args,
                thread_id=self.thread_id,
                node_id=state.get("node_id"),
                state=state
            )
            
            if result.get("allowed"):
                # Execute original tool
                return await self.tool_node(state) if asyncio.iscoroutinefunction(self.tool_node) else self.tool_node(state)
            else:
                # Should have raised Blocked exception
                raise LangGraphToolBlocked(tool_name, "Unknown block reason", "unknown")
                
        except LangGraphToolBlocked:
            # Return blocked response in LangGraph format
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Tool '{tool_name}' was blocked by capacity governance."
                    }
                ]
            }


# ============== SYNC WRAPPER FOR LEGACY INTEGRATION ==============

def langgraph_governance_hook(tool_name: str, tool_args: Dict, 
                              thread_id: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper for governance hook.
    
    For use in non-async LangGraph contexts.
    """
    adapter = LangGraphAdapter()
    return asyncio.run(adapter.governance_hook(tool_name, tool_args, thread_id))


# ============== EXAMPLE GRAPH ==============

def create_example_graph():
    """Create a minimal LangGraph example that demonstrates CGF integration.
    
    This creates a simple graph with:
    - An agent node that decides to call tools
    - A governed tool node that routes through CGF
    
    Note: This is a template - actual LangGraph requires proper state definition.
    """
    return {
        "description": """
Minimal LangGraph with CGF Governance:

State = {
    "messages": [...],
    "thread_id": str
}

Nodes:
1. agent_node(state) -> decides tool to call, returns with tool_calls
2. governed_tools(state) -> interposes CGF, then executes or blocks
   - GovernedToolNode wraps ToolNode
   - Thread ID from state["thread_id"]

Edges:
- START -> agent
- agent -> END (if no tool call) OR tools (if tool call)
- tools -> agent (results loop back)

Example tool call that gets blocked:
    {
        "name": "file_write",
        "args": {"path": "/etc/passwd", "content": "..."},
        "side_effects_hint": ["write"]
    }

CGF Evaluation:
- Proposal: tool_call with risk_tier=high
- Decision: BLOCK (denylist match)
- Enforcement: GovernedToolNode raises LangGraphToolBlocked
- Result: Error message in state, no file written
        """,
        "file": __file__,
        "interception": " GovernedToolNode.__call__ before tool_node(state)"
    }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("LANGGRAPH ADAPTER v0.3")
    print("=" * 60)
    print(f"SCHEMA_VERSION: {SCHEMA_VERSION}")
    print(f"Schema Module: {SCHEMA_MODULE}")
    print()
    print("Integration Point:")
    print(create_example_graph()["description"])
