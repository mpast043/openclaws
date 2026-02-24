# MEMORY.md - Capacity Platform Project

## Project Overview

**Name**: Structural-Stability Capacity-Governed Systems Platform  
**Purpose**: Runtime for enforcing capacity constraints via 5-axis capacity vectors (C_geo, C_int, C_gauge, C_ptr, C_obs)  
**Version**: 0.8.0  
**Phases**: P0-P5 Complete (13 demos)  

---

## Platform Phases

| Phase | Component | Status | Key Files |
|-------|-----------|--------|-----------|
| P0 | Real Substrate | ✅ Host/DB/GPU | `substrate_monitor.py` |
| P1 | Step 3 Truth | ✅ Ledger/Gates/Tracker | `step3_truth.py` |
| P1.5 | Gate Calibration | ✅ Adaptive thresholds | `gate_calibration.py` |
| P2 | Distributed Coordination | ✅ Gossip + Consensus | `distributed.py` |
| P3 | HA/Recovery | ✅ Heartbeat + Registry | `ha_recovery.py` |
| P3.5 | Distributed Recovery | ✅ Quorum + Race-to-Claim | `distributed_recovery.py` |
| P4 | Memory Library | ✅ Pattern Store + Queries | `memory_library.py` |
| P5 | Network Substrate | ✅ Graph Topology + Selection | `network_substrate.py` |

---

## Substrates

### Implemented
| Substrate | Geometry | Gates | Demo File |
|-----------|----------|-------|-----------|
| Host | Continuous (N_geo=262K) | fit, gluing, UV bounds, isolation | `all_substrates_demo.py` |
| DB Pool | 2D Discrete (N_geo=400) | Connection coherence | `all_substrates_demo.py` |
| GPU | 3D Discrete (N_geo=13,824) | SM occupancy, temperature | `all_substrates_demo.py` |
| Network | Graph topology | fit_error, gluing_delta, isolation | `network_substrate_demo.py` |

### Candidate Substrates
| Priority | Substrate | Key Metrics |
|----------|-----------|-------------|
| 1 | Storage I/O | throughput, queue depth, latency |
| 2 | Thermal | CPU/GPU temp, fan speed, throttling |
| 3 | Message Queue | depth, lag, throughput |
| 4 | Cache | L1/L2/L3 miss rates |

---

## Key Metrics

**Detection latency**: ~1.5s (3 missed heartbeats × 0.5s)  
**Recovery time**: <100ms (registry + claim)  
**Gate thresholds**: fit < 0.165, gluing < k/√N_geo

---

## Repository

**URL**: https://github.com/mpast043/openclaws.git  
**Path**: `/tmp/openclaws/Repos/capacity-platform`  
**Branch**: main  
**Latest commit**: f56700b (v0.8.0)

---

## Validation Status

✅ **Functionally proven**: All demos operational, unit tests pass  
⚠️ **Empirically unproven**: Needs real workload data validation

**Next validation step**: Ingest Prometheus/Datadog metrics, compare prediction accuracy vs naive thresholds

---

## Session Logs

- `session_logs/capacity_platform_bg.md` - Background research thread

---

## HostAdapter v1 (2026-02-24)

**Status**: Spec ratified, implementation complete, integration tests passed  
**Purpose**: Universal integration contract between host systems (OpenClaw, LangGraph, etc.) and CGF

### Architecture Decision
Shift from core infrastructure runtime (P0-P5) to **adapter phase** (P6):
- Infrastructure substrates continue running (monitors, gossip, recovery)
- New layer: Host adapters govern action execution on external hosts
- CGF becomes standalone service with REST endpoints

### Design

| Component | Status | File |
|-----------|--------|------|
| HostAdapter v1 Spec | ✅ Ratified | `HostAdapter_v1_SPEC.md` (34KB) |
| OpenClawAdapter v0.1 | ✅ Complete | `openclaw_adapter_v01.py` (18KB) |
| CGF Server v0.1 | ✅ Complete | `cgf_server_v01.py` (14KB) |
| Node.js Integration Patch | ✅ Complete | `openclaw_cgf_hook.mjs` (14KB) |
| Integration Tests | ✅ 3/3 Passed | `test_openclaw_adapter_v01.py` |

### Integration Test Results

| Test | Scenario | Result |
|------|----------|--------|
| 1 | Denylisted tool (`file_write`) | ✅ BLOCKED |
| 2 | Non-denylisted tool (`ls`) | ✅ ALLOWED |
| 3 | CGF down + side-effect tool | ✅ FAIL_CLOSED (BLOCK) |
| 3b | CGF down + read-only tool | ✅ FAIL_OPEN (ALLOW) |

### Fail Mode Validation
- **Side-effect tools**: Fail **closed** → BLOCK on CGF unreachable
- **Read-only tools**: Fail **open** → ALLOW on CGF unreachable (with event logging)

### Core Abstractions
- **5 Decision Types**: ALLOW, CONSTRAIN, AUDIT, DEFER, BLOCK
- **4 Action Types**: tool_call, message_send, memory_write, workflow_step
- **19 Canonical Events**: Full audit trail (adapter_registered → outcome_logged)
- **Fail Mode Config**: Per-risk-tier (high/medium/low)

### CGF Endpoints
```
POST /v1/evaluate          → Governance decisions
POST /v1/adapters/register → Adapter registration
POST /v1/outcomes/report   → Execution outcomes
POST /v1/capacity/signals  → Async capacity updates
```

### Enforcement Invariants
1. Constraint fails → Treat as BLOCK + emit constraint_failed
2. Audit hooks fail → Fallback per risk tier
3. CGF unavailable → Apply configured fail_mode
4. Unexpected errors → Fail closed (BLOCK)

### Performance Targets
| Metric | Target | Max |
|--------|--------|-----|
| Evaluation | <100ms | 500ms |
| Enforcement | <5ms | 20ms |
| Event emission | Async, <50ms | 100ms |

---

## Phase Status

| Phase | Component | Status | Version |
|-------|-----------|--------|---------|
| P0-P5 | Core Runtime | ✅ Complete | 0.8.0 |
| P6 | Host Integration | ✅ v0.3 Multi-Host | Spec 1.0.0, Impl 0.3.0 |
| P6.5 | Schema Hardening | ✅ v0.2 Complete | Schemas 0.2.0 |
| P6.6 | Multi-Host Platform | ✅ v0.3 Complete | LangGraph + OpenClaw |
| P7 | Agent SDK | 📋 Planned | — |
| P8 | Policy Engine | 📋 Planned | — |

---

## HostAdapter v0.2 (2026-02-24)

**Status**: Schema hardened, memory_write governance implemented, **12/12 tests passing**
**Purpose**: Platform contract freeze + memory_write action_type

### Deliverables Completed

| Component | Status | File |
|-----------|--------|------|
| Schema v0.2 Contract | ✅ Ratified | `cgf_schemas_v02.py` |
| CGF Server v0.2 | ✅ Complete | `cgf_server_v02.py` |
| OpenClawAdapter v0.2 | ✅ Complete | `openclaw_adapter_v02.py` |
| Node.js Hook v0.2 | ✅ Complete | `openclaw_cgf_hook_v02.mjs` |
| Replay Script | ✅ Complete | `replay_governance_timeline.py` |
| Test Suite v0.2 | ✅ 12/12 Passed | `test_cgf_v02.py` |

### Schema Hardening (A)

**1. Schema Versioning**: All payloads include `schema_version: "0.2.0"`
- HostEvaluationRequest
- CGFDecision
- HostOutcomeReport
- HostEvent

**2. Canonical EventType Enum** (19 required types):
```
ADAPTER_REGISTERED, ADAPTER_DISCONNECTED
PROPOSAL_RECEIVED, PROPOSAL_ENACTED, PROPOSAL_EXPIRED, PROPOSAL_REVOKED
DECISION_MADE, DECISION_REJECTED
ACTION_ALLOWED, ACTION_BLOCKED, ACTION_CONSTRAINED, ACTION_DEFERRED, ACTION_AUDITED
ERRORS, CONSTRAINT_FAILED, CGF_UNREACHABLE, EVALUATE_TIMEOUT
OUTCOME_LOGGED, SIDE_EFFECT_REPORTED
```

**3. Replay System**:
- `ReplayPack` Pydantic model with complete governance timeline
- `/v1/proposals/{id}/replay` endpoint → JSON with proposal + decision + events
- `replay_governance_timeline.py` CLI for log analysis

### Memory Write Governance (B, C, D)

**Action Type**: `memory_write`

**Parameters**:
```json
{
  "namespace": "sessions/default",
  "size_bytes": 1024,
  "ttl": null,
  "sensitivity_hint": "low|medium|high",
  "content_hash": "sha256_hash",
  "context_refs": ["session-abc"],
  "operation": "create|update|delete|append"
}
```

**Policy v0.2** (CGF Server):
| Condition | Decision | Justification |
|-----------|----------|---------------|
| size_bytes > 10MB | CONSTRAIN | "Large memory write requires quarantine" |
| sensitivity=high + confidence<0.8 | CONSTRAIN | "High sensitivity with insufficient confidence" |
| denylist match | BLOCK | "Tool/memory in denylist" |
| Default | ALLOW | "Within policy thresholds" |

**Fail Mode** (CGF down):
- Medium/high sensitivity → BLOCK (fail closed)
- Low sensitivity → ALLOW (fail open with logging)

**Test Results** (12/12 passed):
| Test | Result |
|------|--------|
| Schema version constant | ✅ PASS |
| Server version endpoint | ✅ PASS |
| Registration schema | ✅ PASS |
| Tool call blocked | ✅ PASS |
| Tool call allowed | ✅ PASS |
| Memory write allowed | ✅ PASS |
| Memory write constrained | ✅ PASS |
| Memory sensitivity high | ✅ PASS |
| Event type enum | ✅ PASS (19 types) |
| Event validation | ✅ PASS |
| Outcome committed/quarantined | ✅ PASS |
| Replay pack generation | ✅ PASS |

### OpenClaw Integration Point

**Memory Write Interception**:
```
File: pi-embedded-helpers-CMf7l1vP.js
Function: updateSessionStore(storePath, mutator, opts)
Line: ~6254

Call Chain:
  agent.ts:persistSessionEntry()
    → updateSessionStore()
       → loadSessionStore()          [load existing]
       → governanceHookMemoryWrite() [CGF evaluation]
       → mutator(store)              [apply mutations]
       → saveSessionStoreUnlocked()  [atomic write]
```

**Integration**:
```javascript
import { governanceHookMemoryWrite } from './openclaw_cgf_hook_v02.mjs';

async function updateSessionStore(storePath, mutator, opts) {
  return await withSessionStoreLock(storePath, async () => {
    const store = loadSessionStore(storePath, { skipCache: true });
    
    // CGF GOVERNANCE HOOK (v0.2)
    const governance = await governanceHookMemoryWrite(
      storePath, store, mutator, opts
    );
    if (!governance.allowed) {
      throw new Error('Memory write blocked by CGF');
    }
    
    const result = await mutator(store);
    await saveSessionStoreUnlocked(storePath, store, opts);
    return result;
  });
}
```

### Host-Agnostic Policy Verification

CGF Server policy **no OpenClaw-specific logic**:
- Evaluates: `action_params.sensitivity_hint`, `size_bytes`, `content_hash`
- No branching on: `host_type`, session keys, tool names (except denylist)
- Constraints: Generic `quarantine_namespace` → `_quarantine_{timestamp}/`
- All decisions return: `decision`, `confidence`, `justification`, `reason_code`

---

## HostAdapter v0.3 — Multi-Host Platform

**Status**: LangGraphAdapter complete, contract compliance framework ready
**Goal**: Validate cross-host contract compatibility

### Deliverables

| Component | Status | File |
|-----------|--------|------|
| Schema v0.3 | ✅ | `cgf_schemas_v03.py` |
| CGF Server v0.3 | ✅ | `cgf_server_v03.py` |
| LangGraph Adapter v0.1 | ✅ | `langgraph_adapter_v01.py` |
| Policy Config v0.3 | ✅ | `policy_config_v03.json` |
| Schema Lint Tool | ✅ | `schema_lint.py` |
| Contract Compliance Tests | ✅ | `contract_compliance_tests.py` |

### Schema Compatibility (v0.2 → v0.3)

| Version | Status |
|---------|--------|
| 0.2.0 | ✅ Accepted |
| 0.2.x (patch) | ✅ Accepted |
| 0.3.0 | ✅ Native |
| < 0.2.0 | ❌ Rejected |

**Rules**: Additive fields allowed, required fields enforced, event ordering validated.

### Host Comparison

| Aspect | OpenClaw | LangGraph |
|--------|----------|-----------|
| Host Type | `openclaw` | `langgraph` |
| Session ID | session_key | thread_id |
| Interception | gateway-cli | GovernedToolNode |
| Action Types | tool_call, memory_write | tool_call |
| Fail Modes | Configurable | Configurable |

### Policy v0.3 (Data-Driven)

```json
{
  "fail_modes": [
    {"action_type": "tool_call", "risk_tier": "high", "fail_mode": "fail_closed"},
    {"action_type": "tool_call", "risk_tier": "medium", "fail_mode": "defer"},
    {"action_type": "tool_call", "risk_tier": "low", "fail_mode": "fail_open"}
  ]
}
```

**Key Invariant**: Policy evaluates only `action_params`, `risk_tier`, `capacity_signals` — **no branching on host_type**.

---

## Notes

*Last updated: 2026-02-24 13:42 EST*
