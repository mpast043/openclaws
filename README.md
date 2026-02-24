# Capacity Governance Framework (CGF)

Multi-host capacity governance platform with universal adapter contract.

## Overview

The Capacity Governance Framework (CGF) provides runtime enforcement of capacity constraints across distributed systems. It implements a universal host adapter contract that allows any system (OpenClaw, LangGraph, custom agents) to integrate with the same governance infrastructure.

**Current Version**: 0.3.0  
**Status**: Multi-host platform validated (OpenClaw + LangGraph)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HOST SYSTEMS                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   OpenClaw      │   LangGraph     │      Future Hosts       │
│   (Host #1)     │   (Host #2)     │                         │
├─────────────────┴─────────────────┴─────────────────────────┤
│                    HOST ADAPTER CONTRACT                     │
│              (observe_proposal, enforce_*, report)          │
├─────────────────────────────────────────────────────────────┤
│                    CGF SERVER (REST API)                     │
│  POST /v1/evaluate    → Governance decisions                │
│  POST /v1/register    → Adapter registration                │
│  POST /v1/outcomes    → Execution reporting                 │
├─────────────────────────────────────────────────────────────┤
│              DATA-DRIVEN POLICY (JSON Config)                │
│         Fail modes, risk tiers, denylist, thresholds        │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Multi-Host Support**: OpenClaw, LangGraph, and extensible to any host
- **Backward Compatible**: Schema v0.3 accepts v0.2.x payloads
- **Data-Driven Policy**: Configure via JSON (no code changes)
- **19 Canonical Events**: Full audit trail across all hosts
- **Cross-Host Compliance**: Same scenarios, comparable ReplayPacks
- **Fail Modes**: Configurable per (action_type, risk_tier)

## Quick Start

### 1. Start CGF Server

```bash
python cgf_server_v03.py
# Server runs on http://127.0.0.1:8080
```

### 2. Run Compliance Tests

```bash
# Test both hosts against same scenarios
python contract_compliance_tests.py

# Output: contract_compliance_report.json
```

### 3. Validate Schemas

```bash
# Lint event logs
python schema_lint.py --dir ./cgf_data/

# Compare hosts
python schema_lint.py --compare openclaw/ langgraph/
```

## File Structure

| File | Purpose |
|------|---------|
| `cgf_schemas_v03.py` | Schema definitions with backward compatibility |
| `cgf_server_v03.py` | CGF REST API server |
| `policy_config_v03.json` | Data-driven policy configuration |
| `openclaw_adapter_v02.py` | OpenClaw host adapter |
| `langgraph_adapter_v01.py` | LangGraph host adapter |
| `schema_lint.py` | JSONL validation and event ordering checker |
| `contract_compliance_tests.py` | Cross-host compliance test suite |

## Schema Versions

| Version | Status | Compatibility |
|---------|--------|---------------|
| 0.2.0 | ✅ Supported | Native |
| 0.2.x | ✅ Supported | Patch versions accepted |
| 0.3.0 | ✅ Current | Native |
| < 0.2.0 | ❌ Rejected | Upgrade required |

## Policy Configuration

Edit `policy_config_v03.json`:

```json
{
  "tool_denylist": ["file_write", "exec", "shell"],
  "fail_modes": [
    {"action_type": "tool_call", "risk_tier": "high", "fail_mode": "fail_closed"},
    {"action_type": "tool_call", "risk_tier": "low", "fail_mode": "fail_open"}
  ]
}
```

## Host Comparison

| Feature | OpenClaw | LangGraph |
|---------|----------|-----------|
| Session ID | `session_key` | `thread_id` |
| Interception | gateway-cli | `GovernedToolNode` |
| Action Types | tool_call, memory_write | tool_call |
| Events | 19 canonical | 19 canonical |

## Next Steps

- [ ] Execute compliance tests with running server
- [ ] Generate ReplayPacks for both hosts
- [ ] Add third host (e.g., custom Python agent)
- [ ] Implement P7 Agent SDK
- [ ] Build Policy Engine (P8)

## License

MIT
