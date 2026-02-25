# Archive: CGF v0.3 Files

**Status**: Historical reference only — superseded by host-adapters

## Overview

The files in `archive/cgf_v03/` represent the **v0.3** implementation of the Capacity Governance Framework. These are preserved for historical reference but are **not the active development line**.

## Active Development

**Current governance, adapters, and policy engine**: https://github.com/mpast043/host-adapters

| Version | Location | Status |
|---------|----------|--------|
| v0.5.0 | host-adapters | ✅ Active (P8 Policy Engine) |
| v0.4.x | host-adapters | ✅ Stable |
| v0.3.x | openclaws/archive | 📦 Archived |

## What's in This Archive

| File | Purpose |
|------|---------|
| `cgf_schemas_v03.py` | Schema definitions (v0.2 + v0.3) |
| `cgf_server_v03.py` | CGF REST API server (historical) |
| `openclaw_adapter_v02.py` | OpenClaw host adapter (historical) |
| `langgraph_adapter_v01.py` | LangGraph host adapter (historical) |
| `contract_compliance_tests.py` | Cross-host compliance tests |
| `schema_lint.py` | JSONL validation tool |
| `policy_config_v03.json` | Data-driven policy config |
| Various test files | Unit/integration tests |

## Migration Path

If you were using these files:

1. **Replace** with host-adapters repo:
   ```bash
   git clone https://github.com/mpast043/host-adapters.git
   ```

2. **Update imports** from `cgf_schemas_v03` to new schemas

3. **Run contract suite** to verify:
   ```bash
   ./tools/run_contract_suite.sh
   ```

## Tags

- `v0.8.0` — Last openclaws release with these files at root
- `v0.5.0` — Current host-adapters release (supersedes these)

## License

MIT (same as parent repo)
