# Capacity Governance Framework (CGF) — Core Runtime

> **Purpose**: This repository (`openclaws`) contains the **core capacity platform runtime (P0–P5)**.
> 
> **Governance, adapters, and policy engine** live in: **https://github.com/mpast043/host-adapters**
> 
> **Note**: The CGF versions in this repo (v0.3) are historical. Active development moved to host-adapters.

## Repo Map

| Repo | Purpose | Latest Tag |
|------|---------|------------|
| **openclaws** | Kernel runtime (P0–P5) | v0.8.0 |
| **host-adapters** | Governance + Policy Engine (P6+) | [v0.5.0](https://github.com/mpast043/host-adapters) |

## Overview

The Capacity Governance Framework (CGF) provides runtime enforcement of capacity constraints across distributed systems. **This repository contains the foundational runtime only.**

**Current Version**: 0.3.0 (historical)  
**Status**: Core runtime complete (P0–P5)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GOVERNANCE LAYER (P6+)                    │
│            → https://github.com/mpast043/host-adapters      │
├─────────────────────────────────────────────────────────────┤
│                    CORE RUNTIME (P0–P5)                      │
│                    [THIS REPOSITORY]                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Host      │  │    DB       │  │       GPU           │  │
│  │ Substrate   │  │   Pool      │  │   Substrate         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Network    │  │   Memory    │  │    Recovery          │  │
│  │ Substrate   │  │  Library    │  │    (HA)              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## What Lives Where

### In This Repo (openclaws) — P0–P5
- Substrate monitoring (`substrate_monitor.py`)
- Step 3 Truth ledger
- Distributed coordination (gossip/consensus)
- HA/recovery infrastructure
- Network substrate
- Memory library (patterns/queries)

### In host-adapters — P6+
- Host Adapter v1 SPEC
- OpenClawAdapter v0.2
- LangGraphAdapter v0.1
- CGF Server v0.5+ (with Policy Engine v1.0)
- Policy Engine (deterministic, explainable)
- Contract compliance suite

## Quick Start

See **host-adapters** repo for current governance integration:
```bash
git clone https://github.com/mpast043/host-adapters.git
cd host-adapters
./tools/run_contract_suite.sh
```

## Historical Files

The CGF v0.3 files in this repo (`cgf_server_v03.py`, adapters, etc.) are preserved under `/archive/` for reference. They are **superseded** by host-adapters v0.5.0.

## Phase Status

| Phase | Component | Location | Status |
|-------|-----------|----------|--------|
| P0 | Real Substrate | openclaws | ✅ Complete |
| P1 | Step 3 Truth | openclaws | ✅ Complete |
| P1.5 | Gate Calibration | openclaws | ✅ Complete |
| P2 | Distributed Coordination | openclaws | ✅ Complete |
| P3 | HA/Recovery | openclaws | ✅ Complete |
| P3.5 | Distributed Recovery | openclaws | ✅ Complete |
| P4 | Memory Library | openclaws | ✅ Complete |
| P5 | Network Substrate | openclaws | ✅ Complete |
| P6 | Host Adapters | host-adapters | ✅ v0.3+ |
| P6.5 | Schema Hardening | host-adapters | ✅ v0.4+ |
| P7 | Agent SDK | host-adapters | ✅ v0.4+ |
| P8 | Policy Engine | host-adapters | ✅ v0.5.0 |

## License

MIT
