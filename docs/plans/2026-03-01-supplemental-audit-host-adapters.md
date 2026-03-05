# Supplemental Audit: host-adapters Workspace

**Generated:** 2026-03-01
**Scope:** `/tmp/openclaws/Repos/host-adapters/` (Canonical Framework Workspace)

---

## Executive Summary

This supplemental audit covers the **canonical repository** for Framework v4.5 governance and WORKFLOW_AUTO execution, which was not included in the initial Clawdbot audit.

### Critical Discovery

**The actual "Framework with Selection" workflow execution is happening in `/tmp/openclaws/Repos/host-adapters/`, NOT in Clawdbot.**

Per `docs/CANONICAL_STATE.md`:
> "The canonical repository for Framework v4.5 governance and workflow execution is: `/tmp/openclaws/Repos/host-adapters`"

---

## Canonical Status

| Repository | Status | Purpose |
|------------|--------|---------|
| `/tmp/openclaws/Repos/host-adapters/` | **CANONICAL** | Framework governance, WORKFLOW_AUTO, Selection |
| `/tmp/openclaws/Repos/host-adapters-experimental-data/` | **CANONICAL** | Experimental artifacts, RUN_* directories |
| `/Users/meganpastore/Clawdbot/` | **NON-CANONICAL** | Development workspace, capacity-demo |
| `/Users/meganpastore/Clawdbot/Repos/framework-recon-clone/` | **NON-CANONICAL** | Archive |

---

## Workspace Structure

```
/tmp/openclaws/Repos/host-adapters/
├── adapters/                    # CGF adapters
│   ├── langgraph_adapter_v01.py
│   ├── openclaw_adapter_v01.py
│   ├── openclaw_adapter_v02.py
│   └── openclaw_cgf_hook_v02.mjs
├── cgf_data/                    # CGF server data
├── cgf_policy/                  # Policy bundles
├── docs/
│   ├── CANONICAL_STATE.md       # Canonical status declaration
│   ├── WORKFLOW_AUTO_AGENTIC.md # Agentic workflow spec
│   ├── OPENCLAW_CGF_RUNTIME.md  # CGF runtime docs
│   └── physics/                 # Physics experiment specs
├── experiments/
│   ├── claim3/                  # Claim 3P experiment runners
│   │   ├── exp3_claim3_physical_convergence_runner.py
│   │   ├── exp3_claim3_optionB_runner.py
│   │   └── w03_controls_runner.py
│   └── physics/                 # P1/P2/P3 experiments
│       ├── exp_p1_spectral_dimension_runner.py
│       ├── exp_p2_capacity_plateau_runner.py
│       └── exp_p3_gluing_excision_stability_runner.py
├── outputs/                     # Job outputs (today's runs)
│   └── JOB_20260301_*/          # W03 control experiments
├── policy/                      # Policy configuration
├── RUN_20260228_150457/         # Complete workflow run
│   ├── results/selection/       # Selection report
│   └── summary.md
└── tools/                       # Workflow tools
```

---

## WORKFLOW_AUTO Execution Status

### RUN_20260228_150457 (COMPLETE)

**Duration:** 25 minutes
**Objective:** PHYSICS_P
**Status:** COMPLETE

#### Baseline Results

| Test | Verdict | Key Result |
|------|---------|------------|
| P1 | **ACCEPTED** | d_s = 1.336 ± 0.029 |
| P2 | **REJECTED** | Saturation rejected at large A |
| P3 | **UNDERDETERMINED** | Violations observed, controls missing |
| P4_Ising | **REJECTED** | Fidelity 0.218 < 0.95 |

#### Exploration Results

| Test | Target | Verdict | Key Finding |
|------|--------|---------|-------------|
| EXP_001 | W02 | REJECTED | chi=8 fidelity 0.217 |
| EXP_002 | W02 | REJECTED | chi=16 fidelity 0.218 |
| EXP_003 | W02 | **TENTATIVE_ACCEPT** | Heisenberg chi=4: fidelity 0.997 |
| EXP_004 | W01 | REJECTED | Large A shows proper saturation |
| EXP_005 | W03 | REJECTED | Violations across 3 seeds |

#### Selection Ledger

| Claim | Verdict | Confidence |
|-------|---------|------------|
| W01 | **ACCEPTED** | High |
| W02 (Heisenberg) | **TENTATIVE_ACCEPT** | Low (leverage insufficient) |
| W02 (Ising) | **REJECTED** | High |
| W03 | **UNDERDETERMINED** | N/A (controls missing) |

### Today's Runs (2026-03-01)

W03 control experiments in progress:
- `JOB_20260301_162730_5060c7aa_outputs/`
- 18 W03 control configurations (positive/negative, A8/A16/A32, 3 seeds each)

---

## Key Documents

### CANONICAL_STATE.md
Declares this repository as the source of truth for:
- CGF server and host adapters
- WORKFLOW_AUTO execution and artifact contract
- Selection/evidence retention pipeline
- Claim 3P experiment runner and result interpretation

### WORKFLOW_AUTO_AGENTIC.md
Describes multi-agent coordinator with three roles:
- `planner`: runs test planning
- `researcher`: runs research on underdetermined claims
- `executor`: runs workflow execution

### Selection Reports
Located in `RUN_*/results/selection/selection_report.md`:
- Verdicts per claim (ACCEPTED/TENTATIVE_ACCEPT/REJECTED/UNDERDETERMINED)
- Evidence witness validation
- Compliance with WORKFLOW_AUTO.md Sections 9.1-9.5

---

## Physics Experiments

### P1: Spectral Dimension
- **Status:** ACCEPTED
- **Result:** d_s = 1.336 ± 0.029 (within 0.15 tolerance of Sierpinski dimension)
- **Runner:** `experiments/physics/exp_p1_spectral_dimension_runner.py`

### P2: Capacity Plateau
- **Status:** REJECTED (as expected)
- **Result:** Saturation rejected at large A
- **Runner:** `experiments/physics/exp_p2_capacity_plateau_runner.py`

### P3: Gluing/Excision Stability
- **Status:** UNDERDETERMINED
- **Blocker:** Missing positive/negative controls per Section 9.4
- **Runner:** `experiments/physics/exp_p3_gluing_excision_stability_runner.py`

### Claim 3P: Physical Convergence
- **Ising:** REJECTED (fidelity saturates at 0.22)
- **Heisenberg:** TENTATIVE_ACCEPT (fidelity 0.997, insufficient leverage)
- **Runner:** `experiments/claim3/exp3_claim3_physical_convergence_runner_v2.py`

---

## CGF Infrastructure

### Server
- Default port: 8080 (configurable via `CGF_PORT`)
- Health endpoint: `/v1/health`
- Start: `python server/cgf_server_v03.py`

### Contract Tests
```bash
pytest -v tools/contract_compliance_tests.py
# Or with custom port:
CGF_PORT=8082 CGF_ENDPOINT=http://127.0.0.1:8082 ./tools/run_contract_suite.sh
```

### Adapters
- `openclaw_adapter_v02.py` - Latest OpenClaw adapter
- `langgraph_adapter_v01.py` - LangGraph integration
- `openclaw_cgf_hook_v02.mjs` - JavaScript hook for ES module

---

## Comparison: Clawdbot vs host-adapters

| Aspect | Clawdbot | host-adapters |
|--------|----------|---------------|
| Status | Development workspace | **Canonical** |
| WORKFLOW_AUTO | Defined but NOT executed | **Actively executed** |
| Selection Reports | Templates only | **Actual verdicts** |
| CGF Server | Not present | **Present** |
| Claim Verdicts | N/A | **W01 ACCEPTED, W02 TENTATIVE, W03 UNDERDETERMINED** |
| Physics Experiments | capacity-demo only | **P1/P2/P3/Claim3P** |
| Multi-agent Workflow | N/A | **planner/researcher/executor** |

---

## Active Work (2026-03-01)

### W03 Control Experiments
Running 18 configurations:
- Positive/negative controls
- Lattice sizes: A8, A16, A32
- Seeds: 42, 123, 456

Purpose: Resolve W03 UNDERDETERMINED verdict by providing explicit controls per Section 9.4.

---

## Recommendations

### 1. Update Primary Workspace
The initial audit should have focused on `/tmp/openclaws/Repos/host-adapters/` as the canonical workspace. Clawdbot is a development mirror.

### 2. Consolidate Documentation
- Move Clawdbot/docs/FRAMEWORK_v45_canonical.txt → host-adapters/docs/
- Update WORKFLOW_AUTO.md in Clawdbot to reference canonical location

### 3. Sync capacity-demo
The capacity-demo in Clawdbot has newer experiment outputs. Consider syncing:
- `outputs/nonseparable_rewire/`
- `outputs/gap_quantile/`
- `outputs/multi_axis_v3/`

### 4. Next Steps for Selection
1. Complete W03 control experiments (in progress)
2. Extend W02 Heisenberg chi sweep to 2,4,8,16,32 with 3+ seeds
3. Re-run selection with expanded evidence

---

## File Locations

| Item | Location |
|------|----------|
| Canonical declaration | `docs/CANONICAL_STATE.md` |
| Workflow runs | `RUN_*/` |
| Selection reports | `RUN_*/results/selection/` |
| Physics experiments | `experiments/physics/` |
| Claim 3 experiments | `experiments/claim3/` |
| Today's jobs | `outputs/JOB_*/` |
| CGF server | `server/cgf_server_v03.py` |
| Contract tests | `tools/contract_compliance_tests.py` |

---

*Supplemental audit generated: 2026-03-01*