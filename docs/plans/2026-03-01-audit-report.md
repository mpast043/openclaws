# Project Audit Report

**Generated:** 2026-03-01
**Scope:** Framework v4.5/v4.6 (Capacity→Geometry) Integration
**Author:** Claude Code Agent

> **⚠️ CRITICAL UPDATE:** The canonical repository for Framework governance is `/tmp/openclaws/Repos/host-adapters/`. See the Supplemental Audit at `docs/plans/2026-03-01-supplemental-audit-host-adapters.md` for the actual WORKFLOW_AUTO execution status and selection verdicts.

---

## Executive Summary

This audit catalogs the current state of the Capacity->Geometry research project. The primary implementation resides at `Clawdbot/Repos/capacity-demo/` with 11 experiment output directories containing validated results. Multiple duplicate repositories and backup directories exist that should be archived. The Framework validation is progressing well with Step 2 gates passing; Step 3 selection gates require truth infrastructure implementation.

### Key Findings

1. **Primary repository is healthy:** `capacity-demo` has comprehensive test coverage and active experiment outputs
2. **Experiment results validated:** Nonseparable rewire, gap quantile, and multi-axis tests show expected behaviors
3. **Delta-lambda rigidity confirmed:** Exponential shielding law validated across gap quantile experiments
4. **Framework validation progressing:** Step 2 gates (fit, gluing, UV, isolation) passing on v3 runs
5. **Multiple duplicates exist:** Four duplicate/legacy directories identified for archival
6. **Selection infrastructure needed:** Step 3 selection gates require truth implementation

---

## 1. Project Inventory

### 1.1 Primary Repositories

| Path | Purpose | Status | Notes |
|------|---------|--------|-------|
| `/Users/meganpastore/Clawdbot/Repos/capacity-demo/` | Framework implementation | ACTIVE | Primary codebase, 97+ tests passing |
| `/Users/meganpastore/Clawdbot/Repos/capacity-platform/` | Operational runtime | ACTIVE | New capacity_kernel module |
| `/Users/meganpastore/Clawdbot/Repos/spectralregime-mvp/` | Trading application | COMPLETE | Spectral regime analysis for markets |
| `/Users/meganpastore/Clawdbot/Repos/capacity-quant/` | Quantitative analysis | ACTIVE | Market capacity experiments |
| `/Users/meganpastore/Clawdbot/Repos/polymarket_arb_bot/` | Arbitrage bot | ACTIVE | Polyarb implementation |
| `/Users/meganpastore/Clawdbot/Repos/IntoTheUnknown/` | Data exploration | ACTIVE | Pipeline scripts and data |
| `/Users/meganpastore/Clawdbot/Repos/memory/` | Session logs | REFERENCE | Daily session summaries |
| `/Users/meganpastore/Clawdbot/Repos/Market Program Work/` | Market analysis | ACTIVE | SPX and market data |

### 1.2 Duplicates and Legacy Directories

| Path | Status | Recommended Action | Notes |
|------|--------|-------------------|-------|
| `/Users/meganpastore/Projects/capacity-demo/` | DUPLICATE | Archive | Clone of Clawdbot/Repos/capacity-demo |
| `/Users/meganpastore/Projects/capacity-demo-local/` | DUPLICATE | Archive | Older local copy |
| `/Users/meganpastore/Projects/polymarket_arb_bot_v2/` | DUPLICATE | Review | May have newer code |
| `/Users/meganpastore/Clawdbot/Repos/framework-recon-clone/` | LEGACY | Archive | Clone with Framework v4.5 PDF |
| `/Users/meganpastore/.openclaw_backup_1770997879/` | BACKUP | Delete | Old OpenClaw backup |
| `/Users/meganpastore/.openclaw_legacy_20260227_184228/` | LEGACY | Archive | Legacy OpenClaw config |
| `/Users/meganpastore/.openclaw-backup-20260221-134754.tar.gz` | BACKUP | Archive | Compressed backup |

### 1.3 Primary Codebase Structure

```
Clawdbot/Repos/capacity-demo/
├── dimshift/              # Core library (16 modules)
│   ├── __init__.py
│   ├── capacity.py
│   ├── framework_spectral.py
│   ├── graph_heat.py
│   ├── multi_axis_capacity.py   # Multi-axis C vector support
│   ├── nonseparable.py          # Nonseparable Laplacian experiments
│   ├── rewire.py                # Rewiring utilities
│   ├── spectral.py
│   ├── spectral_filters.py
│   ├── substrates.py
│   ├── sweep.py
│   └── theorem.py               # 8 formal verification obligations
├── scripts/               # Experiment runners
│   ├── run_capacity_dimshift.py
│   ├── run_gap_quantile.py
│   ├── run_gap_tail.py
│   ├── run_multi_axis_sweep.py
│   ├── run_nonseparable_rewire_test.py
│   └── ...
├── tests/                 # Test suite
├── docs/                  # Documentation
├── outputs/               # Experiment artifacts
└── results/               # Templates and intermediate results
```

---

## 2. Experiment Outputs Inventory

### 2.1 Summary Table

| Directory | Experiment Type | Runs | Status | Key Result |
|-----------|-----------------|------|--------|------------|
| `nonseparable_rewire/` | Step 1 robustness | 18 | COMPLETE | Staircase persists under rewiring |
| `gap_quantile/` | Delta-lambda sweep | 1 | COMPLETE | Power-law regime at q=0.02 |
| `gap_tail/` | Gap collapse test | 1 | COMPLETE | Rigid regime maintained |
| `multi_axis_test/` | Gate validation | 1 | PARTIAL | Gates FAIL (debug run) |
| `multi_axis_v2/` | Gate validation | 1 | PARTIAL | fit PASS, gluing FAIL |
| `multi_axis_v3/` | Gate validation | 1 | PASS | All Step 2 gates PASS |
| `n64_test/` | Large lattice test | 1 | PASS | All Step 2 gates PASS |
| `framework_validation_b/` | Option B validation | 3 | COMPLETE | Monotone d_s confirmed |
| `capacity_dimshift/` | Core experiments | 2 | COMPLETE | Staircase validated |
| `spx_capacity_probe/` | Market application | 2 | COMPLETE | SPX crisis analysis |

### 2.2 Detailed Experiment Results

#### 2.2.1 Nonseparable Rewire (`outputs/nonseparable_rewire/`)

**Purpose:** Test Framework robustness under nonseparable Laplacian rewiring

**Run Configurations:**
- D=2: N=6, r=0.2 (1 run)
- D=3: N=8,16,32,64 with r from 0.01 to 0.08 (17 runs total)
- Seeds: 11, 22, 33, 42 (multi-seed tests for N=64, r=0.03)

**Key Files:**
- `class_splitting_index.csv` (47,735 bytes) - Full run details
- `boundary_stats_by_N_r.csv` - Aggregated boundary statistics
- `split_peak_table.csv` - Peak detection results

**Key Findings:**
| N | r | Boundary Presence | Peak R_gamma | Notes |
|---|---|-------------------|--------------|-------|
| 16 | 0.03 | 0.35-0.70 | 1.60-1.74 | Clear 2-band structure |
| 32 | 0.03 | 0.38-0.67 | 1.98-2.99 | Robust plateau boundaries |
| 64 | 0.03 | 0.35-0.70 | 0.95-2.54 | Multi-seed stability confirmed |
| 64 | 0.05 | 0.35-0.70 | 1.45-3.21 | Higher rewire = more structure |

**Conclusion:** Capacity staircase is robust under nonseparable Laplacian rewiring. Plateau boundaries remain stable across seeds.

#### 2.2.2 Gap Quantile (`outputs/gap_quantile/`)

**Purpose:** Test regime transition as spectral gap (Delta-lambda) shrinks

**Key Files:**
- `gap_quantile_summary.csv` - Sweep results
- `metrics.json` - Fit metrics
- `selection.jsonl` - Selection records

**Results:**
| q_int | lambda_1 | lambda_int | Delta_lambda | Delta_d_s_max | R^2_exp | R^2_power | Regime |
|-------|----------|------------|--------------|---------------|---------|-----------|--------|
| 0.80 | 15.23 | 142.01 | 126.78 | 0.058 | 0.992 | 0.914 | Rigid |
| 0.40 | 15.23 | 99.01 | 83.77 | 0.224 | 0.960 | 0.028 | Rigid |
| 0.10 | 15.23 | 70.77 | 55.54 | 0.356 | 0.994 | 0.985 | Rigid |
| 0.05 | 15.23 | 63.14 | 47.90 | 0.387 | 0.995 | 0.995 | Rigid |
| 0.02 | 15.23 | 53.51 | 38.28 | 0.414 | 0.996 | 0.999 | Critical |

**Conclusion:** Power-law regime detected at q_int=0.02 where R^2_power >= R^2_exp. Delta-lambda serves as geometric rigidity parameter as predicted.

#### 2.2.3 Gap Tail (`outputs/gap_tail/`)

**Purpose:** Test regime transition via cross-link reduction in two-cluster RGG

**Results:**
| epsilon | lambda_1 | lambda_int | Delta_lambda | Delta_d_s_max | R^2_exp | R^2_power | Regime |
|---------|----------|------------|--------------|---------------|---------|-----------|--------|
| 0.300 | 1.176 | 120.45 | 119.28 | 0.047 | 0.994 | 0.832 | Rigid |
| 0.080 | 0.313 | 120.11 | 119.79 | 0.046 | 0.994 | 0.832 | Rigid |
| 0.020 | 0.078 | 119.88 | 119.81 | 0.047 | 0.994 | 0.832 | Rigid |
| 0.005 | 0.020 | 119.88 | 119.86 | 0.047 | 0.994 | 0.832 | Rigid |

**Conclusion:** Even with epsilon down to 0.005, Delta_lambda remains ~119-120, maintaining rigid regime. No regime transition observed; validates exponential shielding with large spectral gap.

#### 2.2.4 Multi-Axis Gate Tests (`outputs/multi_axis_v*/`)

**Purpose:** Validate Step 2 and Step 3 gates for Framework

**Gate Definitions:**
- **fit:** Power-law fit error < epsilon_fit
- **gluing:** Overlap delta < threshold (k_glue * sqrt(N_geo))
- **UV:** lambda values within UV bounds
- **isolation:** Isolation metric < epsilon_iso
- **selection:** Truth infrastructure (not yet implemented)

**Results:**

| Version | fit | gluing | uv | isolation | selection | Notes |
|---------|-----|--------|-----|-----------|-----------|-------|
| multi_axis_test | FAIL (2.70) | FAIL (0.10) | PASS | PASS | SKIP | Debug run |
| multi_axis_v2 | PASS (0.034) | FAIL (0.10) | PASS | PASS | SKIP | Threshold issue |
| multi_axis_v3 | PASS (0.034) | PASS (0.006) | PASS | PASS | SKIP | All gates pass |
| n64_test | PASS (0.017) | PASS (0.002) | PASS | PASS | SKIP | Large lattice validation |

**Conclusion:** Step 2 gates validated on v3 and n64_test runs. Selection (Step 3) requires truth infrastructure.

#### 2.2.5 Framework Validation Option B (`outputs/framework_validation_b/`)

**Purpose:** Validate monotone spectral filter theorem for general graphs

**Suite ID:** `quick_f6703c3a65`
**Date:** 2026-02-28

**Results:**
| Config | Substrate | Filter | |V| | d_s(min) | d_s(max) | Reduction | Mono | R^2 |
|--------|-----------|--------|-----|----------|----------|-----------|------|-----|
| lattice_D2_N16__hard_cutoff | periodic lattice D2 | hard_cutoff | 256 | 1.72 | 2.01 | 0.30 | PASS | 0.998 |
| lattice_D2_N16__soft_cutoff | periodic lattice D2 | soft_cutoff | 256 | 1.87 | 2.01 | 0.14 | PASS | 0.998 |
| lattice_D2_N16__powerlaw | periodic lattice D2 | powerlaw_reweight | 256 | 1.95 | 2.01 | 0.06 | PASS | 0.998 |
| rgg_D2_n200__powerlaw | RGG D2 | powerlaw_reweight | 200 | 2.60 | 2.81 | 0.21 | PASS | 0.966 |
| sw_n200__powerlaw | small world | powerlaw_reweight | 200 | 0.86 | 0.97 | 0.11 | PASS | 0.867 |

**Conclusion:** Monotone d_s(C) validated for all 5 configurations. Dimension reduction observed in 4/5 runs.

#### 2.2.6 Capacity Dimshift (`outputs/capacity_dimshift/`)

**Purpose:** Core staircase demonstration

**Runs:**
1. `20260214T184445_c56c1c1e3fdc` - Initial run
2. `20260214T191825_9200c99e0f29` - Full sweep

**Key Result (sweep_results.csv):**
Shows clear staircase with d_s plateau values:
- C_geo=0.05: d_s ~ 1.08 (1D active)
- C_geo=0.34: d_s ~ 2.11 (2D transition)
- C_geo=0.67: d_s ~ 2.77 (2D to 3D transition)
- C_geo=1.00: d_s ~ 3.03 (full 3D)

#### 2.2.7 SPX Capacity Probe (`outputs/spx_capacity_probe/`)

**Purpose:** Apply capacity filtering to SPX market data

**Results:**
- 5 crisis dates analyzed: 2008-09-15, 2010-05-07, 2020-03-16, 2022-06-13, 2024-10-04
- Hard and soft cutoff filters compared
- Eigenvalue spectra stored for each date

---

## 3. Key Findings Summary

### 3.1 Framework v4.5/v4.6 Claims Status

| Claim | Tier | Status | Evidence Path |
|-------|------|--------|---------------|
| Factorisation | Algebraic | PASS | test_theorem_validation.py |
| Capacity Staircase | Algebraic | PASS | test_theorem_validation.py |
| Eigenvalue Bounds | Algebraic | PASS | test_theorem_validation.py |
| Capacity-Only | Algebraic | PASS | test_audit.py |
| Mid-sigma Plateau | Asymptotic | PASS | capacity_dimshift outputs |
| Infinite-Lattice Limit | Asymptotic | PASS | test_theorem_validation.py |
| Threshold Location | Empirical | PASS | sweep_results.csv |
| Monotonicity | Empirical | PASS | test_theorem_validation.py |
| Nonseparable Robustness | Extension | PASS | nonseparable_rewire outputs |
| Delta-lambda Rigidity | Extension | VALIDATED | gap_quantile + gap_tail |
| Multi-axis Gates (Step 2) | Validation | PASS | multi_axis_v3, n64_test |
| Selection (Step 3) | Validation | PENDING | Requires truth infrastructure |

### 3.2 Delta-Lambda as Rigidity Parameter

The gap quantile experiment confirms the exponential shielding law:

```
Delta_d_s(sigma) ~ sigma^k * exp(-sigma * Delta_lambda)
```

Where `Delta_lambda = lambda_int - lambda_1`:

- Large Delta_lambda (>50): Rigid regime, exponential decay dominates
- Small Delta_lambda (<50): Transition zone
- Delta_lambda ~ 38: Critical regime where power-law becomes competitive

### 3.3 Step 2 Gates Validation

Latest runs (multi_axis_v3, n64_test) show all gates passing:

| Gate | Threshold | Observed | Status |
|------|-----------|----------|--------|
| fit | < 0.25 | 0.017-0.034 | PASS |
| gluing | < 0.011 | 0.002-0.006 | PASS |
| UV (lambda_1) | < 3.84 | 0.01-0.04 | PASS |
| UV (lambda_int) | < 148 | 7.4 | PASS |
| isolation | < 0.15 | 0.005-0.013 | PASS |

### 3.4 Outstanding Work

1. **Step 3 Selection Gates:** Requires truth infrastructure to evaluate selection records
2. **Additional seeds for nonseparable:** Only N=64, r=0.03 has multi-seed coverage
3. **Full WORKFLOW_AUTO.md execution:** Not yet run end-to-end
4. **Cleanup:** Multiple duplicate directories need archival

---

## 4. Recommendations for Cleanup

### 4.1 Immediate Actions

1. **Archive duplicate repositories:**
   ```bash
   mkdir -p /Users/meganpastore/Clawdbot/archive
   mv /Users/meganpastore/Projects/capacity-demo /Users/meganpastore/Clawdbot/archive/
   mv /Users/meganpastore/Projects/capacity-demo-local /Users/meganpastore/Clawdbot/archive/
   mv /Users/meganpastore/Clawdbot/Repos/framework-recon-clone /Users/meganpastore/Clawdbot/archive/
   ```

2. **Remove old backups:**
   ```bash
   rm -rf /Users/meganpastore/.openclaw_backup_1770997879
   mv /Users/meganpastore/.openclaw-backup-20260221-134754.tar.gz /Users/meganpastore/Clawdbot/archive/
   mv /Users/meganpastore/.openclaw_legacy_20260227_184228 /Users/meganpastore/Clawdbot/archive/
   ```

3. **Clean up .openclaw config backups:** Keep only latest 3

### 4.2 Consolidation Actions

1. **Review Projects/polymarket_arb_bot_v2:** Determine if it has newer code than Repos/polymarket_arb_bot

2. **Merge memory logs:** Consolidate session logs from `Repos/memory/` and `.openclaw/workspace/memory/`

3. **Standardize outputs naming:** Consider renaming output directories with timestamps

### 4.3 Estimated Space Recovery

| Item | Size Estimate | Action |
|------|---------------|--------|
| .openclaw_backup_1770997879 | ~500 MB | Delete |
| Projects/capacity-demo | ~200 MB | Archive |
| Projects/capacity-demo-local | ~150 MB | Archive |
| framework-recon-clone | ~100 MB | Archive |
| **Total** | ~950 MB | |

---

## 5. Next Steps

### Priority 1: Complete Validation
- [ ] Implement truth infrastructure for Step 3 selection gates
- [ ] Run full WORKFLOW_AUTO.md pipeline
- [ ] Add multi-seed coverage for all nonseparable configurations

### Priority 2: Documentation
- [ ] Update MEMORY.md with consolidated findings
- [ ] Create EVIDENCE_SUMMARY.md linking all claims to artifacts
- [ ] Document testing path for new contributors

### Priority 3: Cleanup
- [ ] Execute archival recommendations
- [ ] Clean up openclaw config backups
- [ ] Remove obsolete experiment outputs

---

## Appendix A: Git Status Summary

**Repository:** `Clawdbot/Repos/capacity-demo`
**Branch:** main
**Status:** Clean (no staged changes)

Modified external items (not part of capacity-demo):
- `.openclaw/` session and config files
- `.DS_Store` files
- Submodules: IntoTheUnknown, framework-recon-clone, spectralregime-mvp

---

## Appendix B: File Counts

| Directory | Files | Size |
|-----------|-------|------|
| capacity-demo/outputs/ | ~100 | ~200 MB |
| capacity-demo/dimshift/ | 17 | ~100 KB |
| capacity-demo/scripts/ | 13 | ~100 KB |
| capacity-demo/tests/ | ~10 | ~50 KB |
| capacity-quant/outputs/ | ~20 | ~3 MB |
| spectralregime-mvp/ | ~40 | ~500 KB |

---

*End of Audit Report*