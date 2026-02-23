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

## Notes

*Last updated: 2026-02-22 19:02 EST*
