# Changelog

All notable changes to the capacity-demo project (Framework v4.5).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-22

### Added

- **Multi-axis capacity vector** `⃗C = (C_geo, C_int, C_gauge, C_ptr, C_obs)`
  - `CapacityVector` dataclass with typed fields and validation
  - `capacity_weights_int()` - Spectral tail weighting above 70th percentile
  - `capacity_weights_combined()` - Multi-axis fusion (multiply/min/sequential modes)
  - `compute_capacity_metrics()` - Gate-compatible metric generation
  - `make_selection_records()` - Selection records for Step 3 gates

- **Multi-axis sweep runner** (`scripts/run_multi_axis_sweep.py`)
  - Grid and paired sweep modes for capacity vectors
  - Full correlator `G_EFT` vs filtered `G_C` computation
  - Fit error: `‖G_EFT - G_C‖₂`
  - Automatic gate metric aggregation
  - JSON outputs for `v45_apply_step2_step3.py` integration

- **Framework Step 2 gate validation** - All gates passing:
  - ✅ fit - correlator fit within adaptive bounds
  - ✅ gluing - overlap `Δ < k/√N_geo`
  - ✅ UV - eigenvalue bounds satisfied
  - ✅ isolation - cross-axis contamination minimal

- **Documentation**
  - `docs/MULTI_AXIS_SUMMARY.md` - Implementation guide and technical details
  - Updated `README.md` with multi-axis examples and project layout

### Verified

- N=32 and N=64 lattice resolutions both pass all gates
- Framework scales correctly with finer lattices (N_geo = N^D)
- GitHub Actions CI continues to pass

## [1.0.0] - 2026-02-21

### Added

- Initial Framework v4.5 implementation
- Capacity-filtered spectral dimension computation
- Single-axis capacity (`C_geo`) with sequential dimension activation
- `dimshift` library with spectral, capacity, and sweep modules
- Web interface (`app.py`) with interactive sliders and live charts
- Command-line experiment runner (`run_capacity_dimshift.py`)
- Robustness suite (`run_robustness_suite.py`)
- Stress test (`run_stress_test.py`)
- Formal theorem validation (8 obligations, 12 tests)
- GitHub Actions CI pipeline (6 jobs)

## Roadmap

### [1.2.0] - Planned

- Step 3 gate validation (selection with truth labels)
- `C_gauge` implementation (symmetry pattern visibility)
- `C_ptr` implementation (pointer state stability)
- `C_obs` implementation (observer inferential resolution)
- Enhanced gate visualization dashboard

## Notes

- C_geo and C_int are active implementations
- C_gauge, C_ptr, C_obs are defined in API but reserved for future work
- Step 3 gates require truth infrastructure (ground truth labels) for full validation
