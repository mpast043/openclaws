# Framework v4.5 — Capacity → Geometry Demo

Demonstrates the core Framework v4.5 claim: **the same fixed substrate yields
different effective geometry (spectral dimension) when only observational
capacity changes.**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment (default: 3D lattice, N=64, 30 capacity steps)
python scripts/run_capacity_dimshift.py

# Run multi-axis capacity sweep (C_geo + C_int combined)
python scripts/run_multi_axis_sweep.py --D 3 --N 64 --grid \
  --C-geo 0.5 1.0 --C-int 0.0 0.5 1.0 --output-dir ./outputs/

# Or start the interactive web interface
python app.py
# Then open http://localhost:5000 in your browser
```

## What It Does

A D-dimensional periodic cubic lattice is fixed as the substrate. The single
knob `C_geo` (geometric capacity, from 0 to 1) controls a nominal capacity
budget `d_nom = C_geo * D`, which activates dimensions sequentially:

| C_geo | d_nom = C_geo*D | Active dims (w_d > 0) | Measured d_s (plateau) |
|-------|----------------:|----------------------:|-----------------------:|
| 0.10  |            0.30 |                     1 |                     ~1 |
| 1/3   |            1.00 |                     1 |                     ~1 |
| 0.50  |            1.50 |                     2 |                     ~2 |
| 2/3   |            2.00 |                     2 |                     ~2 |
| 1.00  |            3.00 |                     3 |                     ~3 |

### Multi-Axis Capacity

Framework v4.5 introduces the capacity vector `⃗C = (C_geo, C_int, C_gauge, C_ptr, C_obs)` where:

| Axis | Description | Status |
|------|-------------|--------|
| C_geo | Geometric dimension reconstruction | ✅ Implemented |
| C_int | Interaction tail weighting (spectral gap) | ✅ Implemented |
| C_gauge | Symmetry pattern visibility | 🟡 Reserved |
| C_ptr | Pointer state stability | 🟡 Reserved |
| C_obs | Observer inferential resolution | 🟡 Reserved |

Run multi-axis sweeps with automatic gate validation:

```bash
python scripts/run_multi_axis_sweep.py --D 3 --N 64 --grid \
  --C-geo 0.33 0.66 1.0 --C-int 0.0 0.5 1.0 \
  --output-dir ./outputs/multi_axis
```

Produces gate-compatible metrics for Framework Step 2 (fit, gluing, UV, isolation).
See [docs/MULTI_AXIS_SUMMARY.md](docs/MULTI_AXIS_SUMMARY.md) for implementation details.

Within a scan, nothing else changes — same lattice, same eigenvalues, same
diffusion equation, same analysis pipeline. Only `C_geo` varies across
the sweep; the other controls (`D`, `N`, sigma range) set the fixed
configuration for that scan.

## Three Ways to Run

### 1. Command-line experiment

```bash
python scripts/run_capacity_dimshift.py                  # default (3D, N=64)
python scripts/run_capacity_dimshift.py --preset small   # fast (N=32, 15 steps)
python scripts/run_capacity_dimshift.py --preset large   # high-res (N=128, 50 steps)
python scripts/run_capacity_dimshift.py --D 4 --N 16     # custom 4D lattice
```

Produces outputs in `outputs/capacity_dimshift/<run_id>/`:

```
metadata.json              — all fixed parameters, thresholds, timing
sweep_results.csv          — C_geo, d_eff, ds_plateau, weights per step
ds_matrix.json             — full d_s(C_geo, σ) matrix for reanalysis
thresholds.json            — detected dimension crossings
heatmap.png                — d_s heatmap over (C_geo, σ) — the money plot
representative_curves.png  — d_s(σ) curves at key capacity values
phase_diagram.png          — d_eff vs C_geo staircase
```

### 2. Robustness suite

```bash
# Default suite: 6 runs — (D,N) in {(2,32),(2,64),(2,128),(3,32),(3,64),(4,16)}
python scripts/run_robustness_suite.py

# Quick CI-safe subset (~10s): 4 runs
python scripts/run_robustness_suite.py --quick
```

Produces `outputs/capacity_dimshift_suite/<suite_id>/` with per-run artifacts
plus a `suite_summary.csv` and `suite_metadata.json`.

### 3. Web interface

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000). The web UI provides:

- Interactive sliders for D (1-4), N, C_geo range, and sigma grid (these set the fixed scan configuration)
- Four live charts: d_s vs capacity, d_s(sigma) curves, return probability, nominal vs measured dimension
- Automatic threshold detection with table display
- Optional reference overlays from quantum gravity literature (displayed for context; not used in computation)
- Persistent results -- every run is saved and can be reloaded from the sidebar

### 4. Jupyter notebook (Colab-ready)

Open `notebooks/Capacity_DimShift_v4_5.ipynb`. Run all cells top-to-bottom.
It imports the same `dimshift` library used by the script and web app.

## Tests

```bash
# Core tests (eigenvalues, weights, accuracy, thresholds)
python tests/test_dimshift.py

# Audit tests (no hidden knobs, determinism, sigma-grid invariance)
python tests/test_audit.py

# Formal theorem validation (8 obligations, 12 test functions)
python -m pytest tests/test_theorem_validation.py -v

# Stress test across (D, N) matrix
python scripts/run_stress_test.py --quick       # CI-safe (~30s)
python scripts/run_stress_test.py               # default (~2min)
python scripts/run_stress_test.py --exhaustive  # full matrix (~5min)
```

### Core tests (test_dimshift.py)

Five tests, all run in under 10 seconds:

| Test | What it checks |
|------|----------------|
| `test_eigenvalues_1d` | λ_1D range is [0, 4] |
| `test_p1d_normalization` | P_1D(0) = 1 |
| `test_capacity_weights_monotone_in_C_geo` | Weights are non-decreasing in C_geo |
| `test_spectral_dim_full_capacity_matches_dimension` | d_s matches D within 1% for D=1,2,3 at C_geo=1 |
| `test_dimshift_thresholds_smoke` | 3D sweep detects 1→2 and 2→3 transitions near k/D |

### Audit tests (test_audit.py)

Four tests enforcing "no hidden knobs":

| Test | What it checks |
|------|----------------|
| `test_plateau_window_independent_of_C_geo` | Window depends on N/constants, never C_geo |
| `test_sigma_grid_constant_across_sweep` | Sigma grid is identical for all C_geo values |
| `test_capacity_only_dependency` | C_geo flows only through capacity_weights() |
| `test_determinism` | Same config produces numerically identical results |

### Theorem validation (test_theorem_validation.py)

Twelve tests covering 4 algebraic theorems, 2 asymptotic facts, and 2 empirical
obligations (see [docs/THEOREMS.md](docs/THEOREMS.md) for the full mapping):

| Test | Tier | What it checks |
|------|------|----------------|
| `test_eigenvalue_bounds` | Algebraic | λ ∈ [0, 4], parity-correct max, symmetry (even+odd N) |
| `test_factorisation_2d` | Algebraic | Brute-force vs factorised match to machine precision (2D) |
| `test_factorisation_3d` | Algebraic | Same for 3D lattices |
| `test_factorisation_4d` | Algebraic | Same for 4D (exact enumeration, 65536 sites) |
| `test_staircase` | Algebraic | D_active = min(⌈C_geo·D⌉, D) for D=1..6 |
| `test_plateau_full_capacity` | Asymptotic | d_s ≈ D within 6/N+0.02 at C_geo=1 |
| `test_plateau_partial_capacity` | Asymptotic | d_s ≈ D_active (only configs with w_d ≥ 0.5) |
| `test_plateau_assumption_violated` | Asymptotic | Correctly rejects configs with w_d < 0.5 |
| `test_thresholds` | Empirical | d_s crosses k+0.5 near C_geo ≈ k/D |
| `test_monotonicity` | Empirical | d_s non-decreasing in C_geo (within 0.05) |
| `test_continuum_limit` | Asymptotic | P_1D(t;N) → exp(-2t)I₀(2t) as N→∞ (infinite lattice, not ℝ) |
| `test_capacity_only` | Algebraic | C_geo flows only through capacity_weights (allclose, not bitwise) |

### Stress test (run_stress_test.py)

Runs all theorems across a grid of (D, N) configurations with multiple C_geo values.
Outputs CSV results and JSON metadata to `outputs/stress_test/<suite_id>/`.

| Mode | Configs | Dimensions | Lattice sizes |
|------|--------:|:-----------|:--------------|
| `--quick` | 5 | D=2,3,4 | N=32,64 |
| default | 7 | D=2,3,4 | N=32,64,128 |
| `--exhaustive` | 12 | D=1..5 | N=32,64,128 |

## Project Layout

```
capacity-demo/
├── .github/workflows/
│   └── capacity-demo.yml        # CI: 6 jobs (core, audit, theorem, robustness, stress, canonical)
├── dimshift/                    # Library package (all computation)
│   ├── __init__.py              #   Package exports
│   ├── capacity.py              #   capacity_weights()
│   ├── spectral.py              #   eigenvalues_1d(), p1d(), log_return_probability(),
│   │                            #     return_probability(), spectral_dimension()
│   ├── sweep.py                 #   SweepConfig, SweepResult, run_capacity_sweep(),
│   │                            #     write_artifacts()
│   ├── multi_axis_capacity.py   #   CapacityVector, capacity_weights_combined(),
│   │                            #     compute_capacity_metrics(), make_selection_records()
│   ├── plotting.py              #   plot_heatmap(), plot_representative_curves(),
│   │                            #     plot_phase_diagram(), save_all_figures()
│   └── theorem.py               #   8 formal theorems with machine-checkable verification
├── scripts/
│   ├── run_capacity_dimshift.py # Canonical experiment entrypoint
│   ├── run_multi_axis_sweep.py  # Multi-axis capacity sweep with gates
│   ├── run_robustness_suite.py  # Multi-(D,N) robustness suite
│   └── run_stress_test.py       # Stress test across (D,N) matrix
├── notebooks/
│   └── Capacity_DimShift_v4_5.ipynb  # Colab-ready notebook
├── tests/
│   ├── test_dimshift.py         # Core validation tests
│   ├── test_audit.py            # No-hidden-knobs audit tests
│   └── test_theorem_validation.py  # Formal theorem validation (12 tests)
├── docs/
│   ├── EXPERIMENT_CAPACITY_DIMSHIFT.md  # Detailed experiment documentation
│   ├── MULTI_AXIS_SUMMARY.md    # Multi-axis capacity implementation guide
│   └── THEOREMS.md              # Theorem-to-test mapping and tier classification
├── app.py                       # Flask web server (imports dimshift)
├── static/
│   └── index.html               # Web UI
├── outputs/                     # Experiment artifacts go here
├── results/                     # Web interface persistence
└── requirements.txt             # numpy, matplotlib, flask
```

## Mathematical Method

**Spectral dimension** measures the effective dimensionality a diffusion process
"sees" on a geometry:

```
d_s(σ) = -2 d(ln P(σ)) / d(ln σ)
```

where P(σ) is the return probability of a random walker after diffusion time σ.

In the mid-sigma regime (between lattice-spacing discretization and finite-size
saturation), d_s(sigma) approaches D on a D-dimensional lattice. The capacity
filter scales each dimension's Laplacian contribution by a weight w_d, which
factorises the computation:

```
P(σ) = Π_{d=1}^D  P_1D(w_d · σ)
```

This is computed in log-space for numerical stability, with O(D · N · M)
total cost (N = lattice side, M = number of σ points). See
[docs/EXPERIMENT_CAPACITY_DIMSHIFT.md](docs/EXPERIMENT_CAPACITY_DIMSHIFT.md)
for the full derivation.

## Verification Obligations

The `dimshift/theorem.py` module defines 8 verification obligations in three
tiers. See [docs/THEOREMS.md](docs/THEOREMS.md) for the full mapping to tests.

| # | Tier | Obligation | Statement (abbreviated) |
|---|------|-----------|------------------------|
| 1 | Algebraic | Exact Factorisation | P(σ) = Π P_1D(w_d·σ) to machine precision |
| 2 | Algebraic | Capacity Staircase | D_active = min(⌈C_geo·D⌉, D) for C_geo > 0 |
| 3 | Algebraic | Eigenvalue Bounds | λ_k ∈ [0, 4]; parity-aware max; spectral symmetry |
| 4 | Algebraic | Capacity-Only | C_geo enters only through capacity_weights() |
| 5 | Asymptotic | Mid-σ Plateau | d_s ≈ D_active within O(1/N), requires w_d ≥ 0.5 |
| 6 | Asymptotic | Infinite-Lattice Limit | P_1D → exp(-2t)I₀(2t) as N→∞ on ℤ (not ℝ) |
| 7 | Empirical | Threshold Location | d_s = k+0.5 crossing near C_geo ≈ k/D |
| 8 | Empirical | Monotonicity | d_s(plateau) non-decreasing in C_geo |

```python
from dimshift.theorem import verify_all
results = verify_all(D=3, N=64)
for r in results:
    print(f"{r.name}: {'PASS' if r.passed else 'FAIL'}")
```

## CI

GitHub Actions runs 6 jobs on every push/PR touching `capacity-demo/`:

| Job | What it runs |
|-----|-------------|
| `core-tests` | test_dimshift.py (Python 3.10, 3.11, 3.12) |
| `audit-tests` | test_audit.py |
| `theorem-validation` | test_theorem_validation.py (all 12 tests) |
| `robustness-suite` | run_robustness_suite.py --quick |
| `stress-test` | run_stress_test.py --quick |
| `canonical-run` | run_capacity_dimshift.py --preset small |

## Dependencies

- Python 3.9+
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7
- flask >= 3.0 (for web interface only)
