# Experiment: Capacity → Dimension Shift

**Framework v4.5 — Capacity-Only Falsifiable Demo**

## What This Demonstrates

The same fixed substrate (a periodic cubic lattice) yields different effective
geometry (spectral dimension) when **only** the observational capacity `C_geo`
changes.  Everything else is held fixed: lattice dimension `D`, side length `N`,
lattice spacing `a=1`, diffusion convention, σ grid, derivative estimator,
and plateau-detection window.

## What Is Held Fixed

| Parameter | Value | Description |
|---|---|---|
| `D` | 3 (default) | Lattice dimension |
| `N` | 64 (default) | Sites per dimension |
| `a` | 1 | Lattice spacing |
| Diffusion | Continuous-time heat kernel | `P(σ) = (1/N^D) Σ_k exp(-σ λ_k)` |
| σ grid | `geomspace(0.1, 200, 400)` | Log-spaced diffusion times |
| Derivative | Central finite differences | `np.gradient` on log-log axes |
| Plateau window | `σ ∈ [5, min(0.4 N²/(4π²), σ_max·0.6)]` | Fixed, not C_geo-dependent |

## What Changes

Only `C_geo ∈ [0, 1]`.  This maps to per-dimension weights:

```
d_nom = C_geo × D
w_d = clamp(d_nom - (d-1), 0, 1)    for d = 1, ..., D
```

## How P(σ) and d_s(σ) Are Computed

### Return probability

The capacity filter scales each dimension's Laplacian contribution:

```
Δ_w = Σ_{d=1}^D  w_d · Δ^(d)_1D
```

In eigenvalues:

```
λ_w(k₁,...,k_D) = Σ_d  w_d · 2(1 - cos(2π k_d / N))
```

This factorises exactly:

```
P(σ) = Π_{d=1}^D  P_1D(w_d · σ)
```

where `P_1D(t) = (1/N) Σ_{k=0}^{N-1} exp(-t · λ_1D(k))`.

Computation is done in log space for stability:

```
ln P(σ) = Σ_d  ln P_1D(w_d · σ)
```

### Spectral dimension

```
d_s(σ) = -2 d(ln P) / d(ln σ)
```

Computed via `np.gradient(ln_P, ln_sigma)` (central finite differences,
forward/backward at endpoints).

### Plateau extraction

`d_s_plateau` = median of `d_s(σ)` within the fixed plateau window `[σ_lo, σ_hi]`.
If the window contains fewer than 3 sigma points (very small N), falls back to
the middle third of the sigma grid.

## What the Plots Mean

### 1. Heatmap (Money Plot)

- **x-axis**: `C_geo` (capacity)
- **y-axis**: `log₁₀(σ)` (diffusion time)
- **colour**: `d_s(σ)` (spectral dimension)

Shows how the effective geometry changes across the entire (C_geo, σ) plane.
Horizontal bands of uniform colour indicate plateaus at integer dimensions.

### 2. Representative Curves

`d_s(σ)` vs `log₁₀(σ)` for selected `C_geo` values near the activation
thresholds `k/D`.  Each curve should show a plateau at the corresponding
integer dimension in the plateau window.

### 3. Phase Diagram

`d_s_plateau` (measured plateau `d_s`) vs `C_geo`.  Shows an approximately
staircase structure with smooth transitions near `C_geo ≈ k/D`:
d_s ≈ 1 for `C_geo < 1/D`, then ≈ 2 for `C_geo < 2/D`, etc.
Nominal `d_nom = Σ w_d` is overlaid for comparison.

## How to Run

### Locally

```bash
cd capacity-demo

# Default (3D, N=64)
python scripts/run_capacity_dimshift.py

# Quick test
python scripts/run_capacity_dimshift.py --preset small

# High resolution
python scripts/run_capacity_dimshift.py --preset large

# Custom
python scripts/run_capacity_dimshift.py --D 4 --N 16 --C-steps 40
```

### Web interface

```bash
cd capacity-demo
python app.py
# Open http://localhost:5000
```

### In Colab

Open `notebooks/Capacity_DimShift_v4_5.ipynb` and run all cells.
The notebook installs dependencies and imports from the `dimshift/` package.

## Expected Outputs

```
outputs/capacity_dimshift/<timestamp>_<run_id>/
├── metadata.json              # all fixed parameters, thresholds, timing
├── sweep_results.csv          # C_geo, d_eff, ds_plateau, weights per step
├── ds_matrix.json             # full d_s(C_geo, σ) matrix
├── thresholds.json            # detected dimension thresholds
├── heatmap.png                # d_s heatmap
├── representative_curves.png  # d_s(σ) curves at key capacities
└── phase_diagram.png          # d_eff vs C_geo
```

## Why Plateaus Appear (Analytic Approximation)

This section gives a short derivation showing that integer spectral-dimension
plateaus are a mathematical consequence of the factorised return probability on
a weighted torus — not a fitting artefact.

### Setup

On a 1D periodic lattice of side N, the return probability at diffusion time t
is:

```
P_1D(t) = (1/N) Σ_{k=0}^{N-1} exp(-t · 2(1 - cos(2πk/N)))
```

In the **mid-σ regime** (1 ≪ t ≪ N²/4π²), the sum is well approximated by
the continuum integral (Jacobi theta → Bessel) which gives:

```
P_1D(t) = exp(-2t) I₀(2t)
```

where I₀ is the modified Bessel function of the first kind.  For large t
the asymptotic expansion I₀(x) ~ e^x / √(2πx) yields:

```
P_1D(t) ≈ 1 / √(4πt)
```

so

```
ln P_1D(t) ≈ -½ ln t  +  const
```

### Factorised D-dimensional case

With capacity weights w = (w₁, ..., w_D), the factorisation gives:

```
ln P(σ) = Σ_{d=1}^D  ln P_1D(w_d · σ)
```

Partition the dimensions into **active** (w_d > ε for some small ε)
and **frozen** (w_d ≈ 0).  For frozen dimensions, P_1D(0) = 1 so
ln P_1D(0) = 0.  For active dimensions, the mid-σ approximation gives:

```
ln P(σ) ≈ Σ_{d: w_d > 0}  [ -½ ln(w_d σ) + const ]
         = -½ D_active · ln σ  -  ½ Σ ln(w_d)  +  const
```

where D_active = #{d : w_d > 0}.

### Spectral dimension

Differentiating:

```
d_s(σ) = -2 d(ln P) / d(ln σ)
       = -2 · (-½ D_active)
       = D_active
```

This is **independent of σ** in the plateau regime, and equals the number
of active (non-frozen) dimensions.

### Connection to capacity

The capacity weight formula w_d = clamp(C_geo · D − (d−1), 0, 1) activates
dimensions sequentially:

| C_geo range | Active dimensions | d_s plateau |
|-------------|------------------:|------------:|
| (0, 1/D)   | 1                 | 1           |
| (1/D, 2/D) | 2                 | 2           |
| ...         | ...               | ...         |
| ((D-1)/D, 1]| D                | D           |

At the transition points C_geo = k/D, one new weight crosses from 0 to
positive, adding one more active dimension and raising d_s by 1.  Between
transitions, the weights are positive but change value — however the spectral
dimension is insensitive to the magnitude of positive weights (only their
being non-zero matters), producing the flat plateaus.

### Scope

This derivation **proves plateau behaviour for the weighted-separable torus
model** used in this experiment.  It is not a universal proof of the entire
Framework v4.5 — it demonstrates that the capacity → dimension-shift mechanism
is mathematically sound in this specific setting.  Whether the same mechanism
operates in more general settings (irregular geometries, interacting fields,
non-separable capacity filters) remains an open question.

## Known Limitations

1. **Finite-size effects**: On a finite lattice, `d_s` deviates from exact `D`
   at very small σ (lattice-spacing regime: `σ < 1/(4D)`) and very large σ
   (finite-size regime: `σ > N²/(4π²)` where `P→1/N^D`, `d_s→0`).

2. **Plateau width**: The clean plateau where `d_s ≈ D` requires
   `1 ≪ σ ≪ N²`.  Increasing `N` widens the plateau.

3. **Integer transitions are approximate**: Near `C_geo = k/D`, the transition
   between integer dimensions is smooth over a small C_geo interval, not a
   sharp step.  The width narrows with larger N.

4. **Directional filter model**: The capacity-as-dimensional-weight model is
   one specific implementation of the Framework v4.5 capacity filter.  Other
   implementations (eigenvalue cutoff, soft filter, etc.) may give different
   transition profiles but the same asymptotic integer plateaus.

5. **Not a prediction of specific physics**: This demonstrates the mathematical
   mechanism — that capacity filtering changes effective geometry — not any
   specific physical measurement.

## Robustness Suite

```bash
python scripts/run_robustness_suite.py          # default: 6 (D,N) configs
python scripts/run_robustness_suite.py --quick   # CI-safe: 4 configs, ~10s
```

Runs the same canonical capacity sweep across multiple lattice configurations
(D ∈ {2,3,4}, N ∈ {16,32,64,128}).  Produces a `suite_summary.csv` with
per-run accuracy metrics, threshold locations, and timing.

## Validation

### Core tests (`tests/test_dimshift.py`)

- `d_s` at full capacity matches `D` within 1% for D=1,2,3
- Capacity weights are monotonically non-decreasing
- 3D sweep shows distinct plateaus and detects correct thresholds
- 1D eigenvalues have correct range [0, 4]
- P_1D(0) = 1 (normalisation)

### Audit tests (`tests/test_audit.py`)

Enforce the "no hidden knobs" invariant:

- Plateau window depends only on (N, constants), never on C_geo
- The σ grid is computed once and reused identically for every C_geo value
- C_geo flows only through `capacity_weights()` — no other function accepts it
- Same config → numerically identical results (full determinism)
