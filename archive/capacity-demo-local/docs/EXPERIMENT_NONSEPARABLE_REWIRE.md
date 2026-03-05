# Step 1: Non-separable Laplacian Fork Test

## Motivation

The canonical capacity-filtered spectral dimension demo uses a separable
D-dimensional periodic lattice whose return probability factorises exactly:

    P(σ) = Π_d P_1D(w_d σ)

This raises the question: does the capacity-controlled "dimension staircase"
survive when the graph Laplacian is **not** separable?

## Method

### Graph Construction

1. Build a D-dimensional periodic cubic lattice with side N
2. Construct per-dimension Laplacians L_d (d = 1, ..., D)
3. Rewire a fraction r of local edges to random long-range edges
4. The rewired Laplacian is:

       L(C_geo) = Σ_d w_d · L_d + L_rand

   where:
   - w_d = capacity_weights(C_geo, D)[d] — same formula as canonical
   - L_rand is capacity-independent (fixed random long-range edges)
   - L_d have had their rewired local edges removed

### Heat Kernel Trace Estimation

For small graphs (n < 8000): exact eigendecomposition.
For large graphs: Stochastic Lanczos Quadrature (SLQ):
- Rademacher probe vectors (±1)
- Lanczos tridiagonalization → tridiagonal eigendecomposition
- Quadrature: Tr(exp(-σL)) ≈ (1/n_probes) Σ_p v_p^T exp(-σL) v_p

### Key Invariant

**ONLY C_geo varies** across the sweep. The graph structure (L_dims, L_rand),
sigma grid, plateau window, and derivative estimator are all fixed.

## Acceptance Criteria

### A) Monotonicity

ds_plateau must be non-decreasing in C_geo (within tolerance).
- max consecutive drop ≤ mono_tol
- N-dependent: 0.05 (N≥64), 0.08 (N≥32), 0.12 (N<32)

### B) Threshold Locations

The D-1 largest jumps in ds_plateau must occur near C = k/D (k=1,...,D-1).
- Jump detection: find D-1 largest positive differences in ds_plateau
- Check midpoint C of each jump against predicted k/D
- max_error ≤ threshold_tol, each jump magnitude > 0.3

### C) Plateau Banding

Away from transitions (min_active_weight ≥ 0.5), ds_plateau should cluster
near integer values.
- band_frac: fraction of qualifying points within banding_dist_tol of nearest int
- iqr_frac: fraction of qualifying points with plateau IQR below banding_iqr_tol
- Require band_frac ≥ 0.50 AND iqr_frac ≥ 0.50 AND n_qualifying ≥ 3

## Configurations

| Preset | D | N    | n_total  | Rates         | Method |
|--------|---|------|----------|---------------|--------|
| small  | 3 | 16   | 4,096    | 0.01,0.03,0.05| exact  |
| medium | 3 | 32   | 32,768   | 0.01,0.03,0.05| exact  |
| large  | 3 | 64   | 262,144  | 0.01,0.03     | SLQ    |

## Files

- `dimshift/rewire.py` — Vectorized lattice construction + deterministic rewiring
- `dimshift/graph_heat.py` — SLQ trace estimator with exact fallback
- `dimshift/nonseparable.py` — Sweep runner + acceptance criteria
- `scripts/run_nonseparable_rewire_test.py` — CLI entrypoint
- `tests/test_nonseparable_step1.py` — Unit tests

## Results

Results are written to `outputs/nonseparable_rewire/<run_id>/`:
- `metadata.json` — full configuration and criteria
- `sweep_results.csv` — per-C_geo summary
- `thresholds.json` — detected jump locations
- `summary.json` — pass/fail summary

## Key Finding

The dimension staircase is robust to non-separable perturbations.
Random long-range edges boost absolute d_s values (e.g., 3-dim plateau
at d_s ≈ 4.6 instead of 3.0 due to faster diffusion on shortcuts) but
the **staircase structure** — monotone increase with D-1 jumps near
C = k/D — is preserved.
