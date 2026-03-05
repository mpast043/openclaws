# Formal Verification Obligations

This document maps every claim in the capacity-filtered spectral dimension model
to its formal statement, verification function, and test coverage.

## Claim Tiers

| Tier | What it means | Can fail? |
|------|--------------|-----------|
| **Algebraic theorem** | Provable from definitions; exact for all N | Only if code has a bug |
| **Asymptotic fact** | True under stated assumptions (e.g. N large, w_d ≥ 0.5) | If assumptions are violated |
| **Empirical obligation** | Useful check; bounds are fitted, not derived | If the extraction pipeline changes |

---

## Algebraic Theorems

### 1. Exact Factorisation

| | |
|--|--|
| **Statement** | P(σ) = Π_d P_1D(w_d · σ) to machine precision |
| **Why it's exact** | The weighted Laplacian Σ_d w_d λ_1D(k_d) is separable over d, so the exponential and the sum over the product lattice Z_N^D factorise algebraically. |
| **Verification** | `verify_factorisation(D, N, C_geo)` — brute-force N^D enumeration vs factorised product. For N^D > 100,000 sites, returns `algebraic_identity` (too large for brute-force, but true by construction). |
| **Tests** | `test_factorisation_2d` (N=8,16,32 × 4 C_geo values), `test_factorisation_3d` (N=8,16 × 3), `test_factorisation_4d` (N=16, exact enumeration of 65536 sites) |
| **Tolerance** | 1e-12 relative error (machine precision) |
| **Capacity-only** | C_geo enters only through w = capacity_weights(C_geo, D) |

### 2. Capacity Staircase

| | |
|--|--|
| **Statement** | D_active = min(⌈C_geo · D⌉, D) for C_geo > 0; D_active = 0 for C_geo = 0 |
| **Why it's exact** | Follows directly from w_d = clamp(C_geo·D - (d-1), 0, 1). The d-th weight is > 0 iff C_geo·D > d-1, i.e. d < C_geo·D + 1. |
| **Verification** | `verify_staircase(D)` — exhaustive scan of 10,000 C_geo values from 0 to 1 |
| **Tests** | `test_staircase` (D=1..6) |
| **Tolerance** | Exact (integer comparison) |

### 3. Eigenvalue Spectral Bounds

| | |
|--|--|
| **Statement** | λ_k = 2(1 - cos(2πk/N)) ∈ [0, 4]; λ_0 = 0; λ_k = λ_{N-k mod N} |
| **Parity caveat** | Max eigenvalue: N even → 4 exactly (at k=N/2); N odd → 2(1 + cos(π/N)) < 4 (at k=(N-1)/2). The "palindromic" pairing k ↦ k + N/2 only makes sense for even N. |
| **Verification** | `verify_eigenvalue_bounds(N)` — checks zero mode, max eigenvalue (parity-aware), non-negativity, spectral symmetry |
| **Tests** | `test_eigenvalue_bounds` (N=4,7,8,15,16,32,64,128 — both even and odd) |
| **Tolerance** | 1e-12 for max eigenvalue match; 1e-14 for symmetry; 1e-15 for zero mode |

### 4. Capacity-Only Dependence

| | |
|--|--|
| **Statement** | C_geo enters the computation ONLY through capacity_weights(C_geo, D). All other quantities (eigenvalues, σ grid, derivative estimator, plateau window) are functions of fixed parameters (D, N, σ_min, σ_max, n_σ) only. |
| **Verification** | `verify_capacity_only(D, N)` — (a) signature inspection confirms no function except capacity_weights accepts C_geo; (b) manually constructing identical weights and running the pipeline produces results matching within atol=1e-12. |
| **Tests** | `test_capacity_only` (D=2,3,4) |
| **Tolerance** | 1e-12 absolute (not bitwise; robust across platforms/BLAS) |

---

## Asymptotic Facts

### 5. Mid-σ Plateau Dimension

| | |
|--|--|
| **Statement** | In the plateau regime [σ_lo, σ_hi], d_s ≈ D_active within ε(N) → 0 as N → ∞ |
| **Assumptions** | All active weights w_d ≥ 0.5. If violated, returns `assumption_violated` (not a test failure). |
| **Empirical bound** | ε(N) ≲ 6/N + 0.02 (fitted, not derived) |
| **Plateau window** | Uses canonical `SweepConfig.plateau_window()` — same window as the experiment. No custom window inside the theorem. |
| **Verification** | `verify_plateau(D, N, C_geo)` |
| **Tests** | `test_plateau_full_capacity` (C_geo=1.0, 5 configs), `test_plateau_partial_capacity` (C_geo with all w_d ≥ 0.5, 4 configs), `test_plateau_assumption_violated` (confirms rejection for w_d < 0.5) |
| **Capacity-only** | No C_geo-dependent slack. The bound 6/N + 0.02 depends only on N. |

### 6. Infinite-Lattice Limit Convergence

| | |
|--|--|
| **Statement** | P_1D(t; N) → exp(-2t) I₀(2t) as N → ∞ (this is the N→∞ limit on ℤ, NOT the continuum limit on ℝ) |
| **Empirical rate** | Error decreases ~1/N², consistent with Riemann-sum discretisation error. Not formally derived. |
| **Requires** | SciPy (for `scipy.special.i0e`). If missing, returns graceful failure. |
| **Verification** | `verify_continuum_limit(N_values)` — checks errors are non-increasing with N, smallest N has measurable error, largest N has error < 0.01 |
| **Tests** | `test_continuum_limit` (N=16,32,64,128) |
| **Function** | `spectral.p1d_infinite_lattice()` (renamed from `p1d_continuum` to avoid ℝ confusion) |

---

## Empirical Obligations

### 7. Threshold Location

| | |
|--|--|
| **Statement** | d_s crosses k+0.5 near C_geo ≈ k/D (where the (k+1)-th dimension activates) |
| **Why empirical** | The exact crossing depends on the plateau window, finite-difference estimator, and N. The bound δ(N) < 0.1 + 2/N is fitted, not derived. |
| **Verification** | `verify_thresholds(D, N)` — runs a full sweep, finds crossings by interpolation, compares to predicted k/D |
| **Tests** | `test_thresholds` (D=3 N=32,64; D=4 N=32) |

### 8. Monotonicity of d_s in C_geo

| | |
|--|--|
| **Statement** | d_s_plateau(C_geo) is non-decreasing in C_geo within ε < 0.05 |
| **Why empirical** | Monotonicity of the extracted statistic depends on the median estimator and finite-size effects. |
| **Verification** | `verify_monotonicity(D, N)` — runs a full sweep, checks for decreases exceeding 0.05 |
| **Tests** | `test_monotonicity` (D=2,3 N=32; D=3 N=64; D=4 N=32) |

---

## Test Coverage Map

| Verification function | Test file | Test functions | Stress test |
|----------------------|-----------|---------------|-------------|
| `verify_eigenvalue_bounds` | test_theorem_validation.py | `test_eigenvalue_bounds` | Per-config |
| `verify_factorisation` | test_theorem_validation.py | `test_factorisation_{2,3,4}d` | Per-config × C_geo (skip if N^D > 100k) |
| `verify_staircase` | test_theorem_validation.py | `test_staircase` | Per-config |
| `verify_plateau` | test_theorem_validation.py | `test_plateau_{full,partial}_capacity`, `test_plateau_assumption_violated` | Per-config × C_geo (skip if w_d < 0.5) |
| `verify_thresholds` | test_theorem_validation.py | `test_thresholds` | Per-config (D ≥ 2) |
| `verify_monotonicity` | test_theorem_validation.py | `test_monotonicity` | Per-config (D ≥ 2) |
| `verify_continuum_limit` | test_theorem_validation.py | `test_continuum_limit` | Once per suite |
| `verify_capacity_only` | test_theorem_validation.py | `test_capacity_only` | Per-config |

## Invariant

**The capacity-only claim**: within a sweep, nothing changes except C_geo → w.
This is verified algebraically (Theorem 4) and audited structurally (test_audit.py).
No theorem, test, or bound in this module introduces C_geo-dependent parameters.
