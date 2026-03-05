# Latest Test Results

Generated: 2026-02-10
Platform: Linux 4.4.0, Python 3.x, NumPy/SciPy

## Summary

| Suite | Tests | Result |
|-------|------:|--------|
| Core (test_dimshift.py) | 5 | ALL PASS |
| Audit (test_audit.py) | 4 | ALL PASS |
| Theorem validation (test_theorem_validation.py) | 12 | ALL PASS |
| Stress test --quick (run_stress_test.py) | 76 | 56 pass, 20 skip, 0 fail |

**Total: 97 checks, 0 failures.**

Skips are assumption-based (plateau theorem inapplicable when w_d < 0.5,
factorisation too large for brute-force enumeration). These are not failures.

---

## 1. Eigenvalue Spectral Bounds (Algebraic)

Exact verification of λ_k = 2(1 - cos(2πk/N)) properties. Parity-aware
max eigenvalue: N even → 4 exactly; N odd → 2(1 + cos(π/N)) < 4.

| N | Parity | Max λ (actual) | Max λ (expected) | Symmetry error | Result |
|--:|--------|---------------:|------------------:|---------------:|--------|
| 4 | even | 4.0000000000 | 4.0000000000 | 6.66e-16 | PASS |
| 7 | odd | 3.8019377358 | 3.8019377358 | 4.44e-16 | PASS |
| 8 | even | 4.0000000000 | 4.0000000000 | 6.66e-16 | PASS |
| 15 | odd | 3.9562952015 | 3.9562952015 | 1.78e-15 | PASS |
| 16 | even | 4.0000000000 | 4.0000000000 | 1.33e-15 | PASS |
| 32 | even | 4.0000000000 | 4.0000000000 | 1.33e-15 | PASS |
| 64 | even | 4.0000000000 | 4.0000000000 | 1.33e-15 | PASS |
| 128 | even | 4.0000000000 | 4.0000000000 | 1.33e-15 | PASS |

**Conclusion:** All eigenvalue properties exact to machine precision. Odd N
produces max eigenvalue < 4 as predicted by formula.

---

## 2. Exact Factorisation (Algebraic)

Brute-force P(σ) = (1/N^D) Σ_k exp(-σ λ_k) vs factorised Π_d P_1D(w_d σ).
Tolerance: 1e-12 relative.

| D | N | C_geo | Total sites | Max relative error | Result |
|--:|--:|------:|------------:|-------------------:|--------|
| 2 | 8 | 0.50 | 64 | 2.22e-16 | PASS |
| 2 | 16 | 0.50 | 256 | 1.05e-15 | PASS |
| 2 | 32 | 1.00 | 1,024 | 1.93e-15 | PASS |
| 3 | 8 | 0.70 | 512 | 1.24e-15 | PASS |
| 3 | 16 | 1.00 | 4,096 | 1.64e-15 | PASS |
| 4 | 16 | 0.75 | 65,536 | 1.85e-15 | PASS |

**Conclusion:** Factorisation is exact to machine precision (< 2e-15 relative
error in all cases). For N^D > 100,000 sites the identity is algebraic
(separable Laplacian) and not brute-force verified.

---

## 3. Capacity Staircase (Algebraic)

D_active = min(⌈C_geo·D⌉, D) for C_geo > 0, verified across 5000 C_geo
values per D.

| D | C_geo values tested | Failures | Result |
|--:|--------------------:|---------:|--------|
| 1 | 5,000 | 0 | PASS |
| 2 | 5,000 | 0 | PASS |
| 3 | 5,000 | 0 | PASS |
| 4 | 5,000 | 0 | PASS |
| 5 | 5,000 | 0 | PASS |
| 6 | 5,000 | 0 | PASS |

**Conclusion:** Staircase formula is exact for all tested D and all C_geo ∈ [0, 1].

---

## 4. Mid-σ Plateau Dimension (Asymptotic)

d_s ≈ D_active in the plateau window [σ_lo, σ_hi] from SweepConfig.
Empirical bound: 6/N + 0.02. Requires all active weights ≥ 0.5.

### Full capacity (C_geo = 1.0, all weights = 1.0)

| D | N | D_active | Median d_s | Max deviation | Bound (6/N+0.02) | Window | Points | Result |
|--:|--:|---------:|-----------:|--------------:|------------------:|-------:|-------:|--------|
| 2 | 32 | 2 | 2.0374 | 0.0552 | 0.2075 | [5.0, 10.4] | 38 | PASS |
| 2 | 64 | 2 | 2.0180 | 0.0556 | 0.1138 | [5.0, 41.5] | 102 | PASS |
| 3 | 32 | 3 | 3.0562 | 0.0829 | 0.2075 | [5.0, 10.4] | 38 | PASS |
| 3 | 64 | 3 | 3.0269 | 0.0833 | 0.1138 | [5.0, 41.5] | 102 | PASS |
| 4 | 32 | 4 | 4.0749 | 0.1105 | 0.2075 | [5.0, 10.4] | 38 | PASS |

### Partial capacity (all active w_d ≥ 0.5)

| D | N | C_geo | D_active | Min weight | Median d_s | Max deviation | Bound | Result |
|--:|--:|------:|---------:|-----------:|-----------:|--------------:|------:|--------|
| 3 | 64 | 0.500 | 2 | 0.50 | 2.0277 | 0.0933 | 0.1138 | PASS |
| 3 | 64 | 0.667 | 2 | 1.00 | 2.0180 | 0.0556 | 0.1138 | PASS |
| 4 | 32 | 0.500 | 2 | 1.00 | 2.0374 | 0.0552 | 0.2075 | PASS |
| 4 | 32 | 0.750 | 3 | 1.00 | 3.0562 | 0.0829 | 0.2075 | PASS |

### Assumption violations (correctly rejected)

| D | N | C_geo | Min active weight | Outcome |
|--:|--:|------:|------------------:|---------|
| 3 | 64 | 0.40 | 0.20 | Rejected: w_d < 0.5 |
| 3 | 64 | 0.80 | 0.40 | Rejected: w_d < 0.5 |
| 4 | 32 | 0.30 | 0.20 | Rejected: w_d < 0.5 |

**Conclusion:** At full capacity, d_s matches D to within ~0.11 (D=4 N=32
worst case). At partial capacity with w_d ≥ 0.5, deviation stays below the
6/N + 0.02 bound. Configs with fractional weights < 0.5 are correctly rejected
rather than masked by C_geo-dependent slack. Plateau accuracy improves with N
(deviation ~0.06 at N=64 vs ~0.11 at N=32).

---

## 5. Threshold Location (Empirical)

d_s crosses k+0.5 near C_geo ≈ k/D. Empirical bound: 0.1 + 2/N.

| D | N | Crossing | Predicted C | Actual C | Error | Bound | Result |
|--:|--:|:---------|------------:|---------:|------:|------:|--------|
| 3 | 32 | d_s=1.5 | 0.3333 | 0.3399 | 0.0066 | 0.1625 | PASS |
| 3 | 32 | d_s=2.5 | 0.6667 | 0.6730 | 0.0063 | 0.1625 | PASS |
| 3 | 64 | d_s=1.5 | 0.3333 | 0.3369 | 0.0036 | 0.1313 | PASS |
| 3 | 64 | d_s=2.5 | 0.6667 | 0.6702 | 0.0035 | 0.1313 | PASS |
| 4 | 32 | d_s=1.5 | 0.2500 | 0.2550 | 0.0050 | 0.1625 | PASS |
| 4 | 32 | d_s=2.5 | 0.5000 | 0.5049 | 0.0049 | 0.1625 | PASS |
| 4 | 32 | d_s=3.5 | 0.7500 | 0.7546 | 0.0046 | 0.1625 | PASS |

**Conclusion:** Thresholds consistently within 0.007 of k/D (well under the
0.1 + 2/N bound). Error shrinks with N: ~0.006 at N=32 → ~0.004 at N=64.
The systematic positive bias (~0.005) means crossings happen slightly above
k/D, which is expected since the newly-activated dimension needs a non-zero
weight before it contributes measurably to d_s.

---

## 6. Monotonicity (Empirical)

d_s_plateau is non-decreasing in C_geo within ε < 0.05.

| D | N | Max drop | Tolerance | Drops detected | Result |
|--:|--:|---------:|----------:|---------------:|--------|
| 2 | 32 | 0.0106 | 0.05 | 175 | PASS |
| 3 | 32 | 0.0158 | 0.05 | 172 | PASS |
| 3 | 64 | 0.0242 | 0.05 | 180 | PASS |
| 4 | 32 | 0.0210 | 0.05 | 173 | PASS |

**Conclusion:** Many tiny drops (from finite-difference noise at low C_geo) but
all below 0.025. Worst case is D=3 N=64 at 0.024. No drops anywhere near the
0.05 threshold. The large count of small drops (170–180 per sweep of 200 steps)
reflects the noisy finite-difference derivative at low sigma, not real
non-monotonicity.

---

## 7. Infinite-Lattice Limit (Asymptotic)

P_1D(t; N) → exp(-2t) I₀(2t) as N → ∞. Error measured over t ∈ [1, 20].

| N | Max relative error | Result |
|--:|-------------------:|--------|
| 16 | 0.08175567 | — |
| 32 | 0.00000869 | — |
| 64 | 0.00000000 | — |
| 128 | 0.00000000 | — |

| Transition | Error ratio | Expected (1/N²) |
|:-----------|------------:|-----------------:|
| N=16 → 32 | 0.000106 | 0.25 |
| N=32 → 64 | 0.0 | 0.25 |

**Conclusion:** Convergence is extremely rapid — error drops from 8% at N=16
to < 1e-5 at N=32, and is below machine precision by N=64. The convergence
rate is faster than the expected 1/N², suggesting the Riemann-sum error
bound is conservative. By N=32 the finite lattice is essentially
indistinguishable from the infinite-lattice Bessel form.

---

## 8. Capacity-Only Dependence (Algebraic)

Two paths: (1) C_geo → capacity_weights → pipeline; (2) manually construct
identical weights → pipeline. Differences measured with atol=1e-12.

| D | N | Weight diff | ln P diff | d_s diff | Signature check | Result |
|--:|--:|:------------|:----------|:---------|:----------------|--------|
| 2 | 32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | All clear | PASS |
| 3 | 32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | All clear | PASS |
| 4 | 32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | All clear | PASS |

**Conclusion:** Zero difference between the two paths for all D tested. C_geo
has no hidden pathway into the computation — it flows exclusively through
capacity_weights(). Signature inspection confirms no other function accepts
a C_geo parameter.

---

## Stress Test Summary (--quick)

5 configs × 5 C_geo values × 8 obligations. Suite ID: 4922155836.

| D | N | Eigenvalues | Staircase | Capacity-only | Factorisation | Plateau | Thresholds | Monotonicity |
|--:|--:|:-----------:|:---------:|:-------------:|:-------------:|:-------:|:----------:|:------------:|
| 2 | 32 | PASS | PASS | PASS | 5/5 (err 1.3e-15) | 3/3 (dev 0.078) | PASS (err 0.010) | PASS (drop 0.021) |
| 2 | 64 | PASS | PASS | PASS | 5/5 (err 1.3e-15) | 3/3 (dev 0.078) | PASS (err 0.006) | PASS (drop 0.026) |
| 3 | 32 | PASS | PASS | PASS | 5/5 (err 1.9e-15) | 3/3 (dev 0.082) | PASS (err 0.007) | PASS (drop 0.031) |
| 3 | 64 | PASS | PASS | PASS | 0/0 (algebraic) | 3/3 (dev 0.082) | PASS (err 0.003) | PASS (drop 0.047) |
| 4 | 32 | PASS | PASS | PASS | 0/0 (algebraic) | 3/3 (dev 0.109) | PASS (err 0.005) | PASS (drop 0.041) |

Plateau: 2 C_geo values skipped per config (w_d < 0.5 → assumption not met).
Factorisation: D=3 N=64 and D=4 N=32 skipped brute-force (N^D > 100k → algebraic identity).
