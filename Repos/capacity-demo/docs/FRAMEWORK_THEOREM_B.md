# Framework v4.5 — Monotone Spectral Filter Theorem (Option B)

## 1. Definitions

### Substrate operator

Let **G = (V, E)** be a finite connected graph with |V| = n vertices.
The *combinatorial graph Laplacian* is the n x n matrix:

    L = D - A

where D = diag(deg(v)) and A is the adjacency matrix.  L is symmetric
positive semi-definite with eigenvalues:

    0 = λ_0 ≤ λ_1 ≤ ... ≤ λ_{n-1}

The substrate is **fixed**: the graph, its Laplacian, and all eigenvalues
are determined before any capacity sweep begins.

### Monotone spectral filter family

A *monotone spectral filter family* is a collection of functions
{g_C : [0, ∞) → [0, 1]}, indexed by a scalar capacity parameter C ∈ (0, 1],
satisfying:

1. **Range:**  0 ≤ g_C(λ) ≤ 1  for all λ ≥ 0, all C ∈ (0, 1].
2. **Monotonicity in C:**  For C₁ ≤ C₂ and all λ ≥ 0:
       g_{C₁}(λ) ≤ g_{C₂}(λ)
3. **Full capacity:**  g_1(λ) = 1 for all λ  (or approaches 1).

Within a single sweep, **only C varies**.  All other filter parameters
(reference scales λ₀, steepness parameters, baseline dimension d₀) are
fixed per-run and logged in metadata.

### Filtered return probability

Given eigenvalues {λ_i} and filter weights g_C(λ_i), the *filtered return
probability* at diffusion time σ > 0 is:

    P_C(σ) = (1/n) Σ_{i=0}^{n-1} g_C(λ_i) exp(-σ λ_i)

This is the trace of the filtered heat kernel, normalized by |V|.

### Spectral dimension

The *spectral dimension* at scale σ is:

    d_s^C(σ) = -2 d(ln P_C) / d(ln σ)

In practice, this is computed via central finite differences on a
log-spaced σ grid (using `np.gradient`).

---

## 2. Theorem Statement

### Assumptions

**A1 (Weyl-type spectral density).**
In a *scaling window* [λ_lo, λ_hi], the eigenvalue density of the
substrate satisfies a power-law form:

    ρ(λ) ≈ K λ^{d/2 - 1}

where d is the effective (spectral) dimension of the substrate and K > 0
is a constant.  This is exact for periodic lattices (via Weyl's law) and
approximate for other substrates in a finite window.

**A2 (Filter high-λ behavior).**
The filter g_C acts, in the scaling window, like a power-law reweighting:

    g_C(λ) ≈ (λ/λ₀)^{-β(C)}    for λ in [λ_lo, λ_hi]

where β(C) ≥ 0 is a decreasing function of C with β(1) = 0.

**A3 (Scaling window exists).**
There exists a σ-regime where the integral is dominated by eigenvalues
in the scaling window, i.e., contributions from λ < λ_lo and λ > λ_hi
are negligible.

### Statement

**Theorem (Monotone Spectral Filter — Dimension Shift).**

Under assumptions A1–A3, in the corresponding σ scaling window:

    P_C(σ) ≈ A(C) · σ^{-(d - 2β(C))/2}

and therefore:

    d_s^C(σ) ≈ d - 2β(C)

where A(C) is a C-dependent amplitude that does not affect the spectral
dimension (which depends only on the power-law exponent of P_C).

**Consequences:**
- At C = 1: β(1) = 0, so d_s ≈ d (full substrate geometry recovered).
- At C < 1: β(C) > 0, so d_s < d (reduced effective dimension).
- Monotonicity of β(C) in C implies monotonicity of d_s in C.

### Classification

This is a **derived asymptotic fact under stated assumptions**, not a pure
algebraic theorem.  The assumptions (Weyl density, power-law filter behavior,
scaling window existence) must be checked empirically for each substrate.

---

## 3. Proof Sketch

### Step 1: Eigenvalue integral approximation

Replace the discrete sum with an integral using the spectral density:

    P_C(σ) = (1/n) Σ_i g_C(λ_i) e^{-σλ_i}
           ≈ ∫_0^∞ ρ(λ) g_C(λ) e^{-σλ} dλ

### Step 2: Substitute power-law forms

In the scaling window, substitute A1 and A2:

    P_C(σ) ≈ K ∫_{λ_lo}^{λ_hi} λ^{d/2-1} · (λ/λ₀)^{-β(C)} · e^{-σλ} dλ

         = K λ₀^{β(C)} ∫_{λ_lo}^{λ_hi} λ^{d/2-1-β(C)} e^{-σλ} dλ

### Step 3: Laplace transform asymptotics

The integral ∫ λ^{α-1} e^{-σλ} dλ over [λ_lo, λ_hi], when σ is in the
regime where the integrand peaks inside the window, gives:

    ∫ λ^{α-1} e^{-σλ} dλ ≈ Γ(α) σ^{-α}

with α = d/2 - β(C).  This is the standard Laplace transform of a power law.

### Step 4: Extract spectral dimension

Therefore:

    P_C(σ) ≈ K λ₀^{β(C)} Γ(d/2 - β(C)) · σ^{-(d/2 - β(C))}

Taking d_s = -2 d(ln P)/d(ln σ):

    d_s^C(σ) = 2(d/2 - β(C)) = d - 2β(C)     ∎

---

## 4. Implemented Filters

Three monotone spectral filters are implemented in `dimshift/spectral_filters.py`:

### Filter 1: Hard Cutoff (rank-based low-pass)

    m(C) = max(1, floor(C * n))
    g_C(λ_i) = 1 if i < m(C), else 0

Monotone: increasing C increases m, keeping more modes.
This is a step function in eigenvalue rank, not a smooth power law;
the theorem's power-law assumption (A2) is not well-satisfied, but
monotonicity of d_s is still observed empirically.

### Filter 2: Soft Cutoff (logistic taper)

    g_C(i) = sigmoid((C*n - i) / s)

where s = steepness_frac * n (default 0.05*n) is **fixed** per run.
Monotone: increasing C shifts the sigmoid rightward, increasing all weights.
Smoother than hard cutoff but still rank-based, not a clean power law.

### Filter 3: Power-Law Reweight

    β(C)  = (1 - C) * d₀/2
    λ₀    = median nonzero eigenvalue (fixed per substrate)
    g_C(λ) = (1 + λ/λ₀)^{-β(C)}

This is the filter that directly realizes the theorem mechanism.
At C = 1, β = 0, g = 1 everywhere.  At C < 1, high eigenvalues are
suppressed, reducing effective spectral dimension.

**Capacity-only guarantee:** λ₀ and d₀ are computed/set once per substrate
and do not change across the C sweep.

---

## 5. Substrates

Three deterministic substrates are provided in `dimshift/substrates.py`:

| Substrate | Construction | Spectral density | Weyl-type? |
|-----------|-------------|-----------------|------------|
| Periodic lattice (D-dim) | Z_N^D with periodic BC | Exact Weyl: ρ(λ) ~ λ^{D/2-1} | Yes (exactly) |
| Random geometric graph | Points in [0,1]^D, connect if dist < r | Approximate Weyl in bulk | Approximately |
| Small-world (Watts-Strogatz) | Ring + rewired edges | No clean power law | Empirical only |

All substrates are deterministic (fixed seeds logged in metadata).

---

## 6. Validation

### What is checked

The validation harness (`scripts/run_framework_validation_b.py`) runs a
matrix of (substrate × filter × config) and for each sweep:

1. **Monotonicity:** d_s(C) is approximately non-decreasing in C
2. **Dimension reduction:** d_s at low C < d_s at high C
3. **Scaling assumption (R²):** In the declared window, ln P vs ln σ
   is approximately linear (power-law behavior)

### What is mathematically proved

- The filtered return probability P_C(σ) is well-defined for any substrate
  eigenvalues and any filter weights g_C(λ) ∈ [0, 1].
- If the spectral density satisfies a Weyl-type law (A1) and the filter
  behaves as a power-law reweighting (A2) in a scaling window (A3), then
  d_s^C ≈ d - 2β(C) in that window.
- The three implemented filter families are monotone in C (algebraically
  verifiable from their definitions).

### What is empirically validated

- The scaling assumption A1–A3 holds reasonably for periodic lattices
  (R² typically > 0.99 in the plateau window) and approximately for
  RGG (R² > 0.9).
- For small-world graphs, the Weyl-type assumption is not well-satisfied;
  dimension reduction is still observed but the quantitative prediction
  d_s ≈ d - 2β(C) is less accurate.
- Monotonicity of extracted plateau d_s in C holds within ±0.15 tolerance
  across tested configurations.

### What is NOT claimed

- The exact numerical value of d_s at a given C on a given substrate.
- That the scaling window bounds are optimal or derived.
- That the theorem applies to substrates violating A1 (non-Weyl spectra).

---

## 7. Test Coverage

| Test | File | What it checks |
|------|------|----------------|
| `test_filter_monotonicity_*` | test_framework_b.py | g_{C1} ≤ g_{C2} for all filters |
| `test_filter_range` | test_framework_b.py | 0 ≤ g ≤ 1 |
| `test_substrate_determinism_*` | test_framework_b.py | Same seed → identical eigenvalues |
| `test_pipeline_lattice_full_capacity` | test_framework_b.py | d_s ≈ D at C=1 for all filters |
| `test_pipeline_capacity_reduces_dimension` | test_framework_b.py | d_s(C=1) > d_s(C=0.3) |
| `test_no_hidden_knobs_*` | test_framework_b.py | Filter fixed params independent of C |
| `test_scaling_assumption_check` | test_framework_b.py | R² and residuals returned |
| Validation harness | run_framework_validation_b.py | Full matrix sweep with artifacts |
