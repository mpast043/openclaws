# Acceptance and quality gates
What must remain true as the code evolves

This file consolidates the “must pass” properties across the demo and extension tracks.

## A. Canonical separable lattice demo

The authoritative list is in docs/THEOREMS.md and is enforced by tests.

Summary of the eight obligations

1. Exact factorisation
   P(σ) equals the product of 1D contributions to machine precision (where brute force is feasible).

2. Capacity staircase
   D_active follows the implied integer staircase from capacity_weights.

3. Eigenvalue bounds
   1D eigenvalues obey [0, 4] bounds with parity aware maximum and spectral symmetry.

4. Capacity only dependence
   C_geo enters only through capacity_weights(C_geo, D). No other dependence is permitted.

5. Mid sigma plateau
   In a fixed plateau window, ds_plateau is near D_active when weights are sufficiently activated (w_d ≥ 0.5).

6. Infinite lattice limit
   The 1D return probability approaches the correct Bessel expression on Z, not on R.

7. Threshold location
   The k + 0.5 crossings occur near C_geo ≈ k/D.

8. Monotonicity
   ds_plateau is non decreasing in C_geo within tolerance.

Any change that affects one of these must update the associated tests.

## B. Step 1 nonseparable Laplacian fork test

Acceptance criteria are defined in EXPERIMENT_NONSEPARABLE_REWIRE.md.

Key checks

1. Monotonicity
   ds_plateau non decreasing in C_geo within mono_tol.
   Mono tolerance is N dependent.

2. Threshold locations
   The D−1 largest jumps occur near C = k/D (k=1..D−1), with max_error bounded and each jump magnitude above a floor.

3. Plateau banding away from transitions
   For points with min_active_weight ≥ 0.5, ds_plateau clusters near integers and has low IQR.

## C. General graph capacity via monotone spectral filters (option B)

Acceptance criteria are defined in FRAMEWORK_THEOREM_B.md.

Key checks

1. Filter range
   0 ≤ g_C(λ) ≤ 1 for all λ.

2. Filter monotonicity
   If C1 ≤ C2 then g_C1(λ) ≤ g_C2(λ) for all λ.

3. No hidden knobs
   Any fixed scales (λ0, steepness fractions, baseline d0) must be computed once per substrate and held constant across the sweep.

4. Empirical scaling window validity
   ln P vs ln σ should be approximately linear in the declared window, and the harness must report R².

5. Monotonicity of extracted plateau d_s in C
   Within tolerance across the sweep.

