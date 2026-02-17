# MEMORY.md

Long-term, curated notes for this workspace.

What this project is
Capacity→Geometry (Framework v4.5–v4.6): a fixed substrate can exhibit different effective geometric observables as observational capacity changes.

Key artifacts (repo)
1. README.md: capacity-weighted lattice demo and the capacity staircase concept.
2. THEOREMS.md: maps claims to formal statements and tests.
3. FRAMEWORK_THEOREM_B.md: monotone spectral filter theorem (general graph extension track).
4. EXPERIMENT_NONSEPARABLE_REWIRE.md: acceptance criteria for the nonseparable Laplacian fork test.

Current focus
1. Step 1: nonseparable Laplacian fork test (rewiring) to check robustness of staircase/plateaus under nonseparable Laplacian.
2. Keep code and tests aligned to formal obligations; avoid silent definition drift.
3. Validate the exponential decay law for Δd_s across substrates and extract Δλ = λ_int − λ_1 as the rigidity knob.

Objectives
1. Explore the science (Capacity→Geometry program) and develop novel products from the discoveries.

Recent turning point (2026-02-12)
1. Confirmed Δd_s(σ) ∼ σ^k e^{-σ(λ_int − λ_1)} in rgg400_q80 window-shift data.
2. Identified Δλ as a geometric rigidity parameter: large gaps enforce exponential shielding of intrinsic IR geometry; small gaps allow capacity-driven deformation.
3. Reframed the core statement: intrinsic IR geometry is substrate-defined; observable geometry is capacity-structured at finite resolution with exponential convergence to IR governed by Δλ.

Gap-tail probe (2026-02-13)
1. Implemented run_gap_tail.py (two-cluster RGG with diminishing cross-links) to test for regime transition as Δλ → 0.
2. With n=400, r=0.30, ε down to 0.005, the interaction sweep remained in the rigid regime: Δλ ≈ 97 for all ε, Δd_s ≈ 0.055, R²_exp ~0.99 » R²_power ~0.81.
3. No stop condition triggered; diagnostic outcome: “No regime transition observed.”

Gap collapse via quantile sweep (2026-02-13)
1. Built run_gap_quantile.py to keep the rgg400 substrate fixed and lower λ_int via q_int ∈ [0.8, 0.02] while holding C_geo, σ window, and interaction filter constant.
2. Δλ shrank from ~127 (q=0.8) down to ~38 (q=0.02); Δd_s grew to ~0.414 inside the fixed window.
3. Stop triggered at q=0.02 because R²_power ≥ R²_exp (0.999 ≥ 0.996), diagnostic: “Power-law regime detected.”
