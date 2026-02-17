# OBJECTIVES.md

1. Primary outcomes
   - Demonstrate that the Capacity→Geometry staircase persists under nonseparable Laplacian rewiring across D=3 lattices, with Δd_s plateaus aligned to the Framework v4.6 acceptance spec.
   - Extract and track the Δλ = λ_int − λ_1 rigidity knob across all sweeps so the exponential shielding claim stays empirically grounded.
   - Keep the diagnostic pipeline audit-ready: every run must have reproducible configs, artifacts, and derived diagnostics (class_splitting, thresholds, Δd_s fits).

2. Near-term deliverables
   - Finalize the Step 1 nonseparable rewire run set for D=3, N ∈ {16, 32, 64}, r ∈ {0.01…0.08}, including consolidated class_splitting_index.csv with run metadata.
   - Write up the exponential Δd_s(σ) decay verification for rgg400_q80 window-shift data, including Δλ interpretation.
   - Prepare Option B (general-graph filters) scaffolding by documenting required changes in FRAMEWORK_THEOREM_B and enumerating missing tests.

3. What “done” looks like for Step 1 (nonseparable) and the next milestone
   - Step 1 done: all required rewired runs pass Criteria A–C, artifacts + indices are logged, and EXPERIMENT_NONSEPARABLE_REWIRE.md acceptance checklist is satisfied with citations to outputs/* paths.
   - Next milestone: show Δλ-controlled exponential shielding across at least two distinct substrates (lattice + RGG) with a written comparison memo feeding into Framework v4.6 canonical notes.

4. Guardrails
   - Do not change substrate, sigma window, estimators, or seeds mid-sweep unless explicitly documenting the redesign; capacity-only claims require capacity-only variations.
   - No external publication/sharing of artifacts without Megan’s explicit approval.
   - Avoid definition drift: if any observable definition is tweaked, update THEOREMS.md/tests and announce the change.

5. Preferred operating style when uncertain
   - Surface unknowns immediately with a concrete proposal for how to resolve them (experiment, derivation, or documentation request).
   - Bias toward over-documenting provenance (command lines, hashes, parameter grids) rather than summarizing loosely.
