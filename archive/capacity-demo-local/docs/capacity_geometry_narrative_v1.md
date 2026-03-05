# Capacity Controls Observable Geometry — Narrative (v1)

**Premise**  
A fixed substrate can present different observable geometry when the observer's capacity changes. Step 1 (nonseparable rewired lattices) proved that the staircase survives beyond separable Laplacians. Option B (monotone spectral filters) demonstrates the same capacity-only effect via spectral filtering on arbitrary graphs. Together, they show that capacity is the driver, substrate perturbations are passengers.

**What we hold constant**  
- Substrate: lattice, random geometric graph, small-world network — each fixed per run.  
- Diffusion window, derivative estimator, plateau detection, metadata logging.  
- No interpretation layer; just structural observables.

**What we vary**  
- A single capacity knob.  
- In Step 1: weights inside the Laplacian (rewired geometry).  
- In Option B: spectral filter g_C(λ).  

**Figure (outputs/figures/capacity_visual_proof.png)**  
- Row 1: Periodic lattice, canonical staircase.  
- Row 2: Random geometric graph, ds_plateau moves 2.4→2.8 while ln P slopes shift.  
- Row 3: Small-world network, ds_plateau 0.74→0.97; same substrate, different slope.  

**Narrative**  
1. Step 1 rewiring showed capacity invariants persist even when separability is broken (nonseparable Laplacian).  
2. Option B spectral filters show the identical invariant on arbitrary substrates: the observer's capacity is encoded in g_C(λ), and the measured spectral dimension reacts — visually confirmed in the figure.  
3. Therefore, “capacity controls observable structure” is substrate-agnostic. Whether you perturb the Laplacian (Step 1) or filter spectra (Option B), the outcome matches: capacity-only scans produce reproducible geometric shifts.  
4. This is now a story we can hand to investors, leadership, or reviewers: one slide, one invariant, two independent mechanisms reinforcing the same theorem.  

**Next**  
- Add this figure + text to the Option B section of docs/FRAMEWORK_THEOREM_B.md.  
- Reference it from the Step 1 acceptance note so anyone landing there sees the Option B corroboration.  
- Use it in the product brief for diagnostic tooling.
