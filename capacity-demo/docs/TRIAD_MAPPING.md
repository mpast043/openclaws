# Triadic Mapping for `dimshift`

This note anchors the Framework v4.5 triad `(M, ω, U)` and depth truncation `π_n` to
the concrete objects inside `dimshift/`. It is the canonical reference for future
multi-axis capacity work.

## Triad faces ↔ library objects

| Triad face | Framework meaning | `dimshift` object |
|------------|-------------------|-------------------|
| `M` (distinction) | Algebra of resolvable observables | `eigenvalues_1d(N)` and their tensor-product enumeration inside `run_capacity_sweep`. The eigenvalue table is the discrete Laplacian algebra the experiment can distinguish. |
| `ω` (actuality) | State restricted to the current capacity slice | The log return probabilities `ln_P_matrix` (or `ln_P` per row) computed after applying capacity weights. Each row is the state of the substrate as seen at that capacity. |
| `U` (dynamics) | Evolution / connection between observations | The sigma grid `sigma_values` plus the derivative operator (`np.gradient` in `spectral_dimension`). They encode how probabilities flow with diffusion time, i.e. how observations connect across σ. |

## Depth truncation `π_n` ↔ capacity filter

The truncation from prealgebraic depth to capacity is realised as:

1. **Nominal depth:** `d_nom = C_geo * D` inside `capacity_weights`.
2. **Projection:** `weights = np.clip(d_nom - np.arange(D), 0, 1)` keeps only the first
   `⌈d_nom⌉` triadic layers active; deeper layers are clamped to zero. This is the
   exact analogue of `π_{f(C)}` in Sec. −1.4.
3. **Representation:** `log_return_probability(eigs_1d, weights, sigma)` maps the
   truncated structure back into the observable algebra (`M`, `ω`).

The plateau extraction window is fixed per lattice size (no dependence on `C_geo`),
implying that, operationally, each `run_capacity_sweep` call is a specific depth slice
of the same prealgebraic object.

## Observers as sub-triads

Each capacity step `i` defines an observer-like sub-triad `φ_i` with:

- Algebra `M_i`: the weighted eigenvalue combination encoded by `weights_list[i]`.
- State `ω_i`: the corresponding `ln_P_matrix[i]` (or equivalently the row in
  `ds_matrix`).
- Dynamics `U_i`: the restriction of `sigma_values` to the plateau window for that row.

The shared classical core is the plateau statistic stored in `ds_plateau[i]`, which is
used downstream for comparisons (`sweep_results.csv`).

## Next extensions

- Carry the full capacity vector `(C_geo, C_int, C_gauge, C_ptr, C_obs)` alongside
  each row so non-geometric truncations can share the same plumbing.
- Attach explicit observer metadata (identifier + capacity vector) to each capacity
  row in the artifacts, enabling shared-capacity checks (`min_i C_i`).
- When additional capacity axes are implemented, the `weights` construction becomes
  the geometric component of a larger `F_{\vec C}` map; other axes will feed into
  future filter modules the same way `weights` feeds the Laplacian now.
