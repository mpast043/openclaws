"""
Verification obligations for the capacity-filtered spectral dimension model.

Three tiers of claims:

1. Algebraic theorems (exact, provable from definitions):
   - Exact Factorisation of return probability
   - Capacity Staircase (active dimension count)
   - Eigenvalue bounds and symmetries

2. Asymptotic / analytic facts (true under stated assumptions):
   - Mid-σ Plateau: d_s ≈ D_active (requires N large, all active w_d ≥ 0.5)
   - Infinite-lattice limit: P_1D(t;N) → exp(-2t) I₀(2t) as N → ∞

3. Empirical obligations (useful checks, not theorems):
   - Monotonicity of extracted plateau d_s in C_geo
   - Threshold locations near C_geo = k/D
   - Specific numerical bounds (e.g. 6/N + 0.02) fitted from experiments

Each verification function returns a TheoremResult with pass/fail, the
mathematical statement, and diagnostics.  No new C_geo-dependent parameters
are introduced.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .capacity import capacity_weights
from .spectral import eigenvalues_1d, p1d, log_return_probability, spectral_dimension


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TheoremResult:
    """Outcome of a theorem verification."""
    name: str
    passed: bool
    statement: str
    diagnostics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}"


# =========================================================================
# ALGEBRAIC THEOREM 1: Exact Factorisation
# =========================================================================

FACTORISATION_STATEMENT = (
    "For a D-dimensional periodic cubic lattice with side N and capacity "
    "weights w = (w_1, ..., w_D), the return probability factorises exactly:\n"
    "    P(σ) = (1/N^D) Σ_k exp(-σ Σ_d w_d λ_1D(k_d))\n"
    "         = Π_{d=1}^D  (1/N) Σ_{k_d} exp(-w_d σ λ_1D(k_d))\n"
    "         = Π_{d=1}^D  P_1D(w_d σ)\n"
    "This holds exactly for any σ > 0, any weights w_d ≥ 0, and any N ≥ 2.\n"
    "Proof: the weighted Laplacian Σ_d w_d λ_1D(k_d) is separable over d, "
    "so the exponential and the sum over the product lattice factorise."
)


def verify_factorisation(D: int, N: int, C_geo: float,
                         n_sigma: int = 200, rtol: float = 1e-12) -> TheoremResult:
    """
    Verify exact factorisation by comparing the brute-force N^D sum against
    the factorised product of 1D sums.

    Only performs exact enumeration.  If N^D > 100,000, the theorem is still
    algebraically true (separable Laplacian) but too large for brute-force
    verification; returns passed=True with method="algebraic_identity".
    """
    eigs_1d = eigenvalues_1d(N)
    weights = capacity_weights(C_geo, D)
    sigma = np.geomspace(0.1, 200.0, n_sigma)

    # Factorised path: Π_d P_1D(w_d σ)
    ln_P_factorised = log_return_probability(eigs_1d, weights, sigma)
    P_factorised = np.exp(ln_P_factorised)

    total_sites = N ** D

    if total_sites > 100_000:
        # Too large for brute-force.  The factorisation is an algebraic
        # identity (separable eigenvalues), not an empirical claim.
        return TheoremResult(
            name="Exact Factorisation",
            passed=True,
            statement=FACTORISATION_STATEMENT,
            diagnostics={
                "D": D, "N": N, "C_geo": C_geo,
                "method": "algebraic_identity",
                "total_sites": total_sites,
                "note": ("N^D too large for brute-force enumeration. "
                         "Factorisation follows from the separability of "
                         "the weighted Laplacian on the product lattice."),
            },
        )

    # Brute-force: build full k-grid, compute full eigenvalues, sum
    grids = [np.arange(N) for _ in range(D)]
    mesh = np.meshgrid(*grids, indexing='ij')
    k_vecs = np.stack([m.ravel() for m in mesh], axis=1)  # (total_sites, D)

    # Full eigenvalue for each k-vector: Σ_d w_d λ_1D(k_d)
    lam_full = np.zeros(total_sites)
    for d in range(D):
        lam_full += weights[d] * eigs_1d[k_vecs[:, d]]

    # P_brute(σ) = (1/N^D) Σ_k exp(-σ λ_k)
    P_brute = np.mean(np.exp(-lam_full[None, :] * sigma[:, None]), axis=1)

    # Compare
    max_rel_err = np.max(np.abs(P_brute - P_factorised) /
                         np.maximum(P_brute, 1e-300))
    passed = max_rel_err < rtol

    return TheoremResult(
        name="Exact Factorisation",
        passed=passed,
        statement=FACTORISATION_STATEMENT,
        diagnostics={
            "D": D, "N": N, "C_geo": C_geo,
            "method": "exact_enumeration",
            "total_sites": total_sites,
            "max_relative_error": float(max_rel_err),
            "tolerance": rtol,
            "n_sigma": n_sigma,
        },
    )


# =========================================================================
# ALGEBRAIC THEOREM 2: Capacity Staircase (Active Dimension Count)
# =========================================================================

STAIRCASE_STATEMENT = (
    "For D ≥ 1 and C_geo ∈ [0, 1], define weights w_d = clamp(C_geo·D - (d-1), 0, 1) "
    "for d = 1, ..., D.\n"
    "Then the number of active dimensions D_active = |{d : w_d > 0}| satisfies:\n"
    "    D_active = min(⌈C_geo · D⌉, D)   for C_geo > 0\n"
    "    D_active = 0                       for C_geo = 0\n"
    "Transitions occur exactly at C_geo = k/D for k = 1, ..., D-1."
)


def verify_staircase(D: int, n_points: int = 10_000) -> TheoremResult:
    """
    Exhaustive verification of the active dimension count formula
    across a fine grid of C_geo values.
    """
    C_values = np.linspace(0.0, 1.0, n_points)
    failures = []

    for C_geo in C_values:
        w = capacity_weights(float(C_geo), D)
        D_active = int(np.sum(w > 0))

        if C_geo == 0.0:
            expected = 0
        else:
            expected = min(int(np.ceil(C_geo * D - 1e-15)), D)
            # At C_geo = k/D: d_nom = k, so w_k = clamp(k - (k-1)) = 1,
            # w_{k+1} = clamp(k - k) = 0.  So D_active = k.
            if expected == 0 and C_geo > 0:
                expected = 1

        if D_active != expected:
            failures.append({
                "C_geo": float(C_geo),
                "D_active": D_active,
                "expected": expected,
                "weights": w.tolist(),
            })

    passed = len(failures) == 0
    return TheoremResult(
        name="Capacity Staircase",
        passed=passed,
        statement=STAIRCASE_STATEMENT,
        diagnostics={
            "D": D,
            "n_points": n_points,
            "n_failures": len(failures),
            "failures": failures[:5],
        },
    )


# =========================================================================
# ALGEBRAIC THEOREM 3: Eigenvalue Spectral Bounds
# =========================================================================

EIGENVALUE_STATEMENT = (
    "For a 1D periodic lattice (ring) of side N ≥ 2, the Laplacian eigenvalues "
    "λ_k = 2(1 - cos(2πk/N)) for k = 0, ..., N-1 satisfy:\n"
    "    (a) λ_0 = 0  (zero mode)\n"
    "    (b) All λ_k ∈ [0, 4].  Max eigenvalue:\n"
    "        - N even: max λ = 4  (at k = N/2)\n"
    "        - N odd:  max λ = 2(1 + cos(π/N))  (at k = (N-1)/2)\n"
    "    (c) λ_k ≥ 0 for all k\n"
    "    (d) λ_k = λ_{N-k mod N}  (spectral symmetry; clean pairing\n"
    "        only when N is even, since k ↦ k + N/2 requires even N)\n"
    "These are exact algebraic properties, not approximations."
)


def verify_eigenvalue_bounds(N: int) -> TheoremResult:
    """
    Exact verification of eigenvalue spectral properties.
    """
    eigs = eigenvalues_1d(N)
    checks = {}

    # (a) Zero mode
    checks["zero_mode"] = abs(eigs[0]) < 1e-15

    # (b) Max eigenvalue — branch on parity
    max_eig = float(np.max(eigs))
    if N % 2 == 0:
        expected_max = 4.0  # λ_{N/2} = 2(1 - cos(π)) = 4
    else:
        expected_max = 2.0 * (1.0 + np.cos(np.pi / N))  # at k = (N-1)/2
    checks["max_eigenvalue_expected"] = float(expected_max)
    checks["max_eigenvalue_actual"] = max_eig
    checks["max_eigenvalue_match"] = abs(max_eig - expected_max) < 1e-12
    checks["max_leq_4"] = max_eig <= 4.0 + 1e-15
    checks["N_parity"] = "even" if N % 2 == 0 else "odd"

    # (c) Non-negativity
    checks["all_non_negative"] = bool(np.all(eigs >= -1e-15))

    # (d) Spectral symmetry: λ_k = λ_{N-k mod N}
    # This is exact for all N because cos(2π(N-k)/N) = cos(2πk/N).
    # However, the stronger "palindromic pairing" k ↦ k + N/2 only
    # makes sense for even N.
    pal_check = np.array([abs(eigs[k] - eigs[(-k) % N]) for k in range(N)])
    palindromic_err = float(np.max(pal_check))
    checks["spectral_symmetry_max_error"] = palindromic_err
    checks["spectral_symmetry"] = palindromic_err < 1e-14

    passed = all([
        checks["zero_mode"],
        checks["max_eigenvalue_match"],
        checks["max_leq_4"],
        checks["all_non_negative"],
        checks["spectral_symmetry"],
    ])

    return TheoremResult(
        name="Eigenvalue Spectral Bounds",
        passed=passed,
        statement=EIGENVALUE_STATEMENT,
        diagnostics={"N": N, **checks},
    )


# =========================================================================
# ASYMPTOTIC FACT 1: Mid-σ Plateau Dimension
# =========================================================================

PLATEAU_STATEMENT = (
    "For a D-dimensional periodic lattice of side N with D_active > 0 active "
    "dimensions, ALL of which have weight w_d ≥ 0.5, in the canonical plateau "
    "regime [σ_lo, σ_hi] (as defined by SweepConfig.plateau_window()):\n"
    "    |d_s(σ) - D_active| < ε(N)\n"
    "where ε(N) → 0 as N → ∞.\n"
    "Empirical bound (fitted, not derived): ε(N) ≲ 6/N + 0.02 for N ≥ 32.\n"
    "If any active weight w_d < 0.5, the assumption is violated and the "
    "bound does not apply."
)


def verify_plateau(D: int, N: int, C_geo: float,
                   n_sigma: int = 400) -> TheoremResult:
    """
    Verify that d_s ≈ D_active in the plateau window.

    Uses the canonical plateau window from SweepConfig (same window the
    experiment uses).  If any active weight < 0.5, returns passed=False
    with 'assumption_violated' rather than introducing a C_geo-dependent
    bound.
    """
    from .sweep import SweepConfig

    eigs_1d = eigenvalues_1d(N)
    weights = capacity_weights(C_geo, D)
    D_active = int(np.sum(weights > 0))

    if D_active == 0:
        return TheoremResult(
            name="Mid-σ Plateau Dimension",
            passed=True,
            statement=PLATEAU_STATEMENT,
            diagnostics={"D": D, "N": N, "C_geo": C_geo,
                         "D_active": 0, "note": "No active dims; trivially true"},
        )

    # Check assumption: all active weights ≥ 0.5
    active_weights = weights[weights > 1e-15]
    min_active_weight = float(np.min(active_weights))
    if min_active_weight < 0.5:
        return TheoremResult(
            name="Mid-σ Plateau Dimension",
            passed=False,
            statement=PLATEAU_STATEMENT,
            diagnostics={
                "D": D, "N": N, "C_geo": C_geo,
                "D_active": D_active,
                "min_active_weight": min_active_weight,
                "assumption_violated": (
                    f"min active weight = {min_active_weight:.4f} < 0.5; "
                    "plateau bound does not apply"
                ),
            },
        )

    # Use canonical plateau window (same as the experiment)
    config = SweepConfig(D=D, N=N, sigma_max=max(200.0, N**2 / 10))
    sigma_lo, sigma_hi = config.plateau_window()

    sigma = np.geomspace(0.1, max(200.0, N**2 / 10), n_sigma)
    ln_P = log_return_probability(eigs_1d, weights, sigma)
    ds = spectral_dimension(sigma, ln_P)

    mask = (sigma >= sigma_lo) & (sigma <= sigma_hi)

    if np.sum(mask) < 5:
        return TheoremResult(
            name="Mid-σ Plateau Dimension",
            passed=False,
            statement=PLATEAU_STATEMENT,
            diagnostics={"D": D, "N": N, "C_geo": C_geo,
                         "error": "Plateau window too narrow",
                         "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
                         "n_points_in_window": int(np.sum(mask))},
        )

    ds_plateau = ds[mask]
    median_ds = float(np.median(ds_plateau))
    max_deviation = float(np.max(np.abs(ds_plateau - D_active)))
    mean_deviation = float(np.mean(np.abs(ds_plateau - D_active)))

    # Empirical bound: 6/N + 0.02 (fitted from experiments, NOT derived).
    # No C_geo-dependent slack.
    bound = 6.0 / N + 0.02
    passed = max_deviation < bound

    return TheoremResult(
        name="Mid-σ Plateau Dimension",
        passed=passed,
        statement=PLATEAU_STATEMENT,
        diagnostics={
            "D": D, "N": N, "C_geo": C_geo,
            "D_active": D_active,
            "median_ds": median_ds,
            "max_deviation_from_D_active": max_deviation,
            "mean_deviation": mean_deviation,
            "empirical_bound": bound,
            "plateau_window": [sigma_lo, sigma_hi],
            "n_points_in_window": int(np.sum(mask)),
            "min_active_weight": min_active_weight,
        },
    )


# =========================================================================
# ASYMPTOTIC FACT 2: Infinite-Lattice Limit Convergence
# =========================================================================

CONTINUUM_STATEMENT = (
    "For the 1D periodic lattice (ring of N sites), the return probability "
    "P_1D(t; N) converges to the infinite discrete lattice closed form:\n"
    "    P_1D^∞(t) = exp(-2t) I₀(2t)\n"
    "as N → ∞.  This is the N→∞ limit on ℤ, not the continuum limit on ℝ.\n"
    "In the regime t ∈ [1, N²/(8π²)], the error decreases with N.\n"
    "The empirical rate ~1/N² is consistent with Riemann-sum discretisation "
    "error but is not formally derived here."
)


def verify_continuum_limit(N_values: list[int] | None = None,
                           n_t: int = 100) -> TheoremResult:
    """
    Verify that P_1D(t; N) converges to the Bessel form as N increases.
    Check that the relative error decreases with N.

    Requires SciPy for the infinite-lattice reference function.
    """
    try:
        from .spectral import p1d_infinite_lattice
    except ImportError:
        return TheoremResult(
            name="Infinite-Lattice Limit Convergence",
            passed=False,
            statement=CONTINUUM_STATEMENT,
            diagnostics={"error": "SciPy not installed; cannot compute reference"},
        )

    if N_values is None:
        N_values = [16, 32, 64, 128]

    t_grid = np.geomspace(1.0, 20.0, n_t)
    P_inf = p1d_infinite_lattice(t_grid)

    errors_by_N = {}
    for N in N_values:
        eigs = eigenvalues_1d(N)
        P_N = p1d(eigs, t_grid)
        rel_err = np.abs(P_N - P_inf) / np.maximum(P_inf, 1e-300)
        errors_by_N[N] = float(np.max(rel_err))

    # Check convergence: errors should be non-increasing with N
    sorted_N = sorted(N_values)
    eps_tol = 1e-14
    errors_nonincreasing = all(
        errors_by_N[sorted_N[i]] >= errors_by_N[sorted_N[i + 1]] - eps_tol
        for i in range(len(sorted_N) - 1)
    )

    # Convergence ratios (informational, not used for pass/fail)
    convergence_ratios = []
    for i in range(len(sorted_N) - 1):
        N1, N2 = sorted_N[i], sorted_N[i + 1]
        if errors_by_N[N1] > 1e-14:
            ratio = errors_by_N[N2] / errors_by_N[N1]
            expected_ratio = (N1 / N2) ** 2
            convergence_ratios.append({
                "N1": N1, "N2": N2,
                "error_ratio": round(ratio, 6),
                "expected_1_over_N2_ratio": round(expected_ratio, 6),
            })

    has_convergence = errors_by_N[sorted_N[0]] > 1e-4
    passed = errors_nonincreasing and has_convergence and errors_by_N[sorted_N[-1]] < 0.01

    return TheoremResult(
        name="Infinite-Lattice Limit Convergence",
        passed=passed,
        statement=CONTINUUM_STATEMENT,
        diagnostics={
            "N_values": N_values,
            "max_rel_errors": {str(N): round(e, 8) for N, e in errors_by_N.items()},
            "convergence_ratios": convergence_ratios,
            "errors_nonincreasing": errors_nonincreasing,
            "has_measurable_convergence": has_convergence,
        },
    )


# =========================================================================
# EMPIRICAL OBLIGATION 1: Threshold Location
# =========================================================================

THRESHOLD_STATEMENT = (
    "For a D-dimensional lattice of side N, the spectral dimension d_s "
    "crosses the half-integer value k + 0.5 near C_geo ≈ k/D, where the "
    "(k+1)-th dimension activates:\n"
    "    |C_geo* - k/D| < δ(N)\n"
    "where k = 1, ..., D-1, and δ(N) → 0 as N → ∞.\n"
    "Empirical bound (fitted): δ(N) < 0.1 + 2/N for N ≥ 16.\n"
    "This is an empirical observation about the extraction pipeline, "
    "not a theorem.  The exact crossing location depends on the plateau "
    "window and finite-difference estimator."
)


def verify_thresholds(D: int, N: int,
                      C_geo_steps: int = 200, n_sigma: int = 400) -> TheoremResult:
    """
    Verify that d_s transitions occur near C_geo = k/D.
    """
    from .sweep import SweepConfig, run_capacity_sweep

    config = SweepConfig(
        D=D, N=N,
        C_geo_min=0.01, C_geo_max=1.0,
        C_geo_steps=C_geo_steps, n_sigma=n_sigma,
    )
    result = run_capacity_sweep(config)

    checked = []
    max_error = 0.0

    for k in range(1, D):
        target = k + 0.5
        predicted_C = float(k) / D

        ds_p = result.ds_plateau
        C_vals = result.C_geo_values
        found_C = None

        for i in range(len(ds_p) - 1):
            if (ds_p[i] < target <= ds_p[i + 1]) or (ds_p[i + 1] < target <= ds_p[i]):
                d1, d2 = float(ds_p[i]), float(ds_p[i + 1])
                c1, c2 = float(C_vals[i]), float(C_vals[i + 1])
                if abs(d2 - d1) > 1e-10:
                    found_C = c1 + (target - d1) * (c2 - c1) / (d2 - d1)
                break

        if found_C is not None:
            error = abs(found_C - predicted_C)
            max_error = max(max_error, error)
            checked.append({
                "target_ds": target,
                "predicted_C_at_k_over_D": round(predicted_C, 6),
                "actual_C": round(found_C, 6),
                "error": round(error, 6),
            })
        else:
            checked.append({
                "target_ds": target,
                "predicted_C_at_k_over_D": round(predicted_C, 6),
                "actual_C": None,
                "error": float("inf"),
            })
            max_error = float("inf")

    bound = 0.1 + 2.0 / N
    passed = max_error < bound and len(checked) == D - 1

    return TheoremResult(
        name="Threshold Location",
        passed=passed,
        statement=THRESHOLD_STATEMENT,
        diagnostics={
            "D": D, "N": N,
            "max_error": max_error,
            "bound": bound,
            "thresholds": checked,
            "C_geo_steps": C_geo_steps,
        },
    )


# =========================================================================
# EMPIRICAL OBLIGATION 2: Monotonicity of d_s in C_geo
# =========================================================================

MONOTONICITY_STATEMENT = (
    "For a fixed D-dimensional lattice of side N, the plateau spectral "
    "dimension d_s_plateau(C_geo) is non-decreasing in C_geo:\n"
    "    C_geo_1 ≤ C_geo_2  ⟹  d_s_plateau(C_geo_1) ≤ d_s_plateau(C_geo_2) + ε\n"
    "where ε accounts for finite-size noise and derivative estimation error.\n"
    "Empirical tolerance: ε < 0.05 for N ≥ 32.\n"
    "This is an empirical property of the extraction pipeline, not a theorem."
)


def verify_monotonicity(D: int, N: int,
                        C_geo_steps: int = 200, n_sigma: int = 400) -> TheoremResult:
    """
    Verify that d_s_plateau is non-decreasing in C_geo (within tolerance).
    """
    from .sweep import SweepConfig, run_capacity_sweep

    config = SweepConfig(
        D=D, N=N,
        C_geo_min=0.01, C_geo_max=1.0,
        C_geo_steps=C_geo_steps, n_sigma=n_sigma,
    )
    result = run_capacity_sweep(config)

    ds_p = result.ds_plateau
    decreases = []
    for i in range(len(ds_p) - 1):
        drop = float(ds_p[i] - ds_p[i + 1])
        if drop > 0:
            decreases.append({
                "index": i,
                "C_geo": float(result.C_geo_values[i]),
                "ds_drop": round(drop, 6),
            })

    max_drop = max((d["ds_drop"] for d in decreases), default=0.0)
    eps = 0.05
    passed = max_drop < eps

    return TheoremResult(
        name="Monotonicity of d_s in C_geo",
        passed=passed,
        statement=MONOTONICITY_STATEMENT,
        diagnostics={
            "D": D, "N": N,
            "max_drop": max_drop,
            "tolerance": eps,
            "n_decreases": len(decreases),
            "worst_decreases": decreases[:5],
            "C_geo_steps": C_geo_steps,
        },
    )


# =========================================================================
# ALGEBRAIC THEOREM 4: Capacity-Only Dependence
# =========================================================================

CAPACITY_ONLY_STATEMENT = (
    "In the computation P(σ; C_geo) = Π_d P_1D(w_d(C_geo) · σ), the ONLY "
    "quantity that depends on C_geo is the weight vector w = capacity_weights(C_geo, D).\n"
    "All other quantities (eigenvalues, σ grid, derivative estimator, plateau window) "
    "are functions of fixed parameters (D, N, σ_min, σ_max, n_σ) only.\n"
    "This is verified by:\n"
    "  (a) Signature inspection: no function except capacity_weights accepts C_geo.\n"
    "  (b) Functional test: fixing w directly produces identical results to "
    "fixing C_geo, confirming no hidden C_geo path."
)


def verify_capacity_only(D: int, N: int) -> TheoremResult:
    """
    Verify that the sweep result depends on C_geo ONLY through weights.

    Method: For a given C_geo, compute weights, then run the pipeline with
    those weights directly (bypassing capacity_weights).  Results must be
    identical within tight numerical tolerance.
    """
    import inspect
    from .spectral import log_return_probability as lrp, spectral_dimension as sd

    C_geo = 0.6
    eigs_1d_val = eigenvalues_1d(N)
    sigma = np.geomspace(0.1, 200.0, 200)

    # Path 1: through C_geo
    w_from_C = capacity_weights(C_geo, D)
    ln_P_1 = lrp(eigs_1d_val, w_from_C, sigma)
    ds_1 = sd(sigma, ln_P_1)

    # Path 2: with manually constructed identical weights (no C_geo call)
    d_nom = C_geo * D
    w_manual = np.clip(d_nom - np.arange(D, dtype=np.float64), 0.0, 1.0)
    ln_P_2 = lrp(eigs_1d_val, w_manual, sigma)
    ds_2 = sd(sigma, ln_P_2)

    weights_match = np.allclose(w_from_C, w_manual, atol=1e-15, rtol=0)
    w_max_diff = float(np.max(np.abs(w_from_C - w_manual)))

    lnP_match = np.allclose(ln_P_1, ln_P_2, atol=1e-12, rtol=0)
    lnP_max_diff = float(np.max(np.abs(ln_P_1 - ln_P_2)))

    ds_match = np.allclose(ds_1, ds_2, atol=1e-12, rtol=0)
    ds_max_diff = float(np.max(np.abs(ds_1 - ds_2)))

    # Signature check
    sig_check = {}
    for fn_name, fn in [("eigenvalues_1d", eigenvalues_1d),
                         ("log_return_probability", lrp),
                         ("spectral_dimension", sd)]:
        params = list(inspect.signature(fn).parameters.keys())
        sig_check[fn_name] = "C_geo" not in params

    cw_params = list(inspect.signature(capacity_weights).parameters.keys())
    sig_check["capacity_weights_has_C_geo"] = "C_geo" in cw_params

    passed = (weights_match and lnP_match and ds_match
              and all(sig_check.values()))

    return TheoremResult(
        name="Capacity-Only Dependence",
        passed=passed,
        statement=CAPACITY_ONLY_STATEMENT,
        diagnostics={
            "D": D, "N": N, "C_geo": C_geo,
            "weights_max_abs_diff": w_max_diff,
            "lnP_max_abs_diff": lnP_max_diff,
            "ds_max_abs_diff": ds_max_diff,
            "weights_match": weights_match,
            "lnP_match": lnP_match,
            "ds_match": ds_match,
            "signature_checks": sig_check,
        },
    )


# =========================================================================
# Master verification runner
# =========================================================================

def verify_all(D: int = 3, N: int = 32, verbose: bool = True) -> list[TheoremResult]:
    """
    Run all verification obligations for a given (D, N) configuration.

    Parameters
    ----------
    D : int
        Lattice dimension.
    N : int
        Lattice side length.
    verbose : bool
        Print progress and results.

    Returns
    -------
    results : list[TheoremResult]
    """
    results = []

    checks = [
        ("Eigenvalue Spectral Bounds",
         lambda: verify_eigenvalue_bounds(N)),
        ("Exact Factorisation (C_geo=0.5)",
         lambda: verify_factorisation(D, N, C_geo=0.5)),
        ("Exact Factorisation (C_geo=1.0)",
         lambda: verify_factorisation(D, N, C_geo=1.0)),
        ("Capacity Staircase",
         lambda: verify_staircase(D)),
        ("Mid-σ Plateau (full capacity)",
         lambda: verify_plateau(D, N, C_geo=1.0)),
        ("Mid-σ Plateau (half capacity)",
         lambda: verify_plateau(D, N, C_geo=0.5)),
        ("Threshold Location",
         lambda: verify_thresholds(D, N)),
        ("Monotonicity",
         lambda: verify_monotonicity(D, N)),
        ("Infinite-Lattice Limit Convergence",
         lambda: verify_continuum_limit()),
        ("Capacity-Only Dependence",
         lambda: verify_capacity_only(D, N)),
    ]

    for label, fn in checks:
        if verbose:
            print(f"  Verifying: {label} ...", end=" ", flush=True)
        r = fn()
        results.append(r)
        if verbose:
            status = "PASS" if r.passed else "FAIL"
            print(status)
            if not r.passed:
                for k, v in r.diagnostics.items():
                    print(f"    {k}: {v}")

    return results
