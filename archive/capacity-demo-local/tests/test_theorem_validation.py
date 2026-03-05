"""
Automated validation of all formal theorems across multiple configurations.

Tests every verification obligation from dimshift/theorem.py at several (D, N)
combinations.  Total runtime target: < 30s.

Usage:
    python tests/test_theorem_validation.py
    python -m pytest tests/test_theorem_validation.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.capacity import capacity_weights
from dimshift.theorem import (
    verify_eigenvalue_bounds,
    verify_factorisation,
    verify_staircase,
    verify_plateau,
    verify_thresholds,
    verify_monotonicity,
    verify_continuum_limit,
    verify_capacity_only,
)


# -----------------------------------------------------------------------
# Algebraic Theorem 1: Eigenvalue Spectral Bounds
# -----------------------------------------------------------------------

def test_eigenvalue_bounds():
    """Eigenvalue properties hold for all tested N (even and odd)."""
    for N in [4, 7, 8, 15, 16, 32, 64, 128]:
        r = verify_eigenvalue_bounds(N)
        assert r.passed, f"N={N}: {r.diagnostics}"
    print("  Eigenvalue bounds: PASS (N=4..128, even and odd)")


# -----------------------------------------------------------------------
# Algebraic Theorem 2: Exact Factorisation
# -----------------------------------------------------------------------

def test_factorisation_2d():
    """Factorisation holds exactly for 2D lattices."""
    for N in [8, 16, 32]:
        for C_geo in [0.3, 0.5, 0.7, 1.0]:
            r = verify_factorisation(D=2, N=N, C_geo=C_geo)
            assert r.passed, f"D=2 N={N} C={C_geo}: err={r.diagnostics.get('max_relative_error')}"
    print("  Factorisation (2D): PASS")


def test_factorisation_3d():
    """Factorisation holds exactly for 3D lattices (N≤20 for exact enum)."""
    for N in [8, 16]:
        for C_geo in [0.4, 0.7, 1.0]:
            r = verify_factorisation(D=3, N=N, C_geo=C_geo)
            assert r.passed, f"D=3 N={N} C={C_geo}: err={r.diagnostics.get('max_relative_error')}"
    print("  Factorisation (3D): PASS")


def test_factorisation_4d():
    """Factorisation verified for 4D (exact enumeration for N=16: 65536 sites)."""
    r = verify_factorisation(D=4, N=16, C_geo=0.75)
    assert r.passed, f"D=4 N=16: {r.diagnostics}"
    assert r.diagnostics["method"] == "exact_enumeration", (
        f"Expected exact enumeration, got {r.diagnostics['method']}"
    )
    print("  Factorisation (4D exact): PASS")


# -----------------------------------------------------------------------
# Algebraic Theorem 3: Capacity Staircase
# -----------------------------------------------------------------------

def test_staircase():
    """Active dimension count matches formula for D=1..6."""
    for D in range(1, 7):
        r = verify_staircase(D, n_points=5000)
        assert r.passed, f"D={D}: {r.diagnostics['n_failures']} failures"
    print("  Capacity staircase: PASS (D=1..6)")


# -----------------------------------------------------------------------
# Asymptotic Fact 1: Mid-σ Plateau Dimension
# -----------------------------------------------------------------------

def test_plateau_full_capacity():
    """At C_geo=1, plateau d_s ≈ D within bound for multiple (D,N).

    At C_geo=1, all weights = 1.0 (well above the 0.5 assumption threshold).
    N must be large enough for a usable plateau window.
    """
    configs = [(2, 32), (2, 64), (3, 32), (3, 64), (4, 32)]
    for D, N in configs:
        r = verify_plateau(D, N, C_geo=1.0)
        assert r.passed, (
            f"D={D} N={N}: deviation={r.diagnostics.get('max_deviation_from_D_active')}, "
            f"bound={r.diagnostics.get('empirical_bound')}"
        )
    print("  Plateau (full capacity): PASS")


def test_plateau_partial_capacity():
    """At intermediate C_geo where all active weights ≥ 0.5, plateau holds.

    Only test configs where min active weight ≥ 0.5.  Configs with smaller
    weights violate the theorem's assumptions and should return passed=False
    (assumption_violated), which is correct behaviour, not a test failure.
    """
    # Each (D, N, C_geo) chosen so all active weights ≥ 0.5:
    #   D=3, C_geo=0.5:  w = [1.0, 0.5, 0.0]  → D_active=2, min_active=0.5
    #   D=3, C_geo=2/3:  w = [1.0, 1.0, 0.0]  → D_active=2, min_active=1.0
    #   D=4, C_geo=0.5:  w = [1.0, 1.0, 0, 0]  → D_active=2, min_active=1.0
    #   D=4, C_geo=0.75: w = [1.0, 1.0, 1.0, 0] → D_active=3, min_active=1.0
    configs = [
        (3, 64, 0.5),
        (3, 64, 2/3),
        (4, 32, 0.5),
        (4, 32, 0.75),
    ]
    for D, N, C_geo in configs:
        # Verify our assumption about weights
        w = capacity_weights(C_geo, D)
        active = w[w > 1e-15]
        assert len(active) > 0 and np.min(active) >= 0.5, (
            f"Bad test config: D={D} C_geo={C_geo} weights={w.tolist()}"
        )
        r = verify_plateau(D, N, C_geo=C_geo)
        assert r.passed, (
            f"D={D} N={N} C={C_geo}: deviation={r.diagnostics.get('max_deviation_from_D_active')}"
        )
    print("  Plateau (partial capacity, w_d ≥ 0.5): PASS")


def test_plateau_assumption_violated():
    """Plateau correctly rejects configs where active weight < 0.5."""
    # D=3, C_geo=0.4: w = [1.0, 0.2, 0.0] → min_active=0.2 < 0.5
    r = verify_plateau(D=3, N=64, C_geo=0.4)
    assert not r.passed, "Should fail when assumption violated"
    assert "assumption_violated" in r.diagnostics, (
        f"Expected assumption_violated in diagnostics: {r.diagnostics}"
    )
    print("  Plateau (assumption violation detected): PASS")


# -----------------------------------------------------------------------
# Empirical Obligation 1: Threshold Location
# -----------------------------------------------------------------------

def test_thresholds():
    """Transitions occur within tolerance of predicted C_geo = k/D."""
    configs = [(3, 32), (3, 64), (4, 32)]
    for D, N in configs:
        r = verify_thresholds(D, N, C_geo_steps=100, n_sigma=200)
        assert r.passed, f"D={D} N={N}: max_error={r.diagnostics['max_error']}"
    print("  Threshold location: PASS")


# -----------------------------------------------------------------------
# Empirical Obligation 2: Monotonicity of d_s in C_geo
# -----------------------------------------------------------------------

def test_monotonicity():
    """d_s_plateau is non-decreasing in C_geo (within 0.05 tolerance)."""
    configs = [(2, 32), (3, 32), (3, 64), (4, 32)]
    for D, N in configs:
        r = verify_monotonicity(D, N, C_geo_steps=100, n_sigma=200)
        assert r.passed, f"D={D} N={N}: max_drop={r.diagnostics['max_drop']}"
    print("  Monotonicity: PASS")


# -----------------------------------------------------------------------
# Asymptotic Fact 2: Infinite-Lattice Limit Convergence
# -----------------------------------------------------------------------

def test_continuum_limit():
    """P_1D(t; N) converges to Bessel form as N → ∞."""
    r = verify_continuum_limit(N_values=[16, 32, 64, 128])
    assert r.passed, f"Errors: {r.diagnostics.get('max_rel_errors')}"
    print("  Infinite-lattice limit: PASS")


# -----------------------------------------------------------------------
# Algebraic Theorem 4: Capacity-Only Dependence
# -----------------------------------------------------------------------

def test_capacity_only():
    """C_geo flows only through capacity_weights in the full pipeline."""
    for D in [2, 3, 4]:
        r = verify_capacity_only(D, N=32)
        assert r.passed, f"D={D}: {r.diagnostics}"
    print("  Capacity-only dependence: PASS (D=2,3,4)")


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_eigenvalue_bounds,
        test_factorisation_2d,
        test_factorisation_3d,
        test_factorisation_4d,
        test_staircase,
        test_plateau_full_capacity,
        test_plateau_partial_capacity,
        test_plateau_assumption_violated,
        test_thresholds,
        test_monotonicity,
        test_continuum_limit,
        test_capacity_only,
    ]

    print("=" * 60)
    print("Theorem Validation Suite")
    print("=" * 60)
    all_passed = True
    for fn in tests:
        name = fn.__name__
        print(f"\n{name}:")
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL THEOREM VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
        sys.exit(1)
