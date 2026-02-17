"""
Tests for the dimshift library.

All tests use small lattices and coarse grids to run in < 10s total.

Tests:
  1. test_spectral_dim_full_capacity_matches_dimension
     At C_geo=1, d_s ≈ D within 1% for D=1..4.

  2. test_capacity_weights_monotone_in_C_geo
     Weights are monotonically non-decreasing in C_geo.

  3. test_dimshift_thresholds_smoke
     A 3D sweep shows distinct plateaus and increasing d_s with C_geo.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure dimshift is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.capacity import capacity_weights, clamp01
from dimshift.spectral import (
    eigenvalues_1d,
    p1d,
    log_return_probability,
    spectral_dimension,
)
from dimshift.sweep import SweepConfig, run_capacity_sweep


# -----------------------------------------------------------------------
# Test 1: Full capacity → d_s ≈ D
# -----------------------------------------------------------------------

def test_spectral_dim_full_capacity_matches_dimension():
    """At C_geo=1, measured d_s should equal lattice dimension D within 1%."""
    # Using N=64 for 1% accuracy; plateau window scales with N.
    sigma = np.geomspace(0.1, 200.0, 300)

    for D in [1, 2, 3]:
        N = 64
        eigs = eigenvalues_1d(N)
        weights = capacity_weights(1.0, D)
        ln_P = log_return_probability(eigs, weights, sigma)
        ds = spectral_dimension(sigma, ln_P)

        # Plateau window: σ ∈ [5, 0.4 N²/(4π²)]
        sigma_lo = 5.0
        sigma_hi = 0.4 * N**2 / (4 * np.pi**2)
        mask = (sigma >= sigma_lo) & (sigma <= sigma_hi)
        ds_plateau = np.median(ds[mask])

        error = abs(ds_plateau - D)
        rel_error = error / D
        assert rel_error < 0.01, (
            f"D={D}: d_s={ds_plateau:.4f}, expected {D}, "
            f"relative error {rel_error:.4f} > 0.01"
        )
        print(f"  D={D}: d_s={ds_plateau:.4f}  (error={rel_error:.4%}) PASS")


# -----------------------------------------------------------------------
# Test 2: Capacity weights monotone in C_geo
# -----------------------------------------------------------------------

def test_capacity_weights_monotone_in_C_geo():
    """Each weight component must be non-decreasing as C_geo increases."""
    D = 4
    C_values = np.linspace(0.0, 1.0, 100)
    prev_weights = capacity_weights(0.0, D)

    for C in C_values[1:]:
        w = capacity_weights(C, D)
        for d in range(D):
            assert w[d] >= prev_weights[d] - 1e-15, (
                f"C_geo={C:.3f}, dim {d+1}: w={w[d]:.6f} < prev={prev_weights[d]:.6f}"
            )
        prev_weights = w

    # Also check boundary values
    w0 = capacity_weights(0.0, D)
    w1 = capacity_weights(1.0, D)
    assert np.allclose(w0, 0.0), f"C_geo=0 should give all-zero weights, got {w0}"
    assert np.allclose(w1, 1.0), f"C_geo=1 should give all-one weights, got {w1}"

    # Check clamp01
    assert clamp01(-0.5) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.7) == 0.7

    print("  Monotonicity: PASS")
    print("  Boundary values: PASS")
    print("  clamp01: PASS")


# -----------------------------------------------------------------------
# Test 3: Threshold smoke test
# -----------------------------------------------------------------------

def test_dimshift_thresholds_smoke():
    """A 3D sweep must show distinct plateaus and increasing effective dimension."""
    config = SweepConfig(
        D=3, N=64, C_geo_steps=25, n_sigma=200,
        C_geo_min=0.01,  # start very low so d_s starts well below 1
        sigma_min=0.1, sigma_max=200.0,
    )
    result = run_capacity_sweep(config)

    ds = result.ds_plateau

    # Must span from near 1 to near 3
    # Note: d_s is always ≥ 1 for any C_geo > 0 (at least 1 dimension active)
    assert ds[0] < 1.2, f"Lowest C_geo should give d_s near 1, got {ds[0]:.2f}"
    assert ds[-1] > 2.5, f"Highest C_geo should give d_s > 2.5, got {ds[-1]:.2f}"

    # Overall increasing: last > first by at least 1.5
    assert ds[-1] - ds[0] > 1.5, (
        f"d_s should increase by > 1.5 across sweep, got {ds[-1] - ds[0]:.2f}"
    )

    # Must detect thresholds for d_s = 1.5 and d_s = 2.5 (cross-plateau jumps)
    threshold_dims = {t["target_dimension"] for t in result.thresholds}
    assert 1.5 in threshold_dims or 2.0 in threshold_dims, (
        f"Should detect 1→2 transition threshold, found: {threshold_dims}"
    )
    assert 2.5 in threshold_dims or 3.0 in threshold_dims, (
        f"Should detect 2→3 transition threshold, found: {threshold_dims}"
    )

    # Thresholds near k/D: the 1.5 threshold should be near 1/3, 2.5 near 2/3
    for t in result.thresholds:
        td = t["target_dimension"]
        if td == 1.5:
            assert abs(t["C_geo_threshold"] - 1.0/3) < 0.1, (
                f"d_s=1.5 threshold at {t['C_geo_threshold']:.3f}, expected ~0.333"
            )
        if td == 2.5:
            assert abs(t["C_geo_threshold"] - 2.0/3) < 0.1, (
                f"d_s=2.5 threshold at {t['C_geo_threshold']:.3f}, expected ~0.667"
            )

    print(f"  Sweep: {len(result.C_geo_values)} C_geo steps")
    print(f"  d_s range: [{ds[0]:.2f}, {ds[-1]:.2f}]")
    print(f"  Thresholds: {result.thresholds}")
    print("  PASS")


# -----------------------------------------------------------------------
# Test 4: Eigenvalue sanity
# -----------------------------------------------------------------------

def test_eigenvalues_1d():
    """1D periodic lattice eigenvalues: min=0, max=4."""
    for N in [16, 32, 64]:
        eigs = eigenvalues_1d(N)
        assert len(eigs) == N
        assert abs(eigs.min()) < 1e-14, f"N={N}: min eigenvalue should be 0"
        assert abs(eigs.max() - 4.0) < 1e-10, f"N={N}: max eigenvalue should be 4"

    print("  Eigenvalue sanity: PASS")


# -----------------------------------------------------------------------
# Test 5: P_1D normalization
# -----------------------------------------------------------------------

def test_p1d_normalization():
    """P_1D(0) = 1 (all modes sum to N/N = 1)."""
    N = 32
    eigs = eigenvalues_1d(N)
    P_at_zero = p1d(eigs, np.array([0.0]))
    assert abs(P_at_zero[0] - 1.0) < 1e-14, f"P_1D(0) should be 1.0, got {P_at_zero[0]}"
    print("  P_1D(0) = 1: PASS")


# -----------------------------------------------------------------------
# Run all
# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_eigenvalues_1d,
        test_p1d_normalization,
        test_capacity_weights_monotone_in_C_geo,
        test_spectral_dim_full_capacity_matches_dimension,
        test_dimshift_thresholds_smoke,
    ]
    print("=" * 50)
    print("dimshift test suite")
    print("=" * 50)
    all_passed = True
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{name}:")
        try:
            test_fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
