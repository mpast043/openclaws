"""
Unit tests for Framework v4.5 Option B (spectral filters + substrates).

Tests:
  1. Filter monotonicity: g_{C1}(lam) <= g_{C2}(lam) for C1 <= C2
  2. Filter range: 0 <= g <= 1
  3. Substrate determinism: same config -> identical eigenvalues
  4. Pipeline sanity: lattice at C=1 gives d_eff ~ D in plateau window
  5. No hidden knobs: filter fixed params don't depend on C

Usage:
    python tests/test_framework_b.py
    python -m pytest tests/test_framework_b.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.substrates import (
    periodic_lattice,
    random_geometric_graph,
    small_world_graph,
)
from dimshift.spectral_filters import (
    HardCutoffFilter,
    SoftCutoffFilter,
    PowerLawFilter,
)
from dimshift.framework_spectral import (
    filtered_spectral_dimension,
    extract_plateau,
    check_scaling_assumption,
)


# -----------------------------------------------------------------------
# Helper: get a set of eigenvalues for filter tests
# -----------------------------------------------------------------------

def _test_eigenvalues():
    """Small 2D lattice eigenvalues for fast tests."""
    sub = periodic_lattice(D=2, N=16)
    return sub.eigenvalues


# -----------------------------------------------------------------------
# Test 1: Filter monotonicity
# -----------------------------------------------------------------------

def test_filter_monotonicity_hard():
    """Hard cutoff: g_{C1} <= g_{C2} for C1 <= C2."""
    evals = _test_eigenvalues()
    filt = HardCutoffFilter()
    C_grid = np.linspace(0.05, 1.0, 20)
    prev_w = filt.apply(evals, C_grid[0]).weights
    for C in C_grid[1:]:
        curr_w = filt.apply(evals, C).weights
        assert np.all(curr_w >= prev_w - 1e-15), (
            f"Hard cutoff not monotone at C={C}"
        )
        prev_w = curr_w
    print("  Filter monotonicity (hard cutoff): PASS")


def test_filter_monotonicity_soft():
    """Soft cutoff: g_{C1} <= g_{C2} for C1 <= C2."""
    evals = _test_eigenvalues()
    filt = SoftCutoffFilter(steepness_frac=0.05)
    C_grid = np.linspace(0.05, 1.0, 20)
    prev_w = filt.apply(evals, C_grid[0]).weights
    for C in C_grid[1:]:
        curr_w = filt.apply(evals, C).weights
        assert np.all(curr_w >= prev_w - 1e-15), (
            f"Soft cutoff not monotone at C={C}"
        )
        prev_w = curr_w
    print("  Filter monotonicity (soft cutoff): PASS")


def test_filter_monotonicity_powerlaw():
    """Power-law: g_{C1} <= g_{C2} for C1 <= C2."""
    evals = _test_eigenvalues()
    filt = PowerLawFilter(d0=2.0)
    C_grid = np.linspace(0.05, 1.0, 20)
    prev_w = filt.apply(evals, C_grid[0]).weights
    for C in C_grid[1:]:
        curr_w = filt.apply(evals, C).weights
        assert np.all(curr_w >= prev_w - 1e-15), (
            f"Power-law not monotone at C={C}"
        )
        prev_w = curr_w
    print("  Filter monotonicity (power-law): PASS")


# -----------------------------------------------------------------------
# Test 2: Filter range [0, 1]
# -----------------------------------------------------------------------

def test_filter_range():
    """All filters produce weights in [0, 1]."""
    evals = _test_eigenvalues()
    filters = [
        HardCutoffFilter(),
        SoftCutoffFilter(steepness_frac=0.05),
        PowerLawFilter(d0=2.0),
    ]
    for filt in filters:
        for C in [0.05, 0.25, 0.5, 0.75, 1.0]:
            w = filt.apply(evals, C).weights
            assert np.all(w >= 0.0 - 1e-15), f"{filt.name} C={C}: weights < 0"
            assert np.all(w <= 1.0 + 1e-15), f"{filt.name} C={C}: weights > 1"
    print("  Filter range [0,1]: PASS")


# -----------------------------------------------------------------------
# Test 3: Substrate determinism
# -----------------------------------------------------------------------

def test_substrate_determinism_lattice():
    """Periodic lattice is fully deterministic."""
    s1 = periodic_lattice(D=2, N=16)
    s2 = periodic_lattice(D=2, N=16)
    assert np.array_equal(s1.eigenvalues, s2.eigenvalues), "Lattice not deterministic"
    print("  Substrate determinism (lattice): PASS")


def test_substrate_determinism_rgg():
    """RGG with same seed -> identical eigenvalues."""
    s1 = random_geometric_graph(n_vertices=50, D=2, radius=0.4, seed=42)
    s2 = random_geometric_graph(n_vertices=50, D=2, radius=0.4, seed=42)
    assert np.allclose(s1.eigenvalues, s2.eigenvalues, atol=1e-12), (
        "RGG not deterministic"
    )
    print("  Substrate determinism (RGG): PASS")


def test_substrate_determinism_smallworld():
    """Small-world with same seed -> identical eigenvalues."""
    s1 = small_world_graph(n_vertices=50, k_neighbors=4, rewire_prob=0.3, seed=42)
    s2 = small_world_graph(n_vertices=50, k_neighbors=4, rewire_prob=0.3, seed=42)
    assert np.allclose(s1.eigenvalues, s2.eigenvalues, atol=1e-12), (
        "Small-world not deterministic"
    )
    print("  Substrate determinism (small-world): PASS")


# -----------------------------------------------------------------------
# Test 4: Pipeline sanity -- lattice at C=1, d_eff ~ D
# -----------------------------------------------------------------------

def test_pipeline_lattice_full_capacity():
    """For 2D lattice at C=1 (all weights=1), d_s plateau ~ 2."""
    sub = periodic_lattice(D=2, N=32)
    evals = sub.eigenvalues

    # All 3 filters at C=1 should give weights ~1 (full geometry)
    sigma = np.geomspace(0.5, 200.0, 300)

    for filt in [HardCutoffFilter(), SoftCutoffFilter(), PowerLawFilter(d0=2.0)]:
        w = filt.apply(evals, C=1.0).weights
        ds, ln_P = filtered_spectral_dimension(evals, w, sigma)
        plat = extract_plateau(ds, sigma, sigma_lo=5.0, sigma_hi=40.0)
        d_eff = plat["ds_plateau"]
        # At C=1, hard cutoff keeps all modes, soft cutoff ~ all, powerlaw beta=0
        # For hard/soft: d_eff should be close to 2
        # Tolerance: within 0.5 of D=2 (generous for small N)
        assert abs(d_eff - 2.0) < 0.5, (
            f"{filt.name} at C=1: d_eff={d_eff:.3f}, expected ~2.0"
        )
    print("  Pipeline sanity (lattice C=1, d_eff~2): PASS")


def test_pipeline_capacity_reduces_dimension():
    """Reducing C should reduce measured d_s for power-law filter."""
    sub = periodic_lattice(D=3, N=16)
    evals = sub.eigenvalues
    sigma = np.geomspace(0.5, 100.0, 300)
    filt = PowerLawFilter(d0=3.0)

    ds_full, _ = filtered_spectral_dimension(
        evals, filt.apply(evals, C=1.0).weights, sigma
    )
    ds_low, _ = filtered_spectral_dimension(
        evals, filt.apply(evals, C=0.1).weights, sigma
    )

    plat_full = extract_plateau(ds_full, sigma, 3.0, 15.0)["ds_plateau"]
    plat_low = extract_plateau(ds_low, sigma, 3.0, 15.0)["ds_plateau"]

    assert plat_full > plat_low + 0.1, (
        f"Expected d_s(C=1) > d_s(C=0.1): got {plat_full:.3f} vs {plat_low:.3f}"
    )
    print("  Pipeline capacity reduces dimension: PASS")


# -----------------------------------------------------------------------
# Test 5: No hidden knobs -- filter fixed params don't depend on C
# -----------------------------------------------------------------------

def test_no_hidden_knobs_hard():
    """Hard cutoff: no parameters depend on C except m."""
    evals = _test_eigenvalues()
    filt = HardCutoffFilter()
    # The only C-dependent field is 'm' and 'C' itself
    m1 = filt.apply(evals, 0.3).metadata
    m2 = filt.apply(evals, 0.7).metadata
    assert m1["n"] == m2["n"], "n should not depend on C"
    assert m1["filter"] == m2["filter"], "filter name should not depend on C"
    print("  No hidden knobs (hard cutoff): PASS")


def test_no_hidden_knobs_soft():
    """Soft cutoff: steepness s does not depend on C."""
    evals = _test_eigenvalues()
    filt = SoftCutoffFilter(steepness_frac=0.05)
    m1 = filt.apply(evals, 0.3).metadata
    m2 = filt.apply(evals, 0.7).metadata
    assert m1["s"] == m2["s"], f"s depends on C: {m1['s']} vs {m2['s']}"
    assert m1["steepness_frac"] == m2["steepness_frac"]
    print("  No hidden knobs (soft cutoff): PASS")


def test_no_hidden_knobs_powerlaw():
    """Power-law: lambda0 and d0 do not depend on C."""
    evals = _test_eigenvalues()
    filt = PowerLawFilter(d0=2.0)
    m1 = filt.apply(evals, 0.3).metadata
    m2 = filt.apply(evals, 0.7).metadata
    assert m1["lambda0"] == m2["lambda0"], (
        f"lambda0 depends on C: {m1['lambda0']} vs {m2['lambda0']}"
    )
    assert m1["d0"] == m2["d0"], f"d0 depends on C: {m1['d0']} vs {m2['d0']}"
    print("  No hidden knobs (power-law): PASS")


# -----------------------------------------------------------------------
# Test 6: Scaling assumption check returns valid metrics
# -----------------------------------------------------------------------

def test_scaling_assumption_check():
    """check_scaling_assumption returns R^2 and residuals for valid window."""
    sub = periodic_lattice(D=2, N=32)
    evals = sub.eigenvalues
    sigma = np.geomspace(0.5, 200.0, 300)
    filt = PowerLawFilter(d0=2.0)
    w = filt.apply(evals, C=1.0).weights
    _, ln_P = filtered_spectral_dimension(evals, w, sigma)
    result = check_scaling_assumption(ln_P, sigma, sigma_lo=5.0, sigma_hi=40.0)
    assert not np.isnan(result["r_squared"]), "R^2 should not be NaN"
    assert result["r_squared"] > 0.5, f"R^2 too low: {result['r_squared']}"
    assert result["n_points"] >= 5, "Too few points"
    print("  Scaling assumption check: PASS")


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_filter_monotonicity_hard,
        test_filter_monotonicity_soft,
        test_filter_monotonicity_powerlaw,
        test_filter_range,
        test_substrate_determinism_lattice,
        test_substrate_determinism_rgg,
        test_substrate_determinism_smallworld,
        test_pipeline_lattice_full_capacity,
        test_pipeline_capacity_reduces_dimension,
        test_no_hidden_knobs_hard,
        test_no_hidden_knobs_soft,
        test_no_hidden_knobs_powerlaw,
        test_scaling_assumption_check,
    ]

    print("=" * 60)
    print("Framework v4.5 Option B — Unit Tests")
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
        print("ALL FRAMEWORK B TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
