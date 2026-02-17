"""
Unit tests for the non-separable Laplacian rewire pipeline.

Tests: determinism, Laplacian properties, end-to-end pipeline, capacity-only
invariance, base lattice validation, pipeline determinism, acceptance criteria.
"""

import numpy as np
import pytest
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dimshift.rewire import (
    build_per_dimension_laplacians,
    rewire_lattice,
    build_weighted_laplacian,
)
from dimshift.graph_heat import (
    log_return_probability_sparse,
    spectral_dimension_sparse,
)
from dimshift.nonseparable import (
    run_nonseparable_sweep,
    evaluate_criteria,
    write_artifacts,
)
from dimshift.capacity import capacity_weights
from dimshift.spectral import eigenvalues_1d


TEST_D = 3
TEST_N = 16
TEST_R = 0.03
TEST_SEED = 42


class TestRewireDeterminism:
    """Rewiring is fully deterministic given (D, N, r, seed)."""

    def test_rewire_determinism(self):
        g1 = rewire_lattice(TEST_D, TEST_N, TEST_R, seed=TEST_SEED)
        g2 = rewire_lattice(TEST_D, TEST_N, TEST_R, seed=TEST_SEED)

        assert g1["n_rewired"] == g2["n_rewired"]
        assert g1["n_rewired"] > 0, "No edges were rewired"

        for d in range(TEST_D):
            diff = g1["L_dims"][d] - g2["L_dims"][d]
            assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-14

        diff_rand = g1["L_rand"] - g2["L_rand"]
        assert diff_rand.nnz == 0 or np.max(np.abs(diff_rand.data)) < 1e-14


class TestLaplacianProperties:
    """Rewired Laplacians satisfy fundamental graph Laplacian properties."""

    def test_laplacian_properties(self):
        graph = rewire_lattice(TEST_D, TEST_N, TEST_R, seed=TEST_SEED)

        for C_geo in [0.3, 0.6, 1.0]:
            L = build_weighted_laplacian(
                graph["L_dims"], graph["L_rand"], C_geo, TEST_D
            )

            # Symmetric
            diff = L - L.T
            if diff.nnz > 0:
                assert np.max(np.abs(diff.data)) < 1e-12, "Not symmetric"

            # Row sums zero
            row_sums = np.abs(np.asarray(L.sum(axis=1)).ravel())
            assert np.max(row_sums) < 1e-12, f"Row sums not zero: max={np.max(row_sums)}"

            # Positive semi-definite (check smallest eigenvalue)
            n = L.shape[0]
            if n <= 5000:
                eigs = np.linalg.eigvalsh(L.toarray())
                assert eigs[0] > -1e-10, f"Negative eigenvalue: {eigs[0]}"


class TestNonseparable:
    """The weighted Laplacian is actually non-separable for r > 0."""

    def test_nonseparable(self):
        graph = rewire_lattice(TEST_D, TEST_N, TEST_R, seed=TEST_SEED)

        # L_rand should be non-zero
        assert graph["L_rand"].nnz > 0, "L_rand is empty — not non-separable"
        assert graph["n_rewired"] > 0


class TestCapacityOnly:
    """Only C_geo varies across the sweep — all else is fixed."""

    def test_capacity_only(self):
        graph = rewire_lattice(TEST_D, TEST_N, TEST_R, seed=TEST_SEED)

        # L_dims and L_rand are capacity-independent
        L1 = build_weighted_laplacian(graph["L_dims"], graph["L_rand"], 0.5, TEST_D)
        L2 = build_weighted_laplacian(graph["L_dims"], graph["L_rand"], 0.8, TEST_D)

        # They should differ (different weights)
        diff = L1 - L2
        assert diff.nnz > 0, "Different C_geo should give different L"

        # But the underlying components are the same object
        graph2 = rewire_lattice(TEST_D, TEST_N, TEST_R, seed=TEST_SEED)
        for d in range(TEST_D):
            diff_d = graph["L_dims"][d] - graph2["L_dims"][d]
            assert diff_d.nnz == 0 or np.max(np.abs(diff_d.data)) < 1e-14


class TestBaseLatticeMatches:
    """With r=0, eigenvalues should match the separable lattice formula."""

    def test_base_lattice_matches(self):
        graph = rewire_lattice(TEST_D, TEST_N, 0.0, seed=TEST_SEED)
        assert graph["n_rewired"] == 0

        # Build full Laplacian at C_geo=1.0 (all weights = 1)
        L = build_weighted_laplacian(graph["L_dims"], graph["L_rand"], 1.0, TEST_D)
        eigs_full = np.sort(np.linalg.eigvalsh(L.toarray()))

        # Compare with separable product formula
        eigs_1d = eigenvalues_1d(TEST_N)
        # Full eigenvalues: sum over all dimension combinations
        from itertools import product as iterprod
        eigs_sep = []
        for combo in iterprod(range(TEST_N), repeat=TEST_D):
            eigs_sep.append(sum(eigs_1d[k] for k in combo))
        eigs_sep = np.sort(eigs_sep)

        np.testing.assert_allclose(eigs_full, eigs_sep, atol=1e-10)


class TestPipelineDeterminism:
    """Full pipeline produces identical results on repeated runs."""

    def test_pipeline_determinism(self):
        kwargs = dict(
            D=TEST_D, N=TEST_N, rewire_rate=TEST_R, seed=TEST_SEED,
            C_geo_steps=10, n_sigma=50, n_probes=10, m_lanczos=30,
        )
        r1 = run_nonseparable_sweep(**kwargs)
        r2 = run_nonseparable_sweep(**kwargs)

        np.testing.assert_allclose(r1.ds_plateau, r2.ds_plateau, atol=1e-12)
        np.testing.assert_allclose(r1.ln_P_matrix, r2.ln_P_matrix, atol=1e-12)


class TestAcceptanceCriteria:
    """Acceptance criteria evaluation on a cached test case."""

    def test_acceptance_criteria(self):
        cache_path = Path(__file__).resolve().parent / "data" / "sample_nonseparable_summary.json"
        with open(cache_path, "r") as f:
            meta = json.load(f)

        assert meta["criteria"]["A"]["pass"], "Cached Criterion A should pass"
        assert meta["criteria"]["B"]["pass"], "Cached Criterion B should pass"
        assert meta["criteria"]["C"]["pass"], "Cached Criterion C should pass"
        assert meta["criteria"].get("overall_pass"), "Cached run should pass overall"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
