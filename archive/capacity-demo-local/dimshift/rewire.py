"""
Deterministic lattice construction and edge rewiring for non-separability tests.

Builds a D-dimensional periodic lattice as a sum of per-dimension Laplacians:
    L0 = sum_d L_d

Then rewires a fraction r of local edges to random long-range edges,
producing a non-separable Laplacian:
    L(C_geo) = sum_d (w_d * L_d) + L_rand

Key invariants:
  - Capacity weights w_d come from the SAME capacity_weights(C_geo, D) rule.
  - L_rand is fixed (capacity-independent).
  - The rewiring is deterministic given (D, N, r, seed).
  - Edge count is preserved (each rewire removes 1 local edge, adds 1 random edge).

Vectorized implementation: all lattice construction uses numpy indexing,
no Python loops over N^D nodes.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .capacity import capacity_weights


# ---------------------------------------------------------------------------
# Vectorized lattice graph construction (per-dimension sparse Laplacians)
# ---------------------------------------------------------------------------

def build_per_dimension_laplacians(
    D: int, N: int,
) -> list[sparse.csr_matrix]:
    """
    Build D sparse Laplacian matrices, one per dimension, for a periodic
    D-dimensional cubic lattice with side N.

    Fully vectorized: uses numpy stride arithmetic instead of per-node loops.

    Returns
    -------
    laplacians : list of D sparse CSR matrices, each of shape (N^D, N^D)
    """
    n_total = N ** D
    all_nodes = np.arange(n_total, dtype=np.int64)

    # Precompute strides: stride[d] = N^(D-1-d)
    strides = np.array([N ** (D - 1 - d) for d in range(D)], dtype=np.int64)

    laplacians = []
    for d in range(D):
        c_d = (all_nodes // strides[d]) % N

        # Forward neighbor: c_d -> (c_d + 1) % N
        fwd = all_nodes + ((c_d + 1) % N - c_d) * strides[d]
        # Backward neighbor: c_d -> (c_d - 1) % N
        bwd = all_nodes + ((c_d - 1) % N - c_d) * strides[d]

        rows = np.concatenate([all_nodes, all_nodes, all_nodes])
        cols = np.concatenate([fwd, bwd, all_nodes])
        vals = np.concatenate([
            -np.ones(n_total),
            -np.ones(n_total),
            2.0 * np.ones(n_total),
        ])

        L_d = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_total, n_total), dtype=np.float64
        )
        laplacians.append(L_d)

    return laplacians


# ---------------------------------------------------------------------------
# Vectorized edge catalog for rewiring
# ---------------------------------------------------------------------------

def _enumerate_local_edges_vectorized(
    D: int, N: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Enumerate all undirected local lattice edges (one per direction).

    Returns (src_arr, dst_arr, dim_arr) arrays.
    Only edges where src < dst are included.
    """
    n_total = N ** D
    all_nodes = np.arange(n_total, dtype=np.int64)
    strides = np.array([N ** (D - 1 - d) for d in range(D)], dtype=np.int64)

    src_list, dst_list, dim_list = [], [], []

    for d in range(D):
        c_d = (all_nodes // strides[d]) % N
        fwd = all_nodes + ((c_d + 1) % N - c_d) * strides[d]

        mask = all_nodes < fwd
        src_list.append(all_nodes[mask])
        dst_list.append(fwd[mask])
        dim_list.append(np.full(int(mask.sum()), d, dtype=np.int64))

    return (
        np.concatenate(src_list),
        np.concatenate(dst_list),
        np.concatenate(dim_list),
    )


# ---------------------------------------------------------------------------
# Deterministic rewiring
# ---------------------------------------------------------------------------

def rewire_lattice(
    D: int,
    N: int,
    rewire_rate: float,
    seed: int = 42,
) -> dict:
    """
    Build a partially rewired D-dimensional periodic lattice.

    Returns dict with keys: L_dims, L_rand, n_total, n_rewired, n_local_edges, metadata.
    """
    n_total = N ** D
    rng = np.random.default_rng(seed)

    L_dims = build_per_dimension_laplacians(D, N)

    edge_src, edge_dst, edge_dim = _enumerate_local_edges_vectorized(D, N)
    n_local = len(edge_src)
    n_to_rewire = max(0, int(np.floor(rewire_rate * n_local)))

    if n_to_rewire == 0:
        L_rand = sparse.csr_matrix((n_total, n_total), dtype=np.float64)
        return {
            "L_dims": L_dims, "L_rand": L_rand,
            "n_total": n_total, "n_rewired": 0, "n_local_edges": n_local,
            "metadata": {
                "type": "rewired_lattice", "D": D, "N": N,
                "n_total": n_total, "rewire_rate": rewire_rate,
                "n_local_edges_original": n_local, "n_rewired": 0, "seed": seed,
            },
        }

    rewire_indices = rng.choice(n_local, size=n_to_rewire, replace=False)

    existing_edges = set()
    for i in range(n_local):
        s, d = int(edge_src[i]), int(edge_dst[i])
        existing_edges.add((min(s, d), max(s, d)))

    rand_rows, rand_cols, rand_vals = [], [], []
    removal_by_dim: dict[int, tuple[list, list, list]] = {d: ([], [], []) for d in range(D)}

    n_actually_rewired = 0
    for ri in rewire_indices:
        src = int(edge_src[ri])
        dst = int(edge_dst[ri])
        dim = int(edge_dim[ri])
        edge_key = (min(src, dst), max(src, dst))

        if edge_key not in existing_edges:
            continue

        r, c, v = removal_by_dim[dim]
        r.extend([src, dst, src, dst])
        c.extend([dst, src, src, dst])
        v.extend([1.0, 1.0, -1.0, -1.0])
        existing_edges.discard(edge_key)

        added = False
        for _ in range(20):
            a, b = int(rng.integers(0, n_total)), int(rng.integers(0, n_total))
            if a == b:
                continue
            new_key = (min(a, b), max(a, b))
            if new_key in existing_edges:
                continue
            rand_rows.extend([a, b, a, b])
            rand_cols.extend([b, a, a, b])
            rand_vals.extend([-1.0, -1.0, 1.0, 1.0])
            existing_edges.add(new_key)
            added = True
            break

        if added:
            n_actually_rewired += 1

    for d in range(D):
        r, c, v = removal_by_dim[d]
        if len(v) > 0:
            removal = sparse.csr_matrix(
                (v, (r, c)), shape=(n_total, n_total), dtype=np.float64,
            )
            L_dims[d] = L_dims[d] + removal

    if len(rand_vals) > 0:
        L_rand = sparse.csr_matrix(
            (rand_vals, (rand_rows, rand_cols)),
            shape=(n_total, n_total), dtype=np.float64,
        )
    else:
        L_rand = sparse.csr_matrix((n_total, n_total), dtype=np.float64)

    metadata = {
        "type": "rewired_lattice", "D": D, "N": N, "n_total": n_total,
        "rewire_rate": rewire_rate, "n_local_edges_original": n_local,
        "n_rewired": n_actually_rewired, "seed": seed,
    }

    return {
        "L_dims": L_dims, "L_rand": L_rand,
        "n_total": n_total, "n_rewired": n_actually_rewired,
        "n_local_edges": n_local, "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Build capacity-weighted Laplacian
# ---------------------------------------------------------------------------

def build_weighted_laplacian(
    L_dims: list[sparse.csr_matrix],
    L_rand: sparse.csr_matrix,
    C_geo: float,
    D: int,
) -> sparse.csr_matrix:
    """
    Build the capacity-weighted non-separable Laplacian.

    L(C_geo) = sum_d w_d * L_dims[d] + L_rand
    """
    weights = capacity_weights(C_geo, D)
    n = L_dims[0].shape[0]

    L = sparse.csr_matrix((n, n), dtype=np.float64)
    for d in range(D):
        if weights[d] > 1e-15:
            L = L + weights[d] * L_dims[d]

    L = L + L_rand
    return L
