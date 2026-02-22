"""Random geometric graph builder for Capacity→Geometry experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import cKDTree


def build_rgg_layers(
    D: int,
    N: int,
    radius: float,
    seed: int = 42,
) -> dict:
    """Construct D Laplacian layers from a random geometric graph.

    Parameters
    ----------
    D : int
        Capacity dimension (also used as embedding dimension).
    N : int
        "Side length"; total vertices = N**D. For practical runs, keep this small.
    radius : float
        Connection radius in [0, 1]. Larger radius → denser graph.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Keys mirror `rewire_lattice`: L_dims, L_rand, n_total, metadata.
    """
    n_total = N ** D
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, size=(n_total, D))

    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_tree(tree, r=radius)

    layer_rows = [[] for _ in range(D)]
    layer_cols = [[] for _ in range(D)]
    layer_vals = [[] for _ in range(D)]
    diag_accum = [np.zeros(n_total, dtype=np.float64) for _ in range(D)]

    edge_count = 0
    for i in range(n_total):
        for j in neighbor_lists[i]:
            if j <= i:
                continue
            diff = np.abs(points[i] - points[j])
            total = np.sum(diff)
            if total <= 1e-12:
                weights = np.full(D, 1.0 / D, dtype=np.float64)
            else:
                weights = diff / total
            for d in range(D):
                w = float(weights[d])
                layer_rows[d].extend([i, j])
                layer_cols[d].extend([j, i])
                layer_vals[d].extend([-w, -w])
                diag_accum[d][i] += w
                diag_accum[d][j] += w
            edge_count += 1

    L_dims = []
    for d in range(D):
        diag_indices = np.arange(n_total)
        layer_rows[d].extend(diag_indices.tolist())
        layer_cols[d].extend(diag_indices.tolist())
        layer_vals[d].extend(diag_accum[d].tolist())
        L_dims.append(
            sparse.csr_matrix(
                (layer_vals[d], (layer_rows[d], layer_cols[d])),
                shape=(n_total, n_total),
            )
        )

    metadata = {
        "type": "rgg",
        "D": D,
        "N": N,
        "n_total": n_total,
        "radius": radius,
        "n_edges": edge_count,
        "seed": seed,
    }

    return {
        "L_dims": L_dims,
        "L_rand": sparse.csr_matrix((n_total, n_total), dtype=np.float64),
        "n_total": n_total,
        "metadata": metadata,
    }
