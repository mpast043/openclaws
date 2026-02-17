"""
Fixed substrates for the framework validation.

Each substrate produces a graph Laplacian L (as eigenvalues) and metadata.
Substrates are deterministic: any randomness uses a fixed seed stored
in the returned metadata.

Interface: each function returns a SubstrateResult with:
  - eigenvalues: sorted ndarray of Laplacian eigenvalues
  - n_vertices: number of vertices
  - metadata: dict with substrate parameters, seed, etc.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SubstrateResult:
    """A fixed substrate described by its Laplacian eigenvalues."""
    name: str
    eigenvalues: NDArray[np.float64]  # sorted, shape (n_vertices,)
    n_vertices: int
    metadata: dict = field(default_factory=dict)


# -----------------------------------------------------------------------
# Substrate 1: Periodic lattice (existing model, generalised)
# -----------------------------------------------------------------------

def periodic_lattice(D: int, N: int) -> SubstrateResult:
    """
    D-dimensional periodic cubic lattice with side N.

    Eigenvalues of the graph Laplacian on Z_N^D are:
        λ(k_1,...,k_D) = Σ_d 2(1 - cos(2π k_d / N))
    for k_d = 0, ..., N-1.

    Total vertices: N^D.

    Parameters
    ----------
    D : int
        Lattice dimension (must be >= 1).
    N : int
        Side length (must be >= 2).
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")

    # 1D eigenvalues
    k = np.arange(N, dtype=np.float64)
    eigs_1d = 2.0 * (1.0 - np.cos(2.0 * np.pi * k / N))

    # Full D-dimensional eigenvalues via Cartesian product
    # For small N^D, enumerate; for large, note the eigenvalues are sums
    # of 1D eigenvalues over all D-tuples.
    total = N ** D
    if total > 500_000:
        raise ValueError(
            f"N^D = {total} too large for explicit eigenvalue enumeration. "
            f"Use smaller N or D."
        )

    grids = [eigs_1d for _ in range(D)]
    mesh = np.meshgrid(*grids, indexing='ij')
    full_eigs = sum(m.ravel() for m in mesh)
    full_eigs.sort()

    return SubstrateResult(
        name=f"periodic_lattice_D{D}_N{N}",
        eigenvalues=full_eigs,
        n_vertices=total,
        metadata={
            "type": "periodic_lattice",
            "D": D,
            "N": N,
            "n_vertices": total,
            "max_eigenvalue": float(np.max(full_eigs)),
        },
    )


# -----------------------------------------------------------------------
# Substrate 2: Random geometric graph in R^D
# -----------------------------------------------------------------------

def random_geometric_graph(n_vertices: int, D: int, radius: float,
                           seed: int = 42) -> SubstrateResult:
    """
    Random geometric graph (RGG) in [0,1]^D with connection radius r.

    Points are placed uniformly in the unit hypercube. Two points are
    connected if their Euclidean distance is < radius. The graph Laplacian
    L = Degree - Adjacency is computed and its eigenvalues returned.

    Deterministic via fixed seed.

    Parameters
    ----------
    n_vertices : int
        Number of points (must be >= 4).
    D : int
        Embedding dimension (must be >= 1).
    radius : float
        Connection radius (must be > 0).
    seed : int
        Random seed for point placement.
    """
    if n_vertices < 4:
        raise ValueError(f"n_vertices must be >= 4, got {n_vertices}")
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, size=(n_vertices, D))

    # Pairwise distance matrix
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(points, metric='euclidean'))

    # Adjacency: connected if distance < radius (no self-loops)
    adjacency = (dist_matrix < radius).astype(np.float64)
    np.fill_diagonal(adjacency, 0.0)

    # Graph Laplacian: L = D - A
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency

    # Eigenvalues (symmetric matrix)
    eigs = np.linalg.eigvalsh(laplacian)
    eigs = np.maximum(eigs, 0.0)  # clamp numerical noise
    eigs.sort()

    avg_degree = float(adjacency.sum() / n_vertices)

    return SubstrateResult(
        name=f"rgg_D{D}_n{n_vertices}_r{radius:.3f}_s{seed}",
        eigenvalues=eigs,
        n_vertices=n_vertices,
        metadata={
            "type": "random_geometric_graph",
            "D": D,
            "n_vertices": n_vertices,
            "radius": radius,
            "seed": seed,
            "avg_degree": round(avg_degree, 2),
            "n_edges": int(adjacency.sum() / 2),
            "connected": bool(eigs[1] > 1e-10),
            "max_eigenvalue": float(np.max(eigs)),
        },
    )


# -----------------------------------------------------------------------
# Substrate 3: Small-world (Watts-Strogatz) graph
# -----------------------------------------------------------------------

def small_world_graph(n_vertices: int, k_neighbors: int,
                      rewire_prob: float, seed: int = 42) -> SubstrateResult:
    """
    Watts-Strogatz small-world graph.

    Start with a ring of n_vertices, each connected to k_neighbors nearest
    neighbors. Then rewire each edge with probability rewire_prob.

    Deterministic via fixed seed.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (must be >= 6).
    k_neighbors : int
        Each node is connected to k nearest neighbors in ring topology.
        Must be even and >= 2.
    rewire_prob : float
        Probability of rewiring each edge (0 = regular ring, 1 = random).
    seed : int
        Random seed for rewiring.
    """
    if n_vertices < 6:
        raise ValueError(f"n_vertices must be >= 6, got {n_vertices}")
    if k_neighbors < 2 or k_neighbors % 2 != 0:
        raise ValueError(f"k_neighbors must be even and >= 2, got {k_neighbors}")
    if not (0 <= rewire_prob <= 1):
        raise ValueError(f"rewire_prob must be in [0,1], got {rewire_prob}")

    rng = np.random.default_rng(seed)

    # Build initial ring lattice
    adjacency = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for i in range(n_vertices):
        for j in range(1, k_neighbors // 2 + 1):
            right = (i + j) % n_vertices
            adjacency[i, right] = 1.0
            adjacency[right, i] = 1.0

    # Rewire edges
    for i in range(n_vertices):
        for j in range(1, k_neighbors // 2 + 1):
            if rng.random() < rewire_prob:
                right = (i + j) % n_vertices
                if adjacency[i, right] == 0:
                    continue  # already rewired
                # Remove old edge
                adjacency[i, right] = 0.0
                adjacency[right, i] = 0.0
                # Pick new target (not self, not already connected)
                candidates = [v for v in range(n_vertices)
                              if v != i and adjacency[i, v] == 0]
                if candidates:
                    new_target = rng.choice(candidates)
                    adjacency[i, new_target] = 1.0
                    adjacency[new_target, i] = 1.0
                else:
                    # Restore if no candidates
                    adjacency[i, right] = 1.0
                    adjacency[right, i] = 1.0

    # Graph Laplacian
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency

    eigs = np.linalg.eigvalsh(laplacian)
    eigs = np.maximum(eigs, 0.0)
    eigs.sort()

    avg_degree = float(adjacency.sum() / n_vertices)

    return SubstrateResult(
        name=f"smallworld_n{n_vertices}_k{k_neighbors}_p{rewire_prob:.2f}_s{seed}",
        eigenvalues=eigs,
        n_vertices=n_vertices,
        metadata={
            "type": "small_world",
            "n_vertices": n_vertices,
            "k_neighbors": k_neighbors,
            "rewire_prob": rewire_prob,
            "seed": seed,
            "avg_degree": round(avg_degree, 2),
            "n_edges": int(adjacency.sum() / 2),
            "connected": bool(eigs[1] > 1e-10),
            "max_eigenvalue": float(np.max(eigs)),
        },
    )
