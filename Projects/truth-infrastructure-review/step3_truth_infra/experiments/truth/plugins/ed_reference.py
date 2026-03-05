"""Truth plugin that produces an exact-diagonalization reference (example: P02).

This is meant for small systems (L <= 10). It writes a compact .npz artifact
under experiments/truth/ed_reference/ and returns a TruthLabel that references
that artifact by sha256.

Extend this plugin to match your exact model taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..schema import TruthLabel, TruthArtifactRef, sha256_file


SCHEMA_VERSION = "truth.v1"


def pauli() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2.0
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2.0
    return I, sx, sy, sz


def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def heisenberg_hamiltonian(L: int, J: float = 1.0, cyclic: bool = True) -> np.ndarray:
    I, sx, sy, sz = pauli()
    dim = 2**L
    H = np.zeros((dim, dim), dtype=complex)

    def op_on_sites(opA, i, opB, j):
        ops = [I] * L
        ops[i] = opA
        ops[j] = opB
        return kron_n(ops)

    bonds = [(i, i + 1) for i in range(L - 1)]
    if cyclic and L > 2:
        bonds.append((L - 1, 0))

    for i, j in bonds:
        H += J * (op_on_sites(sx, i, sx, j) + op_on_sites(sy, i, sy, j) + op_on_sites(sz, i, sz, j))

    # Ensure Hermitian numerical symmetry
    H = (H + H.conj().T) / 2.0
    return H


def entanglement_entropy_bipartition(psi: np.ndarray, L: int, LA: Optional[int] = None) -> float:
    if LA is None:
        LA = L // 2
    dimA = 2**LA
    dimB = 2**(L - LA)
    psi = psi.reshape(dimA, dimB)
    rhoA = psi @ psi.conj().T
    # Numerical safety
    evals = np.linalg.eigvalsh(rhoA)
    evals = np.real(evals)
    evals = evals[evals > 1e-12]
    return float(-(evals * np.log(evals)).sum())


def run_ed_heisenberg(L: int, J: float = 1.0, cyclic: bool = True) -> dict:
    H = heisenberg_hamiltonian(L=L, J=J, cyclic=cyclic)
    evals, evecs = np.linalg.eigh(H)
    E0 = float(np.real(evals[0]))
    psi0 = evecs[:, 0]
    # Normalize defensively
    psi0 = psi0 / np.linalg.norm(psi0)
    S_ref = entanglement_entropy_bipartition(psi0, L=L)
    return {"E0": E0, "psi0": psi0, "S_ref": S_ref}


def generate_truth(claim_id: str, root: Path, substrate_id: Optional[str] = None, seed: Optional[int] = None) -> TruthLabel:
    # Minimal substrate parser; update to match your naming.
    # Expected substrate_id patterns you might use:
    #   heisenberg_L8_cyclic
    sid = (substrate_id or "heisenberg_L8_cyclic").lower()

    L = 8
    J = 1.0
    cyclic = True

    if "l" in sid:
        # parse ..._l8_...
        for tok in sid.replace("-", "_").split("_"):
            if tok.startswith("l") and tok[1:].isdigit():
                L = int(tok[1:])

    if "open" in sid:
        cyclic = False
    if "cyclic" in sid or "periodic" in sid:
        cyclic = True

    # Generate ED
    ed = run_ed_heisenberg(L=L, J=J, cyclic=cyclic)

    out_dir = root / "experiments" / "truth" / "ed_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = f"{sid}_ed_ref.npz"
    artifact_path = out_dir / artifact_name
    np.savez_compressed(artifact_path, E0=ed["E0"], S_ref=ed["S_ref"], psi0=ed["psi0"])  # psi0 is small for L<=10

    ref = TruthArtifactRef(relpath=artifact_name, sha256=sha256_file(artifact_path))

    return TruthLabel(
        schema_version=SCHEMA_VERSION,
        claim_id=claim_id,
        truth_type="dict",
        expected={
            "E0": ed["E0"],
            "S_ref": ed["S_ref"],
            # Do not embed psi0 directly in JSON; reference artifact instead.
        },
        tolerance={
            # These are example tolerances; tune per claim.
            "E0": 1e-3,
            "S_ref": 1e-2,
        },
        source="exact_diagonalization",
        confidence=1.0,
        substrate_id=substrate_id or sid,
        seed=seed,
        artifacts={"ed_ref": ref},
    )
