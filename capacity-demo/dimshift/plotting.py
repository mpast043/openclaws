"""
Visualisation for capacity → geometry experiments.

Generates three canonical plots:
1. Heatmap of d_s over (C_geo, log₁₀σ)    — the "money plot"
2. Representative d_s(σ) curves             — per-threshold capacity values
3. Phase diagram: d_eff vs C_geo            — single summary curve

All functions accept a SweepResult and return matplotlib Figure objects.
They can also save directly to files.
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Lazy import matplotlib so the library can be used without it
_MPL_AVAILABLE = None


def _ensure_mpl():
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_heatmap(result, ax=None):
    """
    Heatmap of d_s over (C_geo, log₁₀σ).

    x-axis: C_geo
    y-axis: log₁₀(σ)
    colour: d_s(σ)
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    D = result.config.D

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    log_sigma = np.log10(result.sigma_values)

    # ds_matrix shape: (n_C, n_sigma) — transpose for imshow (y=sigma, x=C_geo)
    extent = [
        float(result.C_geo_values[0]),
        float(result.C_geo_values[-1]),
        float(log_sigma[0]),
        float(log_sigma[-1]),
    ]

    # Clip d_s for clean colour range
    ds_clipped = np.clip(result.ds_matrix, 0, D + 0.5)

    im = ax.imshow(
        ds_clipped.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=D + 0.2),
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, label=r"$d_s(\sigma)$", pad=0.02)

    # Mark plateau window
    lo, hi = result.config.plateau_window()
    ax.axhline(np.log10(lo), color="white", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(np.log10(hi), color="white", ls="--", lw=0.8, alpha=0.6)

    # Mark threshold C_geo values
    for t in result.thresholds:
        if t["target_dimension"] == int(t["target_dimension"]):
            ax.axvline(t["C_geo_threshold"], color="white", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel(r"$C_{\mathrm{geo}}$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}\,\sigma$", fontsize=12)
    ax.set_title(
        f"Spectral Dimension Heatmap — {D}D Lattice (N={result.config.N})",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_representative_curves(result, ax=None):
    """
    d_s(σ) vs log₁₀(σ) for representative C_geo values near thresholds.

    For D=3: C_geo ∈ {0.20, ~1/3, ~2/3, 1.00} showing plateaus at ~1, ~2, ~3.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    D = result.config.D

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Choose representative C_geo values: one below 1/D, then near k/D for k=1..D
    targets = [0.5 / D]  # below first threshold
    for k in range(1, D + 1):
        targets.append(k / D)
    # Snap to nearest available C_geo
    chosen_indices = []
    for t in targets:
        idx = int(np.argmin(np.abs(result.C_geo_values - t)))
        if idx not in chosen_indices:
            chosen_indices.append(idx)

    log_sigma = np.log10(result.sigma_values)
    cmap = plt.cm.coolwarm
    n = len(chosen_indices)

    for rank, idx in enumerate(chosen_indices):
        c = float(result.C_geo_values[idx])
        ds = result.ds_matrix[idx]
        colour = cmap(rank / max(n - 1, 1))
        ax.plot(log_sigma, ds, color=colour, lw=2,
                label=f"$C_{{\\mathrm{{geo}}}}={c:.2f}$")

    # Reference lines for integer dimensions
    colours_ref = ["#e8a0a0", "#a0c0e8", "#a0e8a0", "#e8d8a0"]
    for d in range(1, D + 1):
        ax.axhline(d, color=colours_ref[d - 1] if d <= 4 else "#ccc",
                    ls="--", lw=1, alpha=0.6)
        ax.text(log_sigma[-1] + 0.05, d, f"$d_s={d}$",
                va="center", fontsize=9, color="#666")

    # Mark plateau window
    lo, hi = result.config.plateau_window()
    ax.axvspan(np.log10(lo), np.log10(hi), alpha=0.07, color="green",
               label="plateau window")

    ax.set_xlabel(r"$\log_{10}\,\sigma$", fontsize=12)
    ax.set_ylabel(r"$d_s(\sigma)$", fontsize=12)
    ax.set_ylim(-0.2, D + 1)
    ax.set_title(
        f"Spectral Dimension Curves — {D}D Lattice (N={result.config.N})",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    return fig


def plot_phase_diagram(result, ax=None):
    """
    Phase diagram: d_eff (plateau d_s) vs C_geo, with nominal d_eff overlay.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    D = result.config.D

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    C = result.C_geo_values

    # Measured d_s plateau
    ax.plot(C, result.ds_plateau, "o-", color="#2563eb", lw=2, ms=4,
            label=r"$d_s$ (measured plateau)")

    # Nominal d_eff = sum of weights
    ax.plot(C, result.d_eff_nominal, "--", color="#16a34a", lw=1.5,
            label=r"$d_{\mathrm{eff}}$ (nominal = $\Sigma w_d$)")

    # Integer reference lines
    for d in range(1, D + 1):
        ax.axhline(d, color="#ddd", ls="-", lw=0.8)
        ax.text(float(C[-1]) + 0.01, d, str(d), va="center", fontsize=9, color="#999")

    # Threshold markers
    for t in result.thresholds:
        if t["target_dimension"] == int(t["target_dimension"]):
            ax.axvline(t["C_geo_threshold"], color="#f59e0b", ls=":", lw=1, alpha=0.6)
            ax.annotate(
                f"$d_s={int(t['target_dimension'])}$",
                xy=(t["C_geo_threshold"], t["target_dimension"]),
                xytext=(t["C_geo_threshold"] + 0.03, t["target_dimension"] + 0.3),
                fontsize=8, color="#b45309",
                arrowprops=dict(arrowstyle="->", color="#b45309", lw=0.8),
            )

    ax.set_xlabel(r"$C_{\mathrm{geo}}$", fontsize=12)
    ax.set_ylabel(r"$d_{\mathrm{eff}}$", fontsize=12)
    ax.set_ylim(-0.2, D + 0.8)
    ax.set_xlim(float(C[0]) - 0.02, float(C[-1]) + 0.08)
    ax.set_title(
        f"Phase Diagram — {D}D Lattice (N={result.config.N})",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    return fig


def save_all_figures(
    result,
    output_dir: str | Path,
    fmt: str = "png",
    dpi: int = 150,
) -> dict[str, Path]:
    """
    Generate and save all three canonical plots.

    Returns dict mapping name → file path.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, plot_fn in [
        ("heatmap", plot_heatmap),
        ("representative_curves", plot_representative_curves),
        ("phase_diagram", plot_phase_diagram),
    ]:
        fig = plot_fn(result)
        p = out / f"{name}.{fmt}"
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[name] = p

    return paths
