#!/usr/bin/env python3
"""Generate a single visual artifact showing capacity-controlled geometry shifts."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dimshift.substrates import (
    periodic_lattice,
    random_geometric_graph,
    small_world_graph,
)
from dimshift.spectral_filters import PowerLawFilter
from dimshift.framework_spectral import (
    filtered_spectral_dimension,
    extract_plateau,
)


OUTPUT_PATH = Path("outputs/figures")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUTPUT_PATH / "capacity_visual_proof.png"


def powerlaw_filter_for_substrate(substrate, d0: float) -> PowerLawFilter:
    stype = substrate.metadata.get("type", "")
    if stype == "random_geometric_graph":
        return PowerLawFilter(
            d0=d0,
            beta_max=max(d0, 1.5),
            gamma=0.85,
            lambda0_quantile=0.6,
        )
    if stype == "small_world":
        return PowerLawFilter(
            d0=d0,
            beta_max=2.5,
            gamma=0.7,
            lambda0_quantile=0.7,
        )
    return PowerLawFilter(d0=d0)


def lattice_context():
    sub = periodic_lattice(D=2, N=32)
    sigma_values = np.geomspace(0.5, 300.0, 300)
    sigma_lo = 5.0
    sigma_hi = min(0.4 * sub.metadata["N"] ** 2 / (4 * np.pi**2), 100.0)
    sigma_hi = max(sigma_hi, sigma_lo + 2.0)
    return {
        "label": "Periodic lattice (D=2, N=32)",
        "substrate": sub,
        "filter": powerlaw_filter_for_substrate(sub, d0=2.0),
        "sigma_values": sigma_values,
        "sigma_window": (sigma_lo, sigma_hi),
    }


def rgg_context():
    sub = random_geometric_graph(n_vertices=200, D=2, radius=0.35, seed=42)
    nonzero = sub.eigenvalues[sub.eigenvalues > 1e-10]
    lam_med = float(np.median(nonzero)) if len(nonzero) else 1.0
    scale = max(lam_med, 0.1)
    sigma_values = np.geomspace(0.01, 20.0 / scale, 300)
    sigma_lo = max(0.02 / scale, sigma_values[1])
    sigma_hi = 5.0 / scale
    return {
        "label": "Random geometric graph (n=200, D=2)",
        "substrate": sub,
        "filter": powerlaw_filter_for_substrate(sub, d0=2.0),
        "sigma_values": sigma_values,
        "sigma_window": (sigma_lo, sigma_hi),
    }


def small_world_context():
    sub = small_world_graph(n_vertices=200, k_neighbors=6, rewire_prob=0.3, seed=42)
    nonzero = sub.eigenvalues[sub.eigenvalues > 1e-10]
    lam_med = float(np.median(nonzero)) if len(nonzero) else 1.0
    scale = max(lam_med, 0.1)
    sigma_values = np.geomspace(0.01, 20.0 / scale, 300)
    sigma_lo = max(0.02 / scale, sigma_values[1])
    sigma_hi = 5.0 / scale
    return {
        "label": "Small-world graph (n=200, k=6, p=0.3)",
        "substrate": sub,
        "filter": powerlaw_filter_for_substrate(sub, d0=2.0),
        "sigma_values": sigma_values,
        "sigma_window": (sigma_lo, sigma_hi),
    }


def compute_series(ctx):
    eigenvalues = ctx["substrate"].eigenvalues
    sigma_values = ctx["sigma_values"]
    sigma_lo, sigma_hi = ctx["sigma_window"]
    C_values = np.linspace(0.05, 1.0, 25)
    ds_plateau = []
    lnP_series = {}

    for i, C in enumerate(C_values):
        filter_result = ctx["filter"].apply(eigenvalues, float(C))
        ds, lnP = filtered_spectral_dimension(
            eigenvalues, filter_result.weights, sigma_values
        )
        plateau = extract_plateau(ds, sigma_values, sigma_lo, sigma_hi)
        ds_plateau.append(plateau["ds_plateau"])
        if i in (0, len(C_values) - 1):
            lnP_series[i] = {
                "C": C,
                "ln_sigma": np.log(sigma_values),
                "ln_P": lnP,
            }

    return {
        "C_values": C_values,
        "ds_plateau": np.array(ds_plateau),
        "lnP_series": lnP_series,
        "sigma_values": sigma_values,
        "sigma_window": (sigma_lo, sigma_hi),
    }


def main():
    contexts = [lattice_context(), rgg_context(), small_world_context()]
    rows = len(contexts)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 12), sharex="col")

    for row, ctx in enumerate(contexts):
        result = compute_series(ctx)
        C_vals = result["C_values"]
        ds_vals = result["ds_plateau"]

        ax_ds = axes[row, 0]
        ax_ln = axes[row, 1]

        ax_ds.plot(C_vals, ds_vals, color="#D55E00", linewidth=2)
        ax_ds.set_ylabel("d_s plateau")
        ax_ds.set_xlabel("Capacity C")
        ax_ds.set_title(f"{ctx['label']} — d_s vs C")
        ax_ds.grid(True, alpha=0.3)

        series = result["lnP_series"]
        colors = {min(series.keys()): "#0072B2", max(series.keys()): "#009E73"}
        for idx, data in series.items():
            label = f"C = {data['C']:.2f}"
            ax_ln.plot(
                data["ln_sigma"],
                data["ln_P"],
                label=label,
                color=colors.get(idx, None),
                linewidth=2,
            )
        ax_ln.set_xlabel("ln sigma")
        ax_ln.set_ylabel("ln P(sigma)")
        ax_ln.set_title(f"{ctx['label']} — ln P vs ln sigma")
        ax_ln.legend(frameon=False)
        ax_ln.grid(True, alpha=0.3)

    fig.suptitle(
        "Capacity controls observable geometry across substrates",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_PATH, dpi=200)
    pdf_path = FIG_PATH.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved figure to {FIG_PATH} and {pdf_path}")


if __name__ == "__main__":
    main()
