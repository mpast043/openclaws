"""
dimshift — Capacity-filtered spectral dimension library for Framework v4.5.

Demonstrates that the same fixed substrate yields different effective geometry
(spectral dimension) when only the observational capacity C_geo changes.
"""

from .capacity import capacity_weights, clamp01
from .spectral import (
    eigenvalues_1d,
    p1d,
    p1d_infinite_lattice,
    log_return_probability,
    batch_log_return_probability,
    return_probability,
    spectral_dimension,
)
from .sweep import run_capacity_sweep, SweepConfig, SweepResult, CAPACITY_AXES
from .plotting import (
    plot_heatmap,
    plot_representative_curves,
    plot_phase_diagram,
    save_all_figures,
)
from .theorem import (
    TheoremResult,
    verify_all,
    verify_factorisation,
    verify_staircase,
    verify_plateau,
    verify_thresholds,
    verify_monotonicity,
    verify_eigenvalue_bounds,
    verify_continuum_limit,
    verify_capacity_only,
)
from .substrates import (
    SubstrateResult,
    periodic_lattice,
    random_geometric_graph,
    small_world_graph,
)
from .spectral_filters import (
    FilterResult,
    SpectralFilter,
    HardCutoffFilter,
    SoftCutoffFilter,
    PowerLawFilter,
    get_filter,
)
from .framework_spectral import (
    filtered_log_return_probability,
    filtered_return_probability,
    filtered_spectral_dimension,
    extract_plateau,
    check_scaling_assumption,
)

__all__ = [
    "capacity_weights",
    "clamp01",
    "eigenvalues_1d",
    "p1d",
    "p1d_infinite_lattice",
    "log_return_probability",
    "batch_log_return_probability",
    "return_probability",
    "spectral_dimension",
    "run_capacity_sweep",
    "SweepConfig",
    "SweepResult",
    "CAPACITY_AXES",
    "plot_heatmap",
    "plot_representative_curves",
    "plot_phase_diagram",
    "save_all_figures",
    "TheoremResult",
    "verify_all",
    "verify_factorisation",
    "verify_staircase",
    "verify_plateau",
    "verify_thresholds",
    "verify_monotonicity",
    "verify_eigenvalue_bounds",
    "verify_continuum_limit",
    "verify_capacity_only",
    # substrates
    "SubstrateResult",
    "periodic_lattice",
    "random_geometric_graph",
    "small_world_graph",
    # spectral filters
    "FilterResult",
    "SpectralFilter",
    "HardCutoffFilter",
    "SoftCutoffFilter",
    "PowerLawFilter",
    "get_filter",
    # framework spectral
    "filtered_log_return_probability",
    "filtered_return_probability",
    "filtered_spectral_dimension",
    "extract_plateau",
    "check_scaling_assumption",
]
