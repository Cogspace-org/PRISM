"""Visualization: SR heatmaps, uncertainty maps, grid overlays.

References:
    master.md section 6 (figures)
"""


def plot_sr_heatmap(M, state_mapper, source_state=None):
    """Plot SR matrix row as a heatmap on the grid."""
    raise NotImplementedError("Phase 1")


def plot_uncertainty_map(U, state_mapper):
    """Plot uncertainty U(s) as a heatmap on the grid."""
    raise NotImplementedError("Phase 2")


def plot_value_map(V, state_mapper):
    """Plot value function V(s) as a heatmap on the grid."""
    raise NotImplementedError("Phase 1")
