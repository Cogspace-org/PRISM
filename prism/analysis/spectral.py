"""Spectral decomposition of SR matrix for grid cell validation.

References:
    Stachenfeld et al. (2017), master.md section 7.3
"""

import numpy as np


def sr_eigenvectors(M, k=6):
    """Extract top-k eigenvectors of SR matrix M."""
    raise NotImplementedError("Phase 1")


def plot_eigenvectors(eigenvectors, state_mapper, k=6):
    """Plot eigenvectors as heatmaps on the grid."""
    raise NotImplementedError("Phase 1")
