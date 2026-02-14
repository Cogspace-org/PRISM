"""Tabular Successor Representation learned via TD(0).

The key interface requirement: update() must return the full TD error vector
delta_M, which feeds into the meta-SR layer.

References:
    Dayan (1993), Stachenfeld et al. (2017), master.md section 5.3
"""

import numpy as np


class SRLayer:
    """Tabular successor representation with TD(0) learning.

    M(s, s') = E[sum_t gamma^t * I(s_t = s') | s_0 = s, pi]
    V(s) = M(s, :) . R
    """

    def __init__(self, n_states: int, gamma: float = 0.95,
                 alpha_M: float = 0.1, alpha_R: float = 0.3):
        raise NotImplementedError("Phase 1")

    def update(self, s: int, s_next: int, reward: float) -> np.ndarray:
        """Update M and R. Returns TD error vector delta_M (needed by meta-SR)."""
        raise NotImplementedError("Phase 1")

    def value(self, s: int) -> float:
        """Compute V(s) = M[s, :] . R."""
        raise NotImplementedError("Phase 1")

    def all_values(self) -> np.ndarray:
        """Compute V for all states: M @ R."""
        raise NotImplementedError("Phase 1")
