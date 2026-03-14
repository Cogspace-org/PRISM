"""Meta-SR: uncertainty map U(s) and confidence signal C(s).

PRISM's core contribution. Constructs an iso-structural uncertainty map
from SR prediction errors. Same state indexing as M, same spatial granularity.

Three regimes for U(s):
    1. Unvisited: U(s) = U_prior (maximum uncertainty)
    2. Cold start (visits < K): U(s) = U_prior * decay^visits
    3. Sufficient data (visits >= K): U(s) = mean of last K scalar errors

References:
    Ambrogioni & Olafsdottir (2023), master.md section 5.4
"""

import numpy as np
from collections import deque


class MetaSR:
    """Metacognitive layer built on SR prediction errors.

    U(s): uncertainty map, iso-structural to M
    C(s): calibrated confidence signal via sigmoid
    Change detection via recent uncertainty monitoring

    Attributes:
        U: Uncertainty array of shape (n_states,).
        visit_counts: Visit count per state.
    """

    def __init__(self, n_states: int, buffer_size: int = 20,
                 U_prior: float = 0.8, decay: float = 0.85,
                 beta: float = 10.0, theta_C: float = 0.3,
                 theta_change: float = 0.5, recent_visits_size: int = 50,
                 adaptive_beta: bool = False):
        """Initialize the meta-SR layer.

        Args:
            n_states: Number of states (must match SRLayer).
            buffer_size: K — circular buffer size per state.
            U_prior: Prior uncertainty for unvisited states.
            decay: Per-visit decay rate for cold-start regime.
            beta: Sigmoid steepness for confidence signal.
            theta_C: Sigmoid center for confidence threshold.
            theta_change: Change detection threshold.
            recent_visits_size: Window size for change detection.
            adaptive_beta: If True, β(s) = β₀ / (1 + var(buffer_s)).
        """
        self.n_states = n_states
        self.buffer_size = buffer_size
        self.U_prior = U_prior
        self.decay = decay
        self.beta = beta
        self.theta_C = theta_C
        self.theta_change = theta_change
        self.adaptive_beta = adaptive_beta

        # Per-state circular buffers of scalar prediction errors
        self._buffers = [deque(maxlen=buffer_size) for _ in range(n_states)]
        self.visit_counts = np.zeros(n_states, dtype=np.int64)

        # Cached uncertainty map (updated incrementally in observe())
        self._U_cache = np.full(n_states, U_prior, dtype=np.float64)

        # Running normalization for delta scalars
        self._all_deltas = deque(maxlen=5000)
        self._p99_cache = 0.0
        self._observe_count = 0

        # Recent visit tracking for change detection
        self._recent_visits = deque(maxlen=recent_visits_size)

        # Adaptive beta cache
        self._beta_cache = np.full(n_states, self.beta, dtype=np.float64)

    def observe(self, s: int, delta_M: np.ndarray):
        """Process a new SR prediction error at state s.

        Computes the scalar compression ||delta_M||_2, normalizes it,
        and stores in the per-state circular buffer.

        Args:
            s: State index where the transition occurred.
            delta_M: Full TD error vector from SRLayer.update().
        """
        # Scalar compression: L2 norm
        delta_scalar = float(np.linalg.norm(delta_M))

        # Store raw delta for running normalization
        self._all_deltas.append(delta_scalar)
        self._observe_count += 1

        # Normalize to [0, 1] via adaptive percentile clipping
        if len(self._all_deltas) >= 10:
            # Recompute p99 every 100 steps (expensive on large deque)
            if self._observe_count % 100 == 0 or self._p99_cache == 0.0:
                self._p99_cache = float(np.percentile(list(self._all_deltas), 99))
            p99 = self._p99_cache
            if p99 > 0:
                delta_normalized = min(delta_scalar / p99, 1.0)
            else:
                delta_normalized = 0.0
        else:
            # Not enough data yet — use raw value clipped to [0, 1]
            delta_normalized = min(delta_scalar, 1.0)

        self._buffers[s].append(delta_normalized)
        self.visit_counts[s] += 1
        self._recent_visits.append(s)

        # Update cached uncertainty for state s
        visits = self.visit_counts[s]
        if visits < self.buffer_size:
            self._U_cache[s] = self.U_prior * (self.decay ** visits)
        else:
            self._U_cache[s] = float(np.mean(self._buffers[s]))

    def uncertainty(self, s: int) -> float:
        """Return U(s) — uncertainty at state s.

        Cache is maintained incrementally in observe().
        Three regimes:
            visits == 0:       U = U_prior
            0 < visits < K:    U = U_prior * decay^visits
            visits >= K:       U = mean(buffer)

        Returns:
            Uncertainty value in [0, 1].
        """
        return float(self._U_cache[s])

    def all_uncertainties(self) -> np.ndarray:
        """Return U for all states. Returns array of shape (n_states,)."""
        return self._U_cache.copy()

    def buffer_variance(self, s: int) -> float:
        """Return variance of the prediction error buffer for state s.

        Returns 0.0 if fewer than 2 observations.
        """
        if len(self._buffers[s]) < 2:
            return 0.0
        return float(np.var(list(self._buffers[s])))

    def confidence(self, s: int) -> float:
        """Confidence signal C(s) in [0, 1]. 1 = high confidence.

        C(s) = 1 / (1 + exp(beta_s * (U(s) - theta_C)))
        If adaptive_beta: beta_s = beta / (1 + var(buffer_s))
        High U -> low C, low U -> high C.
        """
        U_s = self.uncertainty(s)
        if self.adaptive_beta:
            beta_s = self.beta / (1.0 + self.buffer_variance(s))
            self._beta_cache[s] = beta_s
        else:
            beta_s = self.beta
        return 1.0 / (1.0 + np.exp(beta_s * (U_s - self.theta_C)))

    def all_confidences(self) -> np.ndarray:
        """Compute C for all states. Returns array of shape (n_states,)."""
        U = self.all_uncertainties()
        if self.adaptive_beta:
            beta_arr = np.array([
                self.beta / (1.0 + self.buffer_variance(s))
                for s in range(self.n_states)
            ])
            self._beta_cache[:] = beta_arr
            return 1.0 / (1.0 + np.exp(beta_arr * (U - self.theta_C)))
        return 1.0 / (1.0 + np.exp(self.beta * (U - self.theta_C)))

    def get_beta_map(self) -> np.ndarray:
        """Return cached beta values for all states. Shape (n_states,)."""
        return self._beta_cache.copy()

    def robust_uncertainty_stat(self, method="top_k_mean", k=5):
        """Robust U statistic over recently visited states (p90, max, or top-k mean)."""
        if len(self._recent_visits) == 0:
            return 0.0
        recent_unique = set(self._recent_visits)
        U_vals = np.array([self.uncertainty(s) for s in recent_unique])
        if method == "p90":
            return float(np.percentile(U_vals, 90))
        elif method == "max":
            return float(np.max(U_vals))
        elif method == "top_k_mean":
            top_k = min(k, len(U_vals))
            return float(np.mean(np.sort(U_vals)[-top_k:]))
        else:
            raise ValueError(f"Unknown method: {method}")

    def detect_change(self) -> bool:
        """Detect structural change from recent uncertainty spike.

        Computes mean U over recently visited states.
        Returns True if change_score > theta_change.
        """
        if len(self._recent_visits) == 0:
            return False

        recent_unique = set(self._recent_visits)
        change_score = np.mean([self.uncertainty(s) for s in recent_unique])
        return float(change_score) > self.theta_change
