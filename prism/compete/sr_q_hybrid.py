"""SR+Q Hybrid Agent — hippocampal-striatal interaction.

Uses Q-table for action selection (off-policy -> Q*) and SR pipeline
(M, R, MetaSR, CUSUM) for monitoring, change detection, and warm-start.

Normal mode: Q-Learning action selection.
R-change: continuous V_SR blending pulls Q toward M·R (no one-shot).
After M-change: CUSUM detects, soft Q reset for affected states.

Biological framing: Russek et al. (2017) "Dyna-SR" —
  hippocampus provides the predictive map + meta-map,
  striatum executes the policy.
"""
from __future__ import annotations

import numpy as np

from prism.core.sr_layer import SRLayer
from prism.core.meta_sr import MetaSR

from .base import CompeteAgent


class SRQHybridAgent(CompeteAgent):
    """Hybrid SR+Q agent: Q for decisions, SR for monitoring + warm-start."""

    def __init__(
        self,
        env,
        n_actions: int = 4,
        alpha_Q: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        alpha_M: float = 0.1,
        alpha_R: float = 0.3,
        config=None,
        theta_reset: float = 0.40,
        epsilon_max: float = 0.20,
        alpha_reset: float = 0.5,
        n_patience: int = 500,
        blend_rate: float = 0.05,
        blend_interval: int = 5,
        seed: int = 42,
    ) -> None:
        self._env = env
        n_states = env.n_states
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        # Q-table (primary — action selection)
        self.Q = np.zeros((n_states, n_actions))
        self.alpha_Q = alpha_Q
        self.epsilon = epsilon

        # SR pipeline (auxiliary — monitoring + warm-start)
        self.sr = SRLayer(n_states, gamma=gamma, alpha_M=alpha_M, alpha_R=alpha_R)

        # MetaSR config from AdaptConfig or defaults
        if config is not None:
            self.meta_sr = MetaSR(
                n_states,
                buffer_size=getattr(config, "buffer_size", 20),
                decay=getattr(config, "decay", 0.85),
                beta=getattr(config, "beta", 10.0),
                theta_change=getattr(config, "theta_change", 0.2),
            )
            self._cusum_warmup = config.cusum_warmup
            self._cusum_baseline_window = config.cusum_baseline_window
            self._cusum_stat_method = config.cusum_stat_method
            self._cusum_stat_k = config.cusum_stat_k
            self._cusum_slack_factor = config.cusum_slack_factor
            self._cusum_threshold_factor = config.cusum_threshold_factor
            self._detection_mode = config.detection_mode
            self._max_triggers = getattr(config, "max_triggers", 1)
        else:
            self.meta_sr = MetaSR(n_states)
            self._cusum_warmup = 300
            self._cusum_baseline_window = 100
            self._cusum_stat_method = "top_k_mean"
            self._cusum_stat_k = 5
            self._cusum_slack_factor = 0.3
            self._cusum_threshold_factor = 3.0
            self._detection_mode = "cusum"
            self._max_triggers = 1

        # Recovery parameters (softer than PRISM_adapt)
        self.theta_reset = theta_reset
        self.epsilon_max = epsilon_max
        self.alpha_reset = alpha_reset
        self.n_patience = n_patience

        # Recovery state
        self.recovery_mode = False
        self.S_reset: set[int] = set()
        self._patience_counter = 0
        self._trigger_count = 0

        # Mode-switch: Q-mode (stable) vs V_SR-mode (R-change)
        self.blend_rate = blend_rate
        self.blend_interval = blend_interval
        self._R_snapshot = self.sr.R.copy()
        self._r_velocity_threshold = 0.05
        self._vsr_mode = False
        self._vsr_mode_cooldown = 0

        # CUSUM state
        self._cusum_S = 0.0
        self._cusum_baseline: list[float] = []
        self._cusum_mu0 = 0.0
        self._cusum_sigma0 = 0.0
        self._cusum_ready = False
        self._cusum_history: list[tuple] = []

        self.rng = np.random.default_rng(seed)

    # -- action selection (Q-based) ----------------------------------------

    def select_action(self, state: int) -> int:
        eps = self.epsilon_max if (self.recovery_mode and state in self.S_reset) else self.epsilon
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))

        if self._vsr_mode:
            # V_SR-mode: action selection via one-step lookahead on V_SR
            V_sr = self.sr.M @ self.sr.R
            neighbors = self._env.get_neighbors(state)
            best_v = -np.inf
            best_actions = []
            for action, s_next in neighbors:
                v = V_sr[s_next]
                if v > best_v:
                    best_v = v
                    best_actions = [action]
                elif v == best_v:
                    best_actions.append(action)
            return int(self.rng.choice(best_actions))

        # Q-mode: standard Q-Learning action selection
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    # -- learning (dual pipeline) ------------------------------------------

    def learn(self, s: int, a: int, s_next: int, reward: float, done: bool) -> None:
        # Q-Learning update (off-policy)
        target = reward if done else reward + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha_Q * (target - self.Q[s, a])

        # SR pipeline update (M, R, MetaSR)
        delta_M = self.sr.update(s, s_next, reward)
        self.meta_sr.observe(s, delta_M)

        # Recovery exit check
        if self.recovery_mode:
            self._check_recovery_done()

    # -- continuous V_SR -> Q blending --------------------------------------

    def _blend_q_toward_vsr(self):
        """Gently pull Q toward V_SR = M @ R. Continuous — no threshold."""
        V_sr = self.sr.M @ self.sr.R
        for s in range(self.n_states):
            neighbors = self._env.get_neighbors(s)
            for action, s_next in neighbors:
                self.Q[s, action] += self.blend_rate * (V_sr[s_next] - self.Q[s, action])

    # -- M-change response -------------------------------------------------

    def _trigger_m_change_reset(self):
        """Soft Q reset + moderate exploration for affected states."""
        self.S_reset = {
            s for s in range(self.n_states)
            if self.meta_sr.uncertainty(s) > self.theta_reset
        }

        # Soft Q reset
        for s in self.S_reset:
            self.Q[s, :] *= (1.0 - self.alpha_reset)

        # Soft M reset (blend toward identity)
        I = np.eye(self.n_states)
        for s in self.S_reset:
            self.sr.M[s, :] = 0.5 * I[s, :] + 0.5 * self.sr.M[s, :]

        # Reset MetaSR for S_reset (cold-start restart)
        for s in self.S_reset:
            self.meta_sr._buffers[s].clear()
            self.meta_sr.visit_counts[s] = 0
            self.meta_sr._U_cache[s] = self.meta_sr.U_prior

        self.recovery_mode = True
        self._patience_counter = 0
        self._trigger_count += 1

    def _check_recovery_done(self):
        """Exit recovery when no new change detected for n_patience steps."""
        recent_unique = set(self.meta_sr._recent_visits)
        check_states = recent_unique - self.S_reset

        if len(check_states) > 0:
            change_score = np.mean([
                self.meta_sr.uncertainty(s) for s in check_states
            ])
            still_changing = change_score > self.meta_sr.theta_change
        else:
            still_changing = False

        if not still_changing:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        if self._patience_counter >= self.n_patience:
            self.recovery_mode = False
            self.S_reset = set()

    # -- CUSUM detection ---------------------------------------------------

    def _cusum_episode_update(self, episode: int) -> bool:
        """CUSUM change-point detection on U_robust. Returns True if detected."""
        if self._detection_mode != "cusum":
            return False
        if self.recovery_mode or self._trigger_count >= self._max_triggers:
            return False

        x_t = self.meta_sr.robust_uncertainty_stat(
            method=self._cusum_stat_method,
            k=self._cusum_stat_k,
        )

        # Warmup: skip
        if episode < self._cusum_warmup:
            self._cusum_history.append((episode, x_t, 0.0, False))
            return False

        # Collect baseline
        if not self._cusum_ready:
            self._cusum_baseline.append(x_t)
            if len(self._cusum_baseline) >= self._cusum_baseline_window:
                arr = np.array(self._cusum_baseline)
                self._cusum_mu0 = float(np.mean(arr))
                self._cusum_sigma0 = max(float(np.std(arr)), 1e-6)
                self._cusum_ready = True
                self._cusum_S = 0.0
            self._cusum_history.append((episode, x_t, 0.0, False))
            return False

        # CUSUM accumulation
        k = self._cusum_slack_factor * self._cusum_sigma0
        h = self._cusum_threshold_factor * self._cusum_sigma0
        self._cusum_S = max(0.0, self._cusum_S + (x_t - self._cusum_mu0 - k))
        detected = self._cusum_S > h
        self._cusum_history.append((episode, x_t, self._cusum_S, detected))

        if detected:
            self._trigger_m_change_reset()
        return detected

    # -- episode hooks -----------------------------------------------------

    def on_episode_end(self, episode: int) -> None:
        # Mode-switch: detect R-change via velocity (disabled during recovery)
        if not self.recovery_mode and self.blend_interval > 0 and episode % self.blend_interval == 0:
            r_velocity = float(np.linalg.norm(self.sr.R - self._R_snapshot))
            if r_velocity > self._r_velocity_threshold:
                self._vsr_mode = True
                self._vsr_mode_cooldown = 0
                self._blend_q_toward_vsr()
            elif self._vsr_mode:
                self._vsr_mode_cooldown += 1
                if self._vsr_mode_cooldown >= 5:  # 25 ep of stable R -> exit
                    self._vsr_mode = False
            self._R_snapshot = self.sr.R.copy()

        # M-change detection (CUSUM)
        self._cusum_episode_update(episode)

    # -- metrics -----------------------------------------------------------

    def get_metrics(self) -> dict:
        M = self.sr.M
        M_star = self._env.true_sr(self.sr.gamma)
        sr_error = float(np.mean(np.linalg.norm(M - M_star, axis=1)))
        U = self.meta_sr.all_uncertainties()
        return {
            "sr_error": sr_error,
            "U_mean": float(np.mean(U)),
            "recovery_mode": self.recovery_mode,
            "n_states_reset": len(self.S_reset),
            "q_value_mean": float(np.mean(self.Q)),
        }
