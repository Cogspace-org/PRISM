"""Adaptive PRISM agent: targeted SR reset + triggered exploration.

Extends GridPRISMAgent with three mechanisms:
  M1 — Targeted reset: M[s,:] <- I[s,:] for high-uncertainty states
  M2 — Triggered exploration: V_explore bonus only during recovery_mode
  M3 — Targeted epsilon: high epsilon only for states in S_reset

Detection modes:
  "oracle" — runner signals the agent via notify_change() at M-change
  "auto"   — MetaSR detect_change() with warmup guard

Plus two ablation variants:
  GridPRISMResetOnly  — M1 only (reset without exploration bonus)
  GridPRISMExploreOnly — M2+M3 only (exploration without reset)
"""

import numpy as np
from prism.core.grid_agent import GridPRISMAgent
from prism.adapt.config import AdaptConfig


class GridPRISMAdaptAgent(GridPRISMAgent):
    """PRISM agent with closed-loop adaptation: detect -> act -> verify.

    When structural change is detected (oracle or auto):
      1. Identify S_reset = {s : U(s) > theta_reset}
      2. Reset M[s,:] = I[s,:] for s in S_reset  (M1)
      3. Enter recovery_mode: V_explore gets bonus (M2), epsilon high for S_reset (M3)
      4. Exit recovery_mode after n_patience steps without new change signal

    Parameters
    ----------
    env : TwoRoomEnv
        Environment instance.
    config : AdaptConfig or None
        Configuration. Uses defaults if None.
    seed : int
        Random seed.
    """

    def __init__(self, env, config=None, seed=42):
        if config is None:
            config = AdaptConfig()
        self._config = config

        super().__init__(
            env,
            gamma=config.gamma,
            alpha_M=config.alpha_M,
            alpha_R=config.alpha_R,
            epsilon_min=config.epsilon_min,
            epsilon_max=config.epsilon_max,
            lambda_explore=config.lambda_recover,
            bonus_mode=config.bonus_mode,
            buffer_size=config.buffer_size,
            decay=config.decay,
            beta=config.beta,
            theta_change=config.theta_change,
            seed=seed,
            adaptive_beta=config.adaptive_beta,
        )

        # Adaptation state
        self.recovery_mode = False
        self.S_reset = set()
        self._patience_counter = 0
        self._n_states_reset_history = []
        self._trigger_count = 0

        # CUSUM state (cusum detection mode)
        self._cusum_S = 0.0
        self._cusum_baseline = []
        self._cusum_mu0 = None
        self._cusum_sigma0 = None
        self._cusum_ready = False
        self._cusum_history = []  # [(episode, x_t, S_t, detected)]

    # ------------------------------------------------------------------
    # Oracle detection interface
    # ------------------------------------------------------------------
    def notify_change(self):
        """Called by runner when structural change occurs (oracle mode).

        Triggers adaptation if max_triggers not exceeded.
        """
        if self._trigger_count < self._config.max_triggers:
            self._trigger_reset()

    # ------------------------------------------------------------------
    # CUSUM per-episode update
    # ------------------------------------------------------------------
    def cusum_episode_update(self, episode):
        """Per-episode CUSUM update. Returns True if change detected."""
        if self._config.detection_mode != "cusum":
            return False
        if self.recovery_mode:
            return False  # already triggered
        if self._trigger_count >= self._config.max_triggers:
            return False

        x_t = self.meta_sr.robust_uncertainty_stat(
            method=self._config.cusum_stat_method,
            k=self._config.cusum_stat_k,
        )

        # Skip warmup episodes (U is non-stationary during initial learning)
        if episode < self._config.cusum_warmup:
            self._cusum_history.append((episode, x_t, 0.0, False))
            return False

        # Phase 1: collect baseline (after warmup)
        if not self._cusum_ready:
            self._cusum_baseline.append(x_t)
            if len(self._cusum_baseline) >= self._config.cusum_baseline_window:
                arr = np.array(self._cusum_baseline)
                self._cusum_mu0 = float(np.mean(arr))
                self._cusum_sigma0 = max(float(np.std(arr)), 1e-6)
                self._cusum_ready = True
                self._cusum_S = 0.0
            self._cusum_history.append((episode, x_t, 0.0, False))
            return False

        # Phase 2: CUSUM accumulation
        k = self._config.cusum_slack_factor * self._cusum_sigma0
        h = self._config.cusum_threshold_factor * self._cusum_sigma0
        self._cusum_S = max(0.0, self._cusum_S + (x_t - self._cusum_mu0 - k))
        detected = self._cusum_S > h
        self._cusum_history.append((episode, x_t, self._cusum_S, detected))

        if detected:
            self._trigger_reset()

        return detected

    # ------------------------------------------------------------------
    # M1 — Targeted reset
    # ------------------------------------------------------------------
    def _trigger_reset(self):
        """Identify high-uncertainty states and reset their M rows + MetaSR."""
        theta = self._config.theta_reset
        self.S_reset = {
            s for s in range(self.n_states)
            if self.meta_sr.uncertainty(s) > theta
        }
        # M1: Reset M rows to identity for states in S_reset
        I = np.eye(self.n_states)
        for s in self.S_reset:
            self.sr.M[s, :] = I[s, :]

        # Reset MetaSR for S_reset states so their U follows cold-start decay
        for s in self.S_reset:
            self.meta_sr._buffers[s].clear()
            self.meta_sr.visit_counts[s] = 0
            self.meta_sr._U_cache[s] = self.meta_sr.U_prior

        self.recovery_mode = True
        self._patience_counter = 0
        self._trigger_count += 1
        self._n_states_reset_history.append(len(self.S_reset))

    # ------------------------------------------------------------------
    # Recovery exit check
    # ------------------------------------------------------------------
    def _check_recovery_done(self):
        """Exit recovery_mode after n_patience steps without NEW change signal.

        Excludes S_reset states from check — their elevated U is expected
        during re-learning (M rows were reset to identity) and should not
        prevent recovery exit.
        """
        recent_unique = set(self.meta_sr._recent_visits)
        check_states = recent_unique - self.S_reset

        if len(check_states) > 0:
            change_score = np.mean(
                [self.meta_sr.uncertainty(s) for s in check_states]
            )
            still_changing = change_score > self.meta_sr.theta_change
        else:
            # All recent visits are in S_reset — no evidence of new change
            still_changing = False

        if not still_changing:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        if self._patience_counter >= self._config.n_patience:
            self.recovery_mode = False
            self.S_reset = set()

    # ------------------------------------------------------------------
    # M3 — Targeted epsilon (overrides GridPRISMAgent._adaptive_epsilon)
    # ------------------------------------------------------------------
    def _adaptive_epsilon(self, s):
        """High epsilon for S_reset states during recovery, low otherwise."""
        if self.recovery_mode and s in self.S_reset:
            return self._config.epsilon_max
        return self._config.epsilon_min

    # ------------------------------------------------------------------
    # M2 — Triggered exploration bonus (overrides GridPRISMAgent._exploration_value)
    # ------------------------------------------------------------------
    def _exploration_value(self, s):
        """V + lambda * bonus during recovery, V only otherwise."""
        V = self.sr.value(s)
        if self.recovery_mode:
            return V + self._config.lambda_recover * self._exploration_bonus(s)
        return V

    # ------------------------------------------------------------------
    # Update loop (overrides GridPRISMAgent.update)
    # ------------------------------------------------------------------
    def update(self, s, s_next, reward):
        """Update SR + MetaSR, then check adaptation triggers."""
        # Parent update: SR TD(0) + MetaSR observe
        delta_M = self.sr.update(s, s_next, reward)
        self.meta_sr.observe(s, delta_M)
        self.visit_counts[s] += 1
        self.total_steps += 1

        # Auto-detection mode: warmup guard + MetaSR detect_change
        if self._config.detection_mode == "auto":
            if self.total_steps < self._config.min_steps_before_adapt:
                return
            if not self.recovery_mode:
                if (self._trigger_count < self._config.max_triggers
                        and self.detect_change()):
                    self._trigger_reset()
            else:
                self._check_recovery_done()
        else:
            # Oracle and CUSUM: per-step only checks recovery exit.
            # Oracle triggers via notify_change(); CUSUM triggers via cusum_episode_update().
            if self.recovery_mode:
                self._check_recovery_done()


# ======================================================================
# Ablation variants
# ======================================================================

class GridPRISMResetOnly(GridPRISMAdaptAgent):
    """Ablation: M1 only (reset without exploration bonus or targeted epsilon).

    Tests whether targeted reset alone accelerates recovery.
    """

    def _adaptive_epsilon(self, s):
        """Always low epsilon (no M3)."""
        return self._config.epsilon_min

    def _exploration_value(self, s):
        """Always V(s) only (no M2 bonus)."""
        return self.sr.value(s)


class GridPRISMExploreOnly(GridPRISMAdaptAgent):
    """Ablation: M2+M3 only (exploration without M reset).

    Tests whether triggered exploration alone accelerates recovery.
    S_reset is computed (for epsilon targeting) but M rows are NOT reset.
    """

    def _trigger_reset(self):
        """Compute S_reset and enter recovery, but do NOT reset M rows."""
        theta = self._config.theta_reset
        self.S_reset = {
            s for s in range(self.n_states)
            if self.meta_sr.uncertainty(s) > theta
        }
        # NO M[s,:] = I[s,:] — that's the point of this ablation
        self.recovery_mode = True
        self._patience_counter = 0
        self._trigger_count += 1
        self._n_states_reset_history.append(len(self.S_reset))
