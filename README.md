# PRISM

**Predictive Representation for Introspective Spatial Metacognition**

A computational framework testing the hippocampal meta-map thesis: can an agent build an uncertainty map *iso-structural* to its world model, using only successor representation (SR) prediction errors?

## Architecture

```
prism/
├── core/           # SR layer, Meta-SR (uncertainty map), controller, grid agents
├── adapt/          # Targeted reset + triggered exploration (M1/M2/M3)
├── compete/        # CompeteAgent interface, Dyna-Q, R-max, Sliding-Window Q, SR+Q hybrid
├── env/            # TwoRoomEnv (9×9, 74 states), FourRoomEnv (13×13, 152 states), MiniGrid wrappers
├── baselines/      # SR-Blind, SR-Count, SR-Oracle, SR-Bayesian, Q-Learning, calibration baselines
├── analysis/       # ECE, metacognitive index, bootstrap CI, spectral decomposition, visualization
└── config.py       # Dataclass configs (SRConfig, MetaSRConfig, ControllerConfig, PRISMConfig)
```

### Core algorithm (3-layer stack)

| Layer | Class | Role |
|-------|-------|------|
| **SR** | `SRLayer` | Tabular successor representation, TD(0). Returns δ_M per transition. |
| **Meta-SR** | `MetaSR` | Builds U(s) uncertainty map from ‖δ_M‖. Same state indexing as M. |
| **Controller** | `PRISMController` | Adaptive ε(s) + exploration bonus V_explore = V + λ·U(s). |

### Adaptation (PRISM_adapt)

`GridPRISMAdaptAgent` extends the base agent with a closed-loop recovery cycle:

1. **Detect** — CUSUM on U_robust or oracle signal
2. **M1 — Reset** — M[s,:] ← I[s,:] for high-uncertainty states
3. **M2 — Explore** — activate exploration bonus during recovery only
4. **M3 — Targeted ε** — high ε only at affected states, not globally

### Hybrid (SR+Q)

`SRQHybridAgent` — Q-table for action selection (converges to Q\*), SR pipeline for monitoring and warm-start. Best of both: no SR policy floor, instant R-change transfer via V=M·R.

## Install

```bash
pip install -e .

# For MiniGrid environments (FourRooms, state_mapper, dynamics_wrapper):
pip install -e ".[minigrid]"
```

Requires Python 3.11+.

## Quick start

```python
from prism.core.sr_layer import SRLayer
from prism.core.meta_sr import MetaSR
from prism.env.two_room_env import TwoRoomEnv

# Create environment
env = TwoRoomEnv.default()

# Create SR + Meta-SR
sr = SRLayer(env.n_states, gamma=0.95)
meta = MetaSR(env.n_states)

# Run one transition
s = env.reset()
s_next, reward, done = env.step(s, action=1)

# Update and observe uncertainty
delta_M = sr.update(s, s_next, reward)
meta.observe(s, delta_M)

print(f"U({s}) = {meta.uncertainty(s):.3f}")
print(f"V({s}) = {sr.value(s):.3f}")
```

## Key results

- **Calibration**: U(s) tracks true SR error (MI = 0.50, ECE = 0.13)
- **Exploration**: uncertainty-guided discovery outperforms count-based bonuses
- **Recovery**: PRISM_adapt recovers 3.2× faster than SR-Blind after structural change
- **Hybrid**: SR+Q is the only agent competitive on both R-change and M-change

## References

- Stachenfeld, Botvinick & Gershman (2017). The hippocampus as a predictive map. *Nature Neuroscience*.
- Dayan (1993). Improving generalization for temporal difference learning.
- Russek, Momennejad, Botvinick, Gershman & Daw (2017). Predictive representations can link model-based reinforcement learning to model-free mechanisms. *PLOS Computational Biology*.
