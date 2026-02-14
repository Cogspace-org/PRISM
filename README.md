# PRISM -- Predictive Representation for Introspective Spatial Metacognition

Computational test of the hippocampal meta-map thesis: the successor representation
as a unified substrate for cognition and metacognition, evaluated with psychophysics tools.

## Setup

```bash
# Create virtual environment (requires Python 3.11)
py -3.11 -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify environment
python scripts/verify_env.py

# Run tests
pytest
```

## Project Structure

- `prism/` -- Main Python package
  - `env/` -- MiniGrid wrappers and state mapping
  - `agent/` -- SR layer, meta-SR, controller, PRISM agent
  - `baselines/` -- Comparison agents
  - `analysis/` -- Calibration metrics, spectral analysis, visualization
- `experiments/` -- Experiment scripts (A, B, C)
- `tests/` -- Test suite
- `notebooks/` -- Jupyter notebooks for analysis
- `scripts/` -- Utility scripts

## References

See `master.md` for the full theoretical framework and literature review.
