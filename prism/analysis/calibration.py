"""Psychophysics calibration metrics: ECE, reliability diagrams, MI.

References:
    master.md section 6.1 (metrics)
"""

import numpy as np


def expected_calibration_error(confidences, accuracies, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    raise NotImplementedError("Phase 2")


def reliability_diagram(confidences, accuracies, n_bins=10):
    """Compute data for a reliability diagram."""
    raise NotImplementedError("Phase 2")


def metacognitive_index(U, true_errors):
    """Compute MI = Spearman correlation between U(s) and true SR error."""
    raise NotImplementedError("Phase 2")
