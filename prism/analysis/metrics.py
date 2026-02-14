"""Statistical analysis: Mann-Whitney, bootstrap CI, effect sizes.

References:
    master.md section 6 (statistical protocol)
"""


def bootstrap_ci(data, n_resamples=10000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    raise NotImplementedError("Phase 3")


def compare_conditions(results_dict):
    """Mann-Whitney U tests with Holm-Bonferroni correction."""
    raise NotImplementedError("Phase 3")
