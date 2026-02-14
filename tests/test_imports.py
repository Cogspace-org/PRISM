"""Phase 0 validation: verify all imports work and package structure is correct."""


class TestDependencyImports:
    """Verify all external dependencies are installed and importable."""

    def test_import_gymnasium(self):
        import gymnasium
        assert hasattr(gymnasium, "make")

    def test_import_minigrid(self):
        import minigrid  # noqa: F401

    def test_import_numpy(self):
        import numpy as np
        assert hasattr(np, "ndarray")

    def test_import_scipy(self):
        from scipy import linalg
        assert hasattr(linalg, "eigh")

    def test_import_matplotlib(self):
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401

    def test_import_seaborn(self):
        import seaborn  # noqa: F401

    def test_import_pandas(self):
        import pandas as pd
        assert hasattr(pd, "DataFrame")

    def test_import_tqdm(self):
        from tqdm import tqdm  # noqa: F401


class TestPackageImports:
    """Verify the prism package structure is importable."""

    def test_import_prism(self):
        import prism
        assert prism.__version__ == "0.1.0"

    def test_import_prism_config(self):
        from prism.config import PRISMConfig, SRConfig, MetaSRConfig
        config = PRISMConfig()
        assert config.sr.gamma == 0.95
        assert config.meta_sr.buffer_size == 20

    def test_import_env_subpackage(self):
        from prism.env import state_mapper  # noqa: F401
        from prism.env import dynamics_wrapper  # noqa: F401
        from prism.env import perturbation_schedule  # noqa: F401

    def test_import_agent_subpackage(self):
        from prism.agent import sr_layer  # noqa: F401
        from prism.agent import meta_sr  # noqa: F401
        from prism.agent import controller  # noqa: F401
        from prism.agent import prism_agent  # noqa: F401

    def test_import_baselines_subpackage(self):
        from prism.baselines import sr_blind  # noqa: F401
        from prism.baselines import sr_count  # noqa: F401
        from prism.baselines import sr_bayesian  # noqa: F401

    def test_import_analysis_subpackage(self):
        from prism.analysis import calibration  # noqa: F401
        from prism.analysis import spectral  # noqa: F401
        from prism.analysis import visualization  # noqa: F401
        from prism.analysis import metrics  # noqa: F401
