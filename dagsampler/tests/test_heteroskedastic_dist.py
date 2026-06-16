"""Tests for the configurable base distribution of the heteroskedastic noise.

The heteroskedastic noise model scales a unit-variance base draw by a
parent-driven function, so the function sets the *conditional standard
deviation*. The base distribution is selectable via ``dist``; the default
``gaussian`` is byte-identical to the pre-0.4.0 behaviour.
"""

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from dagsampler import CausalDataGenerator


def _hetero_config(dist=None, n=40_000, seed=5):
    nm = {"name": "heteroskedastic", "func": "abs_parent_plus_const"}
    if dist is not None:
        nm["dist"] = dist
    return {
        "simulation_params": {"n_samples": n, "seed_structure": 3, "seed_data": seed},
        "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [("X", "Y")]},
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Y": {"type": "continuous",
                  "functional_form": {"name": "linear", "weights": {"X": 0.0}},
                  "noise_model": nm},
        },
    }


def test_default_gaussian_is_backward_compatible():
    """Omitting dist and setting dist='gaussian' give identical data."""
    a = CausalDataGenerator(_hetero_config(dist=None)).simulate()["data"]["Y"].to_numpy()
    b = CausalDataGenerator(_hetero_config(dist="gaussian")).simulate()["data"]["Y"].to_numpy()
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dist", ["gaussian", "student_t", "laplace", "uniform", "gamma", "exponential"])
def test_conditional_variance_matches_scale_across_dists(dist):
    """Unit-variance base => conditional variance equals noise_std**2 for any dist.

    The heteroskedastic scale is ``0.5*|X|+0.1``; within a narrow |X| bin the
    residual variance must match that scale squared, regardless of base dist.
    """
    df = CausalDataGenerator(_hetero_config(dist=dist)).simulate()["data"]
    x, y = df["X"].to_numpy(), df["Y"].to_numpy()
    for lo, hi in [(0.4, 0.6), (1.4, 1.6)]:
        m = (np.abs(x) >= lo) & (np.abs(x) < hi)
        assert m.sum() > 300
        target_sd = 0.5 * ((lo + hi) / 2) + 0.1
        emp_sd = y[m].std()
        assert 0.8 < emp_sd / target_sd < 1.2, (dist, lo, emp_sd, target_sd)


def test_student_t_base_is_heavier_tailed_than_gaussian():
    yg = CausalDataGenerator(_hetero_config(dist="gaussian")).simulate()["data"]["Y"].to_numpy()
    yt = CausalDataGenerator(_hetero_config(dist="student_t")).simulate()["data"]["Y"].to_numpy()
    # excess kurtosis: Student-t scale-mixed noise is markedly heavier-tailed
    assert kurtosis(yt) > kurtosis(yg) + 1.0


def test_gamma_base_is_skewed():
    yg = CausalDataGenerator(_hetero_config(dist="gamma")).simulate()["data"]["Y"].to_numpy()
    # standardized gamma base is right-skewed; |skew| clearly above ~0
    assert abs(skew(yg)) > 0.2


def test_cauchy_rejected_no_finite_variance():
    with pytest.raises(ValueError, match="cauchy has no finite variance"):
        CausalDataGenerator(_hetero_config(dist="cauchy", n=500)).simulate()


def test_student_t_low_df_rejected():
    cfg = _hetero_config(dist="student_t", n=500)
    cfg["node_params"]["Y"]["noise_model"]["df"] = 2
    with pytest.raises(ValueError, match="requires df > 2"):
        CausalDataGenerator(cfg).simulate()


def test_unknown_dist_rejected():
    with pytest.raises(ValueError, match="Unsupported heteroskedastic base dist"):
        CausalDataGenerator(_hetero_config(dist="not_a_dist", n=500)).simulate()
