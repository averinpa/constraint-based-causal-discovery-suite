"""Tests for the 'shape' (tail-shape / skew) noise model.

The shape noise model lets a parent drive the *skewness* of a continuous
child's noise while holding its conditional mean and variance fixed. The two
properties that make it a clean higher-moment edge -- (a) mean and variance
invariant to the shape parameter, (b) skewness a monotone function of it --
are asserted directly on the standardizing helper and end-to-end through
``CausalDataGenerator.simulate``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from dagsampler import CausalDataGenerator
from dagsampler.causal_sim import (
    SHAPE_FN_REGISTRY,
    _standardized_skewnorm,
)


# ---------------------------------------------------------------------------
# Helper: standardized skew-normal draws
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_standardized_skewnorm_preserves_mean_and_variance(sigma):
    """Mean ~ 0 and variance ~ sigma**2 for every shape parameter."""
    rng = np.random.default_rng(0)
    n = 400_000
    for alpha in (-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0):
        noise = _standardized_skewnorm(np.full(n, alpha), sigma, rng)
        assert abs(noise.mean()) < 0.02 * sigma, alpha
        assert abs(noise.var() - sigma**2) < 0.05 * sigma**2, alpha


def test_standardized_skewnorm_skew_tracks_alpha():
    """Sample skewness is ~0 at alpha=0, sign-matched and monotone in alpha."""
    rng = np.random.default_rng(1)
    n = 400_000
    alphas = [-12.0, -4.0, 0.0, 4.0, 12.0]
    skews = [skew(_standardized_skewnorm(np.full(n, a), 1.0, rng)) for a in alphas]

    assert abs(skews[2]) < 0.02                      # alpha = 0 -> symmetric
    assert skews[0] < -0.4 and skews[1] < -0.1       # negative alpha -> left skew
    assert skews[3] > 0.1 and skews[4] > 0.4         # positive alpha -> right skew
    assert all(a < b for a, b in zip(skews, skews[1:]))  # monotone increasing


def test_standardized_skewnorm_per_row_alpha_vectorizes():
    """A per-row alpha vector is honoured (no broadcast error, finite output)."""
    rng = np.random.default_rng(2)
    alpha = np.linspace(-8.0, 8.0, 5000)
    noise = _standardized_skewnorm(alpha, 1.0, rng)
    assert noise.shape == alpha.shape
    assert np.all(np.isfinite(noise))


# ---------------------------------------------------------------------------
# End-to-end: a pure tail-shape edge X -> Y
# ---------------------------------------------------------------------------
def _shape_edge_config(n_samples=60_000, seed_data=11):
    """X ~ N(0,1); Y has zero mean-weight on X but X drives Y's noise skew."""
    return {
        "simulation_params": {
            "n_samples": n_samples,
            "seed_structure": 7,
            "seed_data": seed_data,
            "store_ci_oracle": True,
            "ci_oracle_max_cond_set": 0,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y"],
            "edges": [("X", "Y")],
        },
        "node_params": {
            "X": {
                "type": "continuous",
                "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0},
            },
            "Y": {
                "type": "continuous",
                # zero mean-weight: X does NOT shift Y's conditional mean
                "functional_form": {"name": "linear", "weights": {"X": 0.0}},
                "noise_model": {
                    "name": "shape",
                    "func": "skew_first_parent",  # alpha = 4 * X
                    "std": 1.0,
                },
            },
        },
    }


def test_shape_edge_preserves_conditional_mean_and_variance():
    """X drives only the skew of Y: conditional mean/variance flat, skew not."""
    result = CausalDataGenerator(_shape_edge_config()).simulate()
    df = result["data"]
    x, y = df["X"].to_numpy(), df["Y"].to_numpy()

    # No linear/mean dependence: correlation ~ 0
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.03

    lo, hi = y[x < -0.6], y[x > 0.6]
    assert lo.size > 2000 and hi.size > 2000

    # (a) conditional mean preserved
    assert abs(lo.mean() - hi.mean()) < 0.08
    assert abs(lo.mean()) < 0.08 and abs(hi.mean()) < 0.08

    # (b) conditional variance preserved
    assert 0.85 < (lo.var() / hi.var()) < 1.18

    # (c) conditional skew is what moves: opposite signs in the two tails
    s_lo, s_hi = skew(lo), skew(hi)
    assert s_lo < -0.2 and s_hi > 0.2
    assert (s_hi - s_lo) > 0.5


def test_shape_edge_oracle_marks_dependence():
    """The structural edge X->Y survives in the d-separation oracle."""
    result = CausalDataGenerator(_shape_edge_config(n_samples=2000)).simulate()
    assert "ci_oracle" in result and len(result["ci_oracle"]) > 0
    # X and Y are adjacent, hence never d-separated (dependent under faithfulness).
    df = result["data"]
    assert df["Y"].notna().all() and np.isfinite(df["Y"].to_numpy()).all()


# ---------------------------------------------------------------------------
# Config surface: registry, callable funcs, error path, backward-compat
# ---------------------------------------------------------------------------
def test_shape_registry_contains_expected_funcs():
    assert {
        "skew_first_parent",
        "skew_tanh_first_parent",
        "skew_mean_parents",
    } <= set(SHAPE_FN_REGISTRY)


def test_shape_noise_accepts_callable_func():
    cfg = _shape_edge_config(n_samples=3000)
    cfg["node_params"]["Y"]["noise_model"]["func"] = (
        lambda p: 5.0 * p.iloc[:, 0].to_numpy()
    )
    result = CausalDataGenerator(cfg).simulate()
    assert np.isfinite(result["data"]["Y"].to_numpy()).all()


def test_shape_noise_unknown_func_raises():
    cfg = _shape_edge_config(n_samples=500)
    cfg["node_params"]["Y"]["noise_model"]["func"] = "not_a_real_func"
    with pytest.raises(ValueError, match="Unsupported shape-noise func"):
        CausalDataGenerator(cfg).simulate()


def test_shape_is_opt_in_not_a_random_default():
    """Backward-compat: the random noise default must not select 'shape'."""
    import inspect

    from dagsampler import causal_sim

    src = inspect.getsource(causal_sim.CausalDataGenerator._apply_noise_model)
    # 'shape' must be reachable only via explicit config, never the random default.
    assert "['additive', 'multiplicative', 'heteroskedastic']" in src
    assert "'shape'" not in "['additive', 'multiplicative', 'heteroskedastic']"
