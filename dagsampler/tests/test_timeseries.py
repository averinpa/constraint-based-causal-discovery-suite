"""Tests for time-series structure + oracle (Phase 1) and stationary SVAR data (Phase 2)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from dagsampler import (
    LaggedDSeparationOracle,
    LaggedVar,
    SVARParams,
    TimeSeriesDataGenerator,
    TimeSeriesSpec,
    random_ts_spec,
    simulate_svar,
    unroll,
)


def _lv(var: int, lag: int) -> LaggedVar:
    return LaggedVar(var, lag)


# ---------------------------------------------------------------------------- spec + unroll


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSpec(n_vars=2, max_lag=1, lagged_edges=[(0, 1, 2)])  # tau > max_lag
    with pytest.raises(ValueError):
        TimeSeriesSpec(n_vars=2, max_lag=1, contemp_edges=[(0, 0)])  # self-loop
    with pytest.raises(ValueError):
        TimeSeriesSpec(n_vars=2, max_lag=1, n_latent=2)  # n_latent >= n_vars


def test_unroll_is_dag_with_expected_edges() -> None:
    spec = TimeSeriesSpec(n_vars=2, max_lag=1, lagged_edges=[(0, 1, 1)], contemp_edges=[])
    g = unroll(spec, t_horizon=5)
    assert nx.is_directed_acyclic_graph(g)
    assert g.has_edge((0, 0), (1, 1))  # X_{t-1} -> Y_t across the window
    assert not g.has_edge((0, 0), (1, 0))  # no contemporaneous edge


def test_unroll_rejects_contemporaneous_cycle() -> None:
    spec = TimeSeriesSpec(n_vars=2, max_lag=0, contemp_edges=[(0, 1), (1, 0)])
    with pytest.raises(ValueError):
        unroll(spec)


# ---------------------------------------------------------------------------- oracle: d-sep


def test_ar1_chain_dsep() -> None:
    # X_{t-1} -> X_t (AR1). X_t and X_{t-2} are d-separated given X_{t-1}, dependent given {}.
    spec = TimeSeriesSpec(n_vars=1, max_lag=2, lagged_edges=[(0, 0, 1)])
    o = LaggedDSeparationOracle(spec)
    assert o(_lv(0, 0), _lv(0, -2), []) == 0.0  # dependent through X_{t-1}
    assert o(_lv(0, 0), _lv(0, -2), [_lv(0, -1)]) == 1.0  # blocked by X_{t-1}


def test_collider_opens_under_conditioning() -> None:
    # X_{t-1} -> Z_t <- Y_{t-1}: X_{t-1} and Y_{t-1} independent marginally, dependent given Z_t.
    spec = TimeSeriesSpec(n_vars=3, max_lag=1, lagged_edges=[(0, 2, 1), (1, 2, 1)])
    o = LaggedDSeparationOracle(spec)
    assert o(_lv(0, -1), _lv(1, -1), []) == 1.0  # collider closed -> d-separated
    assert o(_lv(0, -1), _lv(1, -1), [_lv(2, 0)]) == 0.0  # conditioning on collider opens it


def test_latent_confounder_makes_observed_dependent() -> None:
    # var 2 is latent, L_{t-1} -> X_t and L_{t-1} -> Y_t. X_t and Y_t are confounded and cannot be
    # separated by any observed set (L is unobserved).
    spec = TimeSeriesSpec(n_vars=3, max_lag=1, lagged_edges=[(2, 0, 1), (2, 1, 1)], n_latent=1)
    o = LaggedDSeparationOracle(spec)
    assert o.n_vars == 2  # only X, Y are observed
    assert o(_lv(0, 0), _lv(1, 0), []) == 0.0  # dependent via the latent parent
    with pytest.raises(IndexError):
        o(_lv(2, 0), _lv(0, 0), [])  # cannot query the latent


def test_is_ancestor() -> None:
    spec = TimeSeriesSpec(n_vars=2, max_lag=1, lagged_edges=[(0, 1, 1)])
    o = LaggedDSeparationOracle(spec)
    assert o.is_ancestor(_lv(0, -1), _lv(1, 0))  # X_{t-1} is an ancestor of Y_t
    assert not o.is_ancestor(_lv(1, 0), _lv(0, -1))  # present is not an ancestor of the past
    assert o.is_ancestor(_lv(1, 0), _lv(1, 0))  # reflexive


# ---------------------------------------------------------------------------- random generator


def test_random_ts_spec_is_valid_and_bounded_degree() -> None:
    spec = random_ts_spec(8, max_lag=2, degree=2.0, n_latent=2, seed=7)
    assert spec.n_vars == 10 and spec.n_observed == 8
    g = unroll(spec)  # must be a valid DAG
    assert nx.is_directed_acyclic_graph(g)
    o = LaggedDSeparationOracle(spec)
    # Conformance to the LaggedCITest protocol: attributes + float / result-with-p_value.
    assert isinstance(o.n_vars, int) and isinstance(o.max_lag, int)
    p = o(_lv(0, 0), _lv(1, 0), [])
    assert p in (0.0, 1.0)
    assert hasattr(o.details(_lv(0, 0), _lv(1, 0), []), "p_value")


def test_random_ts_spec_reproducible() -> None:
    a = random_ts_spec(6, max_lag=2, seed=42)
    b = random_ts_spec(6, max_lag=2, seed=42)
    assert a.lagged_edges == b.lagged_edges and a.contemp_edges == b.contemp_edges


# ---------------------------------------------------------------------------- Phase 2: SVAR data


def test_svar_shape_finite_and_reproducible() -> None:
    spec = TimeSeriesSpec(n_vars=3, max_lag=2, lagged_edges=[(0, 1, 1), (1, 2, 2)])
    p = SVARParams(n_timesteps=500, burn_in=100, seed=11)
    out = simulate_svar(spec, p)
    assert out["data"].shape == (500, 3)
    assert np.all(np.isfinite(out["data"].to_numpy()))
    assert list(out["data"].columns) == [0, 1, 2]
    again = simulate_svar(spec, SVARParams(n_timesteps=500, burn_in=100, seed=11))
    assert np.array_equal(out["data"].to_numpy(), again["data"].to_numpy())


def test_svar_is_stationary_not_explosive() -> None:
    # A gaussian-driven linear SVAR with a stabilized companion has bounded, finite variance.
    spec = random_ts_spec(6, max_lag=2, degree=2.0, seed=5)
    out = simulate_svar(spec, SVARParams(n_timesteps=4000, seed=5, target_spectral_radius=0.9))
    assert out["spectral_radius"] <= 0.9 + 1e-9
    v = out["data"].var().to_numpy()
    assert np.all(v < 100.0)  # no explosion
    # second half variance close to first half -> stationary, not drifting
    half = len(out["data"]) // 2
    v1 = out["data"].iloc[:half].var().to_numpy()
    v2 = out["data"].iloc[half:].var().to_numpy()
    assert np.all(np.abs(v1 - v2) / (v1 + 1e-6) < 0.6)


def test_svar_ar1_recovers_coefficient() -> None:
    # x_t = a x_{t-1} + e  =>  lag-1 autocorrelation ~ a (the tied coefficient).
    spec = TimeSeriesSpec(n_vars=1, max_lag=1, lagged_edges=[(0, 0, 1)])
    out = simulate_svar(spec, SVARParams(n_timesteps=20000, seed=3))
    a = out["coefficients"]["lagged"][(0, 0, 1)]
    x = out["data"][0].to_numpy()
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(ac1 - a) < 0.05


def test_svar_ties_one_coefficient_per_edge() -> None:
    spec = TimeSeriesSpec(n_vars=3, max_lag=2, lagged_edges=[(0, 1, 1), (1, 2, 2)], contemp_edges=[(0, 2)])
    out = simulate_svar(spec, SVARParams(n_timesteps=200, seed=1))
    assert set(out["coefficients"]["lagged"]) == {(0, 1, 1), (1, 2, 2)}
    assert set(out["coefficients"]["contemp"]) == {(0, 2)}


def test_svar_mixed_types() -> None:
    spec = TimeSeriesSpec(n_vars=3, max_lag=1, lagged_edges=[(0, 1, 1), (1, 2, 1)])
    out = simulate_svar(
        spec,
        SVARParams(
            n_timesteps=800,
            seed=7,
            var_types={0: "continuous", 1: "binary", 2: "categorical"},
            cardinality=4,
        ),
    )
    assert set(np.unique(out["data"][1].to_numpy())) <= {0.0, 1.0}
    assert set(np.unique(out["data"][2].to_numpy())) <= {0.0, 1.0, 2.0, 3.0}
    assert out["data"][0].nunique() > 100  # continuous stays continuous


def test_svar_drops_latent_columns() -> None:
    spec = TimeSeriesSpec(n_vars=3, max_lag=1, lagged_edges=[(2, 0, 1), (2, 1, 1)], n_latent=1)
    out = simulate_svar(spec, SVARParams(n_timesteps=300, seed=2))
    assert list(out["data"].columns) == [0, 1]  # latent series 2 dropped
    assert out["oracle"].n_vars == 2


@pytest.mark.parametrize("dist", ["student_t", "laplace", "cauchy", "gamma", "exponential"])
def test_svar_heavy_tail_innovations_run(dist: str) -> None:
    # Heavy-tailed / skewed innovations (the GFCM tail stressors) still produce finite float series.
    spec = TimeSeriesSpec(n_vars=2, max_lag=1, lagged_edges=[(0, 1, 1)])
    out = simulate_svar(spec, SVARParams(n_timesteps=500, seed=4, innovation=dist))
    assert out["data"].shape == (500, 2)
    assert np.all(np.isfinite(out["data"].to_numpy()))


def test_svar_burn_in_policy_and_tanh_mechanism() -> None:
    spec = random_ts_spec(4, max_lag=1, seed=8)
    out = simulate_svar(
        spec, SVARParams(n_timesteps=300, seed=8, stationarity="burn_in", mechanism="tanh")
    )
    assert out["data"].shape[0] == 300
    assert np.all(np.isfinite(out["data"].to_numpy()))


def test_ts_data_generator_wrapper_and_rejects_unknown_params() -> None:
    config = {
        "graph_params": {"n_vars": 3, "max_lag": 1, "lagged_edges": [(0, 1, 1)]},
        "simulation_params": {"n_timesteps": 250, "seed": 9},
    }
    out = TimeSeriesDataGenerator(config).simulate()
    assert out["data"].shape == (250, 3)
    with pytest.raises(ValueError):
        TimeSeriesDataGenerator(
            {"graph_params": {"n_vars": 2, "max_lag": 1}, "simulation_params": {"bogus": 1}}
        )
