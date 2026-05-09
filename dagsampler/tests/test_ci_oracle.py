"""Tests for ``CausalDataGenerator.as_ci_oracle()`` and ``DSeparationOracle``."""

from __future__ import annotations

import networkx as nx
import pytest

from dagsampler import CausalDataGenerator, DSeparationOracle


def _chain_config(n_samples: int = 30) -> dict:
    return {
        "simulation_params": {
            "n_samples": n_samples,
            "seed_structure": 1,
            "seed_data": 2,
            "binary_proportion": 0.0,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Z", "Y"],
            "edges": [["X", "Z"], ["Z", "Y"]],
        },
    }


def _collider_config(n_samples: int = 30) -> dict:
    return {
        "simulation_params": {
            "n_samples": n_samples,
            "seed_structure": 1,
            "seed_data": 2,
            "binary_proportion": 0.0,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Z", "Y"],
            "edges": [["X", "Z"], ["Y", "Z"]],
        },
    }


def _fork_config(n_samples: int = 30) -> dict:
    return {
        "simulation_params": {
            "n_samples": n_samples,
            "seed_structure": 1,
            "seed_data": 2,
            "binary_proportion": 0.0,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Z", "Y"],
            "edges": [["Z", "X"], ["Z", "Y"]],
        },
    }


def test_oracle_satisfies_citest_shape():
    gen = CausalDataGenerator(_chain_config())
    gen.simulate()
    oracle = gen.as_ci_oracle()

    assert isinstance(oracle.n_vars, int)
    assert oracle.n_vars == 3
    assert callable(oracle)
    p = oracle(0, 1, [])
    assert isinstance(p, float)
    result = oracle.details(0, 1, [])
    assert hasattr(result, "p_value")
    assert isinstance(result.p_value, float)


def test_oracle_n_vars_matches_data_columns():
    gen = CausalDataGenerator(_chain_config())
    result = gen.simulate()
    oracle = gen.as_ci_oracle()
    assert oracle.n_vars == result["data"].shape[1]
    assert oracle.var_names == tuple(result["data"].columns)


def test_oracle_chain_d_separation():
    gen = CausalDataGenerator(_chain_config())
    gen.simulate()
    oracle = gen.as_ci_oracle()
    cols = list(gen.data.columns)  # ["X", "Y", "Z"] alphabetically
    x_idx = cols.index("X")
    y_idx = cols.index("Y")
    z_idx = cols.index("Z")

    # X and Y are connected through Z; conditioning on Z separates them
    assert oracle(x_idx, y_idx, []) == 0.0
    assert oracle(x_idx, y_idx, [z_idx]) == 1.0


def test_oracle_collider_d_separation():
    gen = CausalDataGenerator(_collider_config())
    gen.simulate()
    oracle = gen.as_ci_oracle()
    cols = list(gen.data.columns)
    x_idx = cols.index("X")
    y_idx = cols.index("Y")
    z_idx = cols.index("Z")

    # X and Y are marginally independent; conditioning on collider Z opens the path
    assert oracle(x_idx, y_idx, []) == 1.0
    assert oracle(x_idx, y_idx, [z_idx]) == 0.0


def test_oracle_fork_d_separation():
    gen = CausalDataGenerator(_fork_config())
    gen.simulate()
    oracle = gen.as_ci_oracle()
    cols = list(gen.data.columns)
    x_idx = cols.index("X")
    y_idx = cols.index("Y")
    z_idx = cols.index("Z")

    assert oracle(x_idx, y_idx, []) == 0.0
    assert oracle(x_idx, y_idx, [z_idx]) == 1.0


def test_as_ci_oracle_before_simulate_raises():
    gen = CausalDataGenerator(_chain_config())
    with pytest.raises(RuntimeError, match="simulate"):
        gen.as_ci_oracle()


def test_oracle_rejects_out_of_range_index():
    gen = CausalDataGenerator(_chain_config())
    gen.simulate()
    oracle = gen.as_ci_oracle()
    with pytest.raises(IndexError):
        oracle(0, 99, [])
    with pytest.raises(IndexError):
        oracle(0, 1, [42])


def test_oracle_rejects_self_pair():
    gen = CausalDataGenerator(_chain_config())
    gen.simulate()
    oracle = gen.as_ci_oracle()
    with pytest.raises(ValueError, match="differ"):
        oracle(0, 0, [])


def test_oracle_constructor_validates_var_names():
    dag = nx.DiGraph([("X", "Y")])
    with pytest.raises(ValueError, match="not in the DAG"):
        DSeparationOracle(dag, ["X", "Z"])
    with pytest.raises(ValueError, match="unique"):
        DSeparationOracle(dag, ["X", "X"])
