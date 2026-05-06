"""Input handling: DataFrame vs ndarray, var_names propagation, factory equivalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbcd import BackgroundKnowledge, FisherZ, pc
from cbcd.exceptions import CBCDInputError
from tests.algorithms.test_pc_fisherz import _sample_linear_gaussian
from tests.fixtures import ALL_FIXTURES


def test_dataframe_and_ndarray_match() -> None:
    rng = np.random.default_rng(0)
    dag, _ = ALL_FIXTURES["chain"]()
    data = _sample_linear_gaussian(dag, 5000, rng)
    df = pd.DataFrame(data, columns=["a", "b", "c"])

    out_arr = pc(data, alpha=0.05)
    out_df = pc(df, alpha=0.05)
    assert np.array_equal(out_arr.endpoints, out_df.endpoints)
    assert out_df.var_names == ("a", "b", "c")
    assert out_arr.var_names is None


def test_explicit_var_names() -> None:
    rng = np.random.default_rng(0)
    dag, _ = ALL_FIXTURES["chain"]()
    data = _sample_linear_gaussian(dag, 5000, rng)
    out = pc(data, var_names=["x", "y", "z"], alpha=0.05)
    assert out.var_names == ("x", "y", "z")


def test_string_and_explicit_fisherz_match() -> None:
    rng = np.random.default_rng(0)
    dag, _ = ALL_FIXTURES["chain"]()
    data = _sample_linear_gaussian(dag, 5000, rng)
    fz = FisherZ(data)
    out_string = pc(data, ci_test="fisherz", alpha=0.05)
    out_explicit = pc(data, ci_test=fz, alpha=0.05)
    assert np.array_equal(out_string.endpoints, out_explicit.endpoints)


def test_unknown_ci_test_raises() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    with pytest.raises(CBCDInputError, match="unknown CI test"):
        pc(data, ci_test="not-a-real-test")  # type: ignore[arg-type]


def test_alpha_out_of_range_raises() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    with pytest.raises(CBCDInputError):
        pc(data, alpha=0.0)
    with pytest.raises(CBCDInputError):
        pc(data, alpha=1.5)


def test_n_jobs_unsupported_raises() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    with pytest.raises(CBCDInputError, match="n_jobs"):
        pc(data, n_jobs=4)


def test_background_required_edge_respected() -> None:
    rng = np.random.default_rng(0)
    dag, _ = ALL_FIXTURES["chain"]()
    data = _sample_linear_gaussian(dag, 5000, rng)
    bk = BackgroundKnowledge(required_directed=frozenset({(0, 1)}))
    out = pc(data, alpha=0.05, background=bk)
    # 0 → 1 must be directed.
    from cbcd.graph.marks import EndpointMark

    assert out.endpoints[0, 1] == EndpointMark.ARROW
    assert out.endpoints[1, 0] == EndpointMark.TAIL


def test_background_forbidden_adjacent_respected() -> None:
    rng = np.random.default_rng(0)
    dag, _ = ALL_FIXTURES["chain"]()
    data = _sample_linear_gaussian(dag, 5000, rng)
    # Prohibit adjacency between 0 and 1 even though it's in the true graph;
    # PC should never recover the edge.
    bk = BackgroundKnowledge(forbidden_adjacent=frozenset({frozenset({0, 1})}))
    out = pc(data, alpha=0.05, background=bk)
    from cbcd.graph.marks import EndpointMark

    assert out.endpoints[0, 1] == EndpointMark.NO_EDGE
    assert out.endpoints[1, 0] == EndpointMark.NO_EDGE
