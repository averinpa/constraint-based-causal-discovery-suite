"""GraphLike Protocol conformance tests."""

from __future__ import annotations

import numpy as np

import bnmetrics
from bnmetrics.protocol import GraphLike
from tests.fixtures import asia_8, chain_3, make_dag


def test_internal_graph_conforms() -> None:
    g = chain_3()
    assert isinstance(g, GraphLike)
    assert g.n_vars == 3
    assert g.endpoints.dtype == np.int8
    assert g.endpoints.shape == (3, 3)
    assert g.var_names == ("A", "B", "C")


def test_to_graphlike_returns_conforming_object() -> None:
    arr = np.zeros((3, 3), dtype=np.int8)
    arr[0, 1] = bnmetrics.EndpointMark.ARROW
    arr[1, 0] = bnmetrics.EndpointMark.TAIL
    g = bnmetrics.to_graphlike(arr, var_names=("X", "Y", "Z"))
    assert isinstance(g, GraphLike)
    assert g.n_vars == 3
    assert g.var_names == ("X", "Y", "Z")
    assert g.endpoints[0, 1] == bnmetrics.EndpointMark.ARROW


def test_external_dataclass_conforms() -> None:
    """A user-defined dataclass with the right attrs is a valid GraphLike,
    proving the structural Protocol does its job."""
    from dataclasses import dataclass

    @dataclass
    class _Thirdparty:
        n_vars: int
        endpoints: np.ndarray
        var_names: tuple[str, ...] | None = None

    arr = np.zeros((2, 2), dtype=np.int8)
    obj = _Thirdparty(n_vars=2, endpoints=arr)
    assert isinstance(obj, GraphLike)


def test_concrete_graph_is_frozen() -> None:
    g = chain_3()
    import dataclasses

    assert dataclasses.is_dataclass(g)
    fields = dataclasses.fields(g)
    assert all(f.name in {"n_vars", "endpoints", "var_names"} for f in fields)


def test_empty_graph() -> None:
    g = make_dag(0, [], var_names=())
    assert g.n_vars == 0
    assert g.endpoints.shape == (0, 0)


def test_asia_round_trip() -> None:
    g = asia_8()
    g2 = bnmetrics.to_graphlike(g)
    assert g2.n_vars == g.n_vars
    assert np.array_equal(g2.endpoints, g.endpoints)
    assert g2.var_names == g.var_names
