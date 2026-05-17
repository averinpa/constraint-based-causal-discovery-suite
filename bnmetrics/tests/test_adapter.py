"""Tests for `bnmetrics.adapter._to_endpoints` (input normalisation)."""

from __future__ import annotations

import numpy as np
import pytest

from bnmetrics.adapter import _resolve_var, _to_endpoints
from bnmetrics.exceptions import BNMInputError
from bnmetrics.marks import EndpointMark
from tests.fixtures import chain_3


def test_pass_through_graphlike() -> None:
    g = chain_3()
    n, ep, names = _to_endpoints(g)
    assert n == 3
    assert ep.dtype == np.int8
    assert names == ("A", "B", "C")


def test_pass_through_ndarray() -> None:
    arr = np.zeros((3, 3), dtype=np.int8)
    arr[0, 1] = EndpointMark.ARROW
    arr[1, 0] = EndpointMark.TAIL
    n, ep, names = _to_endpoints(arr)
    assert n == 3
    assert ep[0, 1] == EndpointMark.ARROW
    assert names is None


def test_pass_through_list_of_lists() -> None:
    arr = [[0, 2, 0], [1, 0, 0], [0, 0, 0]]
    n, ep, names = _to_endpoints(arr)
    assert n == 3
    assert ep[0, 1] == EndpointMark.ARROW


def test_var_names_propagation() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    n, ep, names = _to_endpoints(arr, var_names=("X", "Y"))
    assert names == ("X", "Y")


def test_var_names_length_mismatch_rejected() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(BNMInputError, match="var_names"):
        _to_endpoints(arr, var_names=("X", "Y", "Z"))


def test_var_names_duplicates_rejected() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(BNMInputError, match="duplicates"):
        _to_endpoints(arr, var_names=("X", "X"))


def test_non_square_matrix_rejected() -> None:
    arr = np.zeros((3, 4), dtype=np.int8)
    with pytest.raises(BNMInputError, match="square"):
        _to_endpoints(arr)


def test_invalid_marks_rejected() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = 9  # invalid mark
    with pytest.raises(BNMInputError, match="endpoint marks"):
        _to_endpoints(arr)


def test_diagonal_must_be_no_edge() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 0] = EndpointMark.TAIL
    with pytest.raises(BNMInputError, match="diagonal"):
        _to_endpoints(arr)


def test_one_sided_edge_rejected() -> None:
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = EndpointMark.ARROW
    # mark at i not set → invariant violation
    with pytest.raises(BNMInputError, match="NO_EDGE"):
        _to_endpoints(arr)


def test_unsupported_input_type() -> None:
    with pytest.raises(BNMInputError, match="unsupported input type"):
        _to_endpoints("not a graph")  # type: ignore[arg-type]


# ---- networkx adapter ---------------------------------------------------


@pytest.fixture
def nx_module():
    nx = pytest.importorskip("networkx")
    return nx


def test_nx_digraph_directed_only(nx_module) -> None:
    g = nx_module.DiGraph()
    g.add_nodes_from(["A", "B", "C"])
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    n, ep, names = _to_endpoints(g)
    assert n == 3
    assert names == ("A", "B", "C")
    assert ep[0, 1] == EndpointMark.ARROW
    assert ep[1, 0] == EndpointMark.TAIL


def test_nx_digraph_with_undirected_edge(nx_module) -> None:
    g = nx_module.DiGraph()
    g.add_nodes_from(["A", "B", "C"])
    g.add_edge("A", "B", type="directed")
    g.add_edge("B", "C", type="undirected")
    n, ep, names = _to_endpoints(g)
    assert ep[1, 2] == EndpointMark.TAIL
    assert ep[2, 1] == EndpointMark.TAIL


def test_nx_digraph_bidirected_type_rejected(nx_module) -> None:
    g = nx_module.DiGraph()
    g.add_edge("A", "B", type="bidirected")
    with pytest.raises(BNMInputError, match="bidirected"):
        _to_endpoints(g)


def test_nx_digraph_both_directions_no_type_rejected(nx_module) -> None:
    g = nx_module.DiGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "A")
    with pytest.raises(BNMInputError, match="both"):
        _to_endpoints(g)


def test_nx_digraph_self_loop_rejected(nx_module) -> None:
    g = nx_module.DiGraph()
    g.add_edge("A", "A")
    with pytest.raises(BNMInputError, match="self-loop"):
        _to_endpoints(g)


def test_nx_digraph_node_ordering_preserved(nx_module) -> None:
    """list(g.nodes()) insertion order is the canonical ordering."""
    g = nx_module.DiGraph()
    # Insert in non-alphabetic order; var_names should follow insertion.
    g.add_nodes_from(["Z", "A", "M"])
    g.add_edge("Z", "A")
    n, ep, names = _to_endpoints(g)
    assert names == ("Z", "A", "M")
    assert ep[0, 1] == EndpointMark.ARROW


def test_nx_digraph_explicit_var_names_override(nx_module) -> None:
    g = nx_module.DiGraph()
    g.add_nodes_from(["Z", "A", "M"])
    g.add_edge("Z", "A")
    n, ep, names = _to_endpoints(g, var_names=("A", "M", "Z"))
    assert names == ("A", "M", "Z")
    # In the new ordering, A=0, M=1, Z=2; the edge Z→A is now 2→0.
    assert ep[2, 0] == EndpointMark.ARROW
    assert ep[0, 2] == EndpointMark.TAIL


# ---- _resolve_var ------------------------------------------------------


def test_resolve_var_int_in_range() -> None:
    assert _resolve_var(2, ("A", "B", "C"), 3) == 2


def test_resolve_var_int_out_of_range() -> None:
    with pytest.raises(BNMInputError, match="out of range"):
        _resolve_var(5, ("A", "B", "C"), 3)


def test_resolve_var_str_resolves() -> None:
    assert _resolve_var("B", ("A", "B", "C"), 3) == 1


def test_resolve_var_str_without_names_rejected() -> None:
    with pytest.raises(BNMInputError, match="no var_names"):
        _resolve_var("B", None, 3)


def test_resolve_var_str_not_found() -> None:
    with pytest.raises(BNMInputError, match="not found"):
        _resolve_var("Q", ("A", "B"), 2)


def test_resolve_var_invalid_type() -> None:
    with pytest.raises(BNMInputError, match="int or str"):
        _resolve_var(1.5, None, 3)  # type: ignore[arg-type]
