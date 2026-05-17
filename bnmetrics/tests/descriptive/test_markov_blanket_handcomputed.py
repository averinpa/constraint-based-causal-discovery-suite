"""Hand-computed Markov-blanket extraction tests."""

from __future__ import annotations

import bnmetrics
from tests.fixtures import collider_3, diamond_4, m_4, y_4


def test_collider_blanket_apex() -> None:
    """In A→C, B→C, the MB of C is {C, A, B}."""
    g = collider_3()
    indices = bnmetrics.markov_blanket_indices(g, "C")
    assert indices == (0, 1, 2)


def test_collider_blanket_parent_includes_co_parent() -> None:
    """In A→C, B→C, MB of A = {A, C, B} (B is co-parent of child C)."""
    g = collider_3()
    indices = bnmetrics.markov_blanket_indices(g, "A")
    assert indices == (0, 1, 2)


def test_y_blanket_root() -> None:
    """In Y_4 (A→C, B→C, C→D), MB of A = {A, C, B} (parent of A's child)."""
    g = y_4()
    indices = bnmetrics.markov_blanket_indices(g, "A")
    assert indices == (0, 1, 2)


def test_y_blanket_leaf() -> None:
    """In Y_4, MB of D = {D, C} (D's parent only)."""
    g = y_4()
    indices = bnmetrics.markov_blanket_indices(g, "D")
    assert indices == (2, 3)


def test_diamond_blanket_apex() -> None:
    """In diamond (A→B, A→C, B→D, C→D), MB of A = {A, B, C}."""
    g = diamond_4()
    indices = bnmetrics.markov_blanket_indices(g, "A")
    assert indices == (0, 1, 2)


def test_blanket_returns_subgraph_with_renumbered_indices() -> None:
    """`markov_blanket` returns a `_Graph` with 0..k-1 indices and original
    var_names for the included variables."""
    g = y_4()  # A=0, B=1, C=2, D=3
    sub = bnmetrics.markov_blanket(g, "D")
    assert sub.n_vars == 2
    assert sub.var_names == ("C", "D")
    # In the sub-graph, C is now index 0 and D index 1; the C→D edge survives.
    assert sub.endpoints[0, 1] == bnmetrics.EndpointMark.ARROW
    assert sub.endpoints[1, 0] == bnmetrics.EndpointMark.TAIL


def test_blanket_metrics_compose() -> None:
    """A sub-graph from `markov_blanket` is itself a GraphLike and feeds
    into all descriptive metrics."""
    g = y_4()
    sub = bnmetrics.markov_blanket(g, "C")  # MB(C) = {A, B, C, D}, all 4 nodes
    assert bnmetrics.count_edges(sub) == 3
    assert bnmetrics.count_colliders(sub) == 1


def test_m_graph_blanket_at_C() -> None:
    """In M_4 (A→C, B→C, B→D), MB(C) = {A, B, C}."""
    g = m_4()
    indices = bnmetrics.markov_blanket_indices(g, "C")
    assert indices == (0, 1, 2)


def test_m_graph_blanket_at_D() -> None:
    """In M_4, MB(D) = {B, D}: parent B, no other children of B's children."""
    g = m_4()
    indices = bnmetrics.markov_blanket_indices(g, "D")
    assert indices == (1, 3)
