"""Tests for v0.2.2 viz additions:

  - O1 polish: ``_matching_edges`` distinguishes CIRCLE-bearing edges
    by their full ``(mij, mji)`` mark pair, so two PAG topologies that
    share kind ``"circle"`` but differ in mark pair don't falsely
    register as matching.
  - O2: ``plot_side_by_side`` gains ``mode: Literal["matches", "diff",
    "none"] = "matches"``. ``"diff"`` highlights additions, deletions,
    reversals, and kind changes; ``"none"`` disables edge highlighting.
  - Highlight-color kwargs: ``highlight_node_color`` /
    ``highlight_edge_color`` on ``plot_graph`` and
    ``plot_side_by_side`` override the pastel defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

import bnm
from tests.fixtures import chain_3, collider_3, fork_3, make_dag

graphviz = pytest.importorskip("graphviz")


# ---- highlight color kwargs (#3) -----------------------------------------


def test_plot_graph_highlight_node_color_kwarg_overrides_default() -> None:
    g = collider_3()
    dot = bnm.plot_graph(g, highlight=["C"], highlight_node_color="#ff8800")
    src = dot.source
    assert "#ff8800" in src
    # Pastel default is no longer applied to the highlighted node.
    assert "fillcolor=#c8e6c9" not in src.replace('"', "")


def test_plot_side_by_side_highlight_edge_color_kwarg_overrides_default() -> None:
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(
        g, g, highlight_edge_color="#0066cc"
    )
    # Self-comparison: every edge is a match → all painted with the
    # custom colour, none in the pastel default.
    assert "#0066cc" in dot1.source
    assert "#0066cc" in dot2.source
    assert "#f08080" not in dot1.source
    assert "#f08080" not in dot2.source


def test_plot_side_by_side_highlight_node_color_kwarg_propagates() -> None:
    g1 = chain_3()
    g2 = fork_3()
    dot1, dot2 = bnm.plot_side_by_side(
        g1, g2, highlight_nodes=["A"], highlight_node_color="#9933ff"
    )
    assert "#9933ff" in dot1.source
    assert "#9933ff" in dot2.source


def test_plot_graph_default_colors_unchanged() -> None:
    """No regression: when the kwargs aren't passed, the pastel
    defaults still apply."""
    g = collider_3()
    dot = bnm.plot_graph(g, highlight=["C"])
    assert "#c8e6c9" in dot.source


# ---- O1 polish: CIRCLE-edge matching granularity -------------------------


def _pag_two_node(mij: int, mji: int):
    """A 2-node PAG over endpoint marks ``(mij, mji)`` — i.e.
    ``endpoints[0, 1] = mij``, ``endpoints[1, 0] = mji``."""
    arr = np.zeros((2, 2), dtype=np.int8)
    arr[0, 1] = mij
    arr[1, 0] = mji
    return bnm.to_graphlike(arr, var_names=("A", "B"))


def test_circle_arrow_does_not_match_circle_circle() -> None:
    """Pre-fix bug: ``_classify_edge`` collapsed every CIRCLE-bearing
    edge to kind ``"circle"``, so a ``(CIRCLE, ARROW)`` edge in g1 and a
    ``(CIRCLE, CIRCLE)`` edge in g2 falsely registered as matching and
    got highlighted in both panels.
    """
    arrow = int(bnm.EndpointMark.ARROW)
    circle = int(bnm.EndpointMark.CIRCLE)
    # g1 has A o→ B (CIRCLE at A, ARROW at B).
    g1 = _pag_two_node(mij=arrow, mji=circle)
    # g2 has A o-o B (CIRCLE at both ends).
    g2 = _pag_two_node(mij=circle, mji=circle)

    dot1, dot2 = bnm.plot_side_by_side(g1, g2, mode="matches")
    # No matches → no edge gets the highlight stroke in either panel.
    assert "#f08080" not in dot1.source
    assert "#f08080" not in dot2.source


def test_circle_arrow_matches_circle_arrow() -> None:
    """The complementary positive case: identical ``(mij, mji)`` PAG
    edges DO match and DO highlight."""
    arrow = int(bnm.EndpointMark.ARROW)
    circle = int(bnm.EndpointMark.CIRCLE)
    g1 = _pag_two_node(mij=arrow, mji=circle)
    g2 = _pag_two_node(mij=arrow, mji=circle)

    dot1, dot2 = bnm.plot_side_by_side(g1, g2, mode="matches")
    assert "#f08080" in dot1.source
    assert "#f08080" in dot2.source


# ---- O2: mode="diff" ------------------------------------------------------


def test_mode_diff_highlights_reversed_edge() -> None:
    """g1 has A→B; g2 has B→A. mode='diff' should highlight the edge
    in both panels (it's a reversal)."""
    g1 = make_dag(2, [(0, 1)], var_names=("A", "B"))
    g2 = make_dag(2, [(1, 0)], var_names=("A", "B"))
    dot1, dot2 = bnm.plot_side_by_side(g1, g2, mode="diff")
    assert "#f08080" in dot1.source
    assert "#f08080" in dot2.source


def test_mode_diff_highlights_addition_in_g2_only() -> None:
    """g1 has no edge between B-C; g2 has B→C. mode='diff' highlights
    in g2 only — there's no g1 edge to paint."""
    g1 = make_dag(3, [(0, 1)], var_names=("A", "B", "C"))
    g2 = make_dag(3, [(0, 1), (1, 2)], var_names=("A", "B", "C"))
    dot1, dot2 = bnm.plot_side_by_side(g1, g2, mode="diff")
    # The shared A→B edge isn't a diff → not highlighted.
    # The B→C edge is in g2 only → only g2 panel has the highlight.
    assert "#f08080" not in dot1.source
    assert "#f08080" in dot2.source


def test_mode_diff_highlights_deletion_in_g1_only() -> None:
    """g1 has B→C; g2 doesn't. mode='diff' highlights in g1 only."""
    g1 = make_dag(3, [(0, 1), (1, 2)], var_names=("A", "B", "C"))
    g2 = make_dag(3, [(0, 1)], var_names=("A", "B", "C"))
    dot1, dot2 = bnm.plot_side_by_side(g1, g2, mode="diff")
    assert "#f08080" in dot1.source
    assert "#f08080" not in dot2.source


def test_mode_diff_no_highlight_when_g1_equals_g2() -> None:
    """Self-comparison under mode='diff' → no diffs → nothing
    highlighted."""
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(g, g, mode="diff")
    assert "#f08080" not in dot1.source
    assert "#f08080" not in dot2.source


def test_mode_matches_default_unchanged() -> None:
    """Default mode='matches' preserves v0.2.0 behaviour."""
    g = chain_3()
    dot1, dot2 = bnm.plot_side_by_side(g, g)
    assert "#f08080" in dot1.source
    assert "#f08080" in dot2.source


def test_mode_invalid_raises() -> None:
    g = chain_3()
    with pytest.raises(bnm.BNMInputError, match="unknown mode"):
        bnm.plot_side_by_side(g, g, mode="bogus")  # type: ignore[arg-type]
