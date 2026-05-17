"""Smoke tests for `bnmetrics.analyse_mb`."""

from __future__ import annotations

import pytest

import bnmetrics
from tests.fixtures import asia_8, chain_3, collider_3, make_dag

pytest.importorskip("plotly.graph_objects")
pytest.importorskip("plotly.subplots")


def test_returns_plotly_figure() -> None:
    fig = bnmetrics.analyse_mb(asia_8())
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


def test_one_panel_per_metric() -> None:
    """Default ``descriptive='all'`` produces one bar trace per
    descriptive metric."""
    fig = bnmetrics.analyse_mb(asia_8())
    assert len(fig.data) == len(bnmetrics.DESCRIPTIVE_METRIC_NAMES)


def test_specific_metrics_subset() -> None:
    fig = bnmetrics.analyse_mb(asia_8(), descriptive=["n_edges", "n_colliders", "n_root_nodes"])
    assert len(fig.data) == 3


def test_unknown_metric_raises() -> None:
    with pytest.raises(bnmetrics.BNMInputError, match="unknown descriptive metric"):
        bnmetrics.analyse_mb(chain_3(), descriptive=["not-a-metric"])


def test_empty_graph_raises() -> None:
    """A graph with no variables has no MBs to analyse."""
    import numpy as np

    g = bnmetrics.to_graphlike(np.zeros((0, 0), dtype=np.int8))
    with pytest.raises(bnmetrics.BNMInputError, match="no variables"):
        bnmetrics.analyse_mb(g)


def test_chain_n_edges_distribution_hand_computed() -> None:
    """Chain_3 (A→B→C). MBs:
       MB(A) = {A, B}    → 1 edge
       MB(B) = {A, B, C} → 2 edges
       MB(C) = {B, C}    → 1 edge
    So the n_edges value-counts should be {1: 2, 2: 1}.
    """
    fig = bnmetrics.analyse_mb(chain_3(), descriptive=["n_edges"])
    trace = fig.data[0]
    pairs = dict(zip(trace.x, trace.y, strict=True))
    assert pairs == {"1": 2, "2": 1}


def test_collider_n_colliders_distribution() -> None:
    """In collider_3 (A→C, B→C), MB(A)=MB(B)=MB(C)={A,B,C} all contain
    the same one collider. So n_colliders distribution = {1: 3}."""
    fig = bnmetrics.analyse_mb(collider_3(), descriptive=["n_colliders"])
    trace = fig.data[0]
    pairs = dict(zip(trace.x, trace.y, strict=True))
    assert pairs == {"1": 3}


def test_save_html(tmp_path) -> None:
    out = tmp_path / "mb.html"
    bnmetrics.analyse_mb(asia_8(), save=out)
    assert out.exists()
    assert "plotly" in out.read_text().lower()


def test_save_unknown_extension_raises(tmp_path) -> None:
    out = tmp_path / "mb.tiff"
    with pytest.raises(bnmetrics.BNMInputError, match="unsupported save format"):
        bnmetrics.analyse_mb(asia_8(), save=out)


def test_works_on_index_only_graph() -> None:
    """Graph with no var_names: per-MB iteration uses int keys; should
    still produce a figure."""
    g = make_dag(4, [(0, 1), (1, 2), (2, 3)])
    fig = bnmetrics.analyse_mb(g, descriptive=["n_edges"])
    assert len(fig.data) == 1


def test_default_title_present() -> None:
    fig = bnmetrics.analyse_mb(asia_8())
    assert "Markov-blanket" in fig.layout.title.text


def test_custom_title() -> None:
    fig = bnmetrics.analyse_mb(asia_8(), title="Custom title")
    assert fig.layout.title.text == "Custom title"


def test_x_axis_values_are_strings_with_clean_integer_formatting() -> None:
    """Integer counts should be formatted without trailing '.0'."""
    fig = bnmetrics.analyse_mb(chain_3(), descriptive=["n_edges"])
    trace = fig.data[0]
    for x in trace.x:
        # Each x value is an integer count, formatted as a clean string.
        assert "." not in x


def test_n_isolated_nodes_in_chain() -> None:
    """Chain_3: each MB sub-graph has 0 isolated nodes (every node has
    at least one edge incident on it within the MB). Distribution
    should be {0: 3}."""
    fig = bnmetrics.analyse_mb(chain_3(), descriptive=["n_isolated_nodes"])
    trace = fig.data[0]
    pairs = dict(zip(trace.x, trace.y, strict=True))
    assert pairs == {"0": 3}
