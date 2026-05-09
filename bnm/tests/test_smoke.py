"""Smoke tests."""

from __future__ import annotations

import bnm


def test_version_is_set() -> None:
    assert bnm.__version__.startswith("0.")


def test_public_api_importable() -> None:
    expected = {
        # graph contract
        "EndpointMark",
        "GraphLike",
        "to_graphlike",
        # exceptions
        "BNMError",
        "BNMInputError",
        "BNMDataError",
        # descriptive
        "count_edges",
        "count_nodes",
        "count_directed_arcs",
        "count_undirected_arcs",
        "count_bidirected_arcs",
        "count_circle_edges",
        "count_colliders",
        "count_root_nodes",
        "count_leaf_nodes",
        "count_isolated_nodes",
        "count_reversible_arcs",
        "in_degree",
        "out_degree",
        # comparative
        "shd",
        "hd",
        "f1",
        "precision",
        "recall",
        "true_positives",
        "false_positives",
        "false_negatives",
        "count_additions",
        "count_deletions",
        "count_reversals",
        # markov blanket
        "markov_blanket",
        "markov_blanket_indices",
        # multi-metric façade
        "compare",
        "Comparison",
        "to_dataframe",
        "DESCRIPTIVE_METRIC_NAMES",
        "COMPARATIVE_METRIC_NAMES",
        # sid
        "SIDResult",
        "sid",
        # viz
        "plot_graph",
        "plot_side_by_side",
        "plot_sid_matrix",
    }
    actual = set(bnm.__all__)
    missing = expected - actual
    assert not missing, f"public API missing: {missing}"
