"""Parity check: every descriptive metric on every legacy-snapshot fixture
matches bnmetrics 0.1.x output exactly."""

from __future__ import annotations

import pytest

import bnmetrics
from tests.fixtures import from_legacy_fixture, load_legacy_snapshot

_METRIC_FUNCS = {
    "n_edges": bnmetrics.count_edges,
    "n_nodes": bnmetrics.count_nodes,
    "n_directed_arcs": bnmetrics.count_directed_arcs,
    "n_undirected_arcs": bnmetrics.count_undirected_arcs,
    "n_colliders": bnmetrics.count_colliders,
    "n_root_nodes": bnmetrics.count_root_nodes,
    "n_leaf_nodes": bnmetrics.count_leaf_nodes,
    "n_isolated_nodes": bnmetrics.count_isolated_nodes,
    "n_reversible_arcs": bnmetrics.count_reversible_arcs,
}


def _all_fixture_ids() -> list[str]:
    snapshot = load_legacy_snapshot()
    return list(snapshot["fixtures"].keys())


@pytest.mark.parametrize("fixture_id", _all_fixture_ids())
def test_descriptive_parity(fixture_id: str) -> None:
    snapshot = load_legacy_snapshot()
    entry = snapshot["fixtures"][fixture_id]
    g = from_legacy_fixture(entry)

    expected = entry["descriptive"]
    for name, func in _METRIC_FUNCS.items():
        actual = func(g)
        assert actual == expected[name], (
            f"{fixture_id}.{name}: v0.2 returned {actual}, legacy snapshot has {expected[name]}"
        )


@pytest.mark.parametrize("fixture_id", _all_fixture_ids())
def test_per_node_degree_parity(fixture_id: str) -> None:
    snapshot = load_legacy_snapshot()
    entry = snapshot["fixtures"][fixture_id]
    g = from_legacy_fixture(entry)

    for var_name, expected in entry["per_node_degree"].items():
        assert bnmetrics.in_degree(g, var_name) == expected["in"], f"{fixture_id}.in_degree[{var_name}]"
        assert bnmetrics.out_degree(g, var_name) == expected["out"], (
            f"{fixture_id}.out_degree[{var_name}]"
        )
