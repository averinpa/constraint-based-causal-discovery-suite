"""Hand-computed descriptive metric checks on canonical fixtures."""

from __future__ import annotations

import bnm
from tests.fixtures import (
    asia_8,
    chain_3,
    collider_3,
    diamond_4,
    empty_3,
    fork_3,
    m_4,
    y_4,
)


def test_chain_descriptive() -> None:
    g = chain_3()
    assert bnm.count_edges(g) == 2
    assert bnm.count_nodes(g) == 3
    assert bnm.count_directed_arcs(g) == 2
    assert bnm.count_undirected_arcs(g) == 0
    assert bnm.count_bidirected_arcs(g) == 0
    assert bnm.count_circle_edges(g) == 0
    assert bnm.count_colliders(g) == 0
    assert bnm.count_root_nodes(g) == 1  # A
    assert bnm.count_leaf_nodes(g) == 1  # C
    assert bnm.count_isolated_nodes(g) == 0
    assert bnm.count_reversible_arcs(g) == 2  # both arcs reversible (no colliders)


def test_collider_descriptive() -> None:
    g = collider_3()
    assert bnm.count_edges(g) == 2
    assert bnm.count_directed_arcs(g) == 2
    assert bnm.count_colliders(g) == 1
    assert bnm.count_root_nodes(g) == 2  # A, B
    assert bnm.count_leaf_nodes(g) == 1  # C
    assert bnm.count_reversible_arcs(g) == 0  # both arcs into the collider


def test_fork_descriptive() -> None:
    g = fork_3()
    assert bnm.count_edges(g) == 2
    assert bnm.count_colliders(g) == 0
    assert bnm.count_root_nodes(g) == 1  # A
    assert bnm.count_leaf_nodes(g) == 2  # B, C
    assert bnm.count_reversible_arcs(g) == 2


def test_y_descriptive() -> None:
    g = y_4()
    assert bnm.count_edges(g) == 3
    assert bnm.count_colliders(g) == 1  # at C: A, B not adjacent
    assert bnm.count_root_nodes(g) == 2  # A, B
    assert bnm.count_leaf_nodes(g) == 1  # D
    # 2 arcs into collider C → not reversible. 1 arc out (C→D) → reversible.
    assert bnm.count_reversible_arcs(g) == 1


def test_m_descriptive() -> None:
    g = m_4()
    assert bnm.count_edges(g) == 3
    assert bnm.count_colliders(g) == 1  # at C: A, B not adjacent
    # roots: A, B; leaves: C, D. C is leaf? It has no out-edges. Yes.
    assert bnm.count_root_nodes(g) == 2
    assert bnm.count_leaf_nodes(g) == 2


def test_diamond_descriptive() -> None:
    g = diamond_4()
    assert bnm.count_edges(g) == 4
    # D has parents B, C — they ARE adjacent? In diamond_4, A→B, A→C, B→D, C→D.
    # B and C are not adjacent, so D is an unshielded collider.
    assert bnm.count_colliders(g) == 1
    assert bnm.count_root_nodes(g) == 1  # A
    assert bnm.count_leaf_nodes(g) == 1  # D


def test_empty_descriptive() -> None:
    g = empty_3()
    assert bnm.count_edges(g) == 0
    assert bnm.count_nodes(g) == 3
    assert bnm.count_isolated_nodes(g) == 3
    assert bnm.count_root_nodes(g) == 3
    assert bnm.count_leaf_nodes(g) == 3


def test_asia_descriptive() -> None:
    g = asia_8()
    assert bnm.count_edges(g) == 8
    assert bnm.count_nodes(g) == 8
    # 'either' has parents tub, lung — not adjacent → collider.
    # 'dysp' has parents either, bronc — not adjacent → collider.
    assert bnm.count_colliders(g) == 2
    # roots: asia, smoke. leaves: xray, dysp.
    assert bnm.count_root_nodes(g) == 2
    assert bnm.count_leaf_nodes(g) == 2
    # 8 arcs total, 4 enter collider apexes (either, dysp). So 4 reversible.
    assert bnm.count_reversible_arcs(g) == 4


def test_in_out_degree_by_index() -> None:
    g = asia_8()
    # 'either' (idx 5) has in 2 (tub→either, lung→either) and out 2.
    assert bnm.in_degree(g, 5) == 2
    assert bnm.out_degree(g, 5) == 2
    # 'asia' (idx 0): in 0, out 1.
    assert bnm.in_degree(g, 0) == 0
    assert bnm.out_degree(g, 0) == 1


def test_in_out_degree_by_name() -> None:
    g = asia_8()
    assert bnm.in_degree(g, "either") == 2
    assert bnm.out_degree(g, "either") == 2
    assert bnm.in_degree(g, "asia") == 0
    assert bnm.out_degree(g, "asia") == 1
