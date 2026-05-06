"""Sanity checks for the d-separation oracle itself."""

from __future__ import annotations

from cbcd.graph import DAG
from tests.oracle import DSeparationOracle


def test_chain_dseparation() -> None:
    # 0 → 1 → 2: 0 and 2 are d-separated by {1}; dependent unconditionally.
    dag = DAG.from_directed_edges(3, [(0, 1), (1, 2)])
    oracle = DSeparationOracle(dag)
    assert oracle(0, 2, []) == 0.0  # dependent
    assert oracle(0, 2, [1]) == 1.0  # independent


def test_fork_dseparation() -> None:
    # 0 → 1, 0 → 2: 1 and 2 d-separated by {0}; dependent unconditionally.
    dag = DAG.from_directed_edges(3, [(0, 1), (0, 2)])
    oracle = DSeparationOracle(dag)
    assert oracle(1, 2, []) == 0.0
    assert oracle(1, 2, [0]) == 1.0


def test_collider_dseparation() -> None:
    # 0 → 2 ← 1: 0 and 1 d-separated unconditionally; conditioning on the
    # collider 2 OPENS the path (dependent).
    dag = DAG.from_directed_edges(3, [(0, 2), (1, 2)])
    oracle = DSeparationOracle(dag)
    assert oracle(0, 1, []) == 1.0
    assert oracle(0, 1, [2]) == 0.0


def test_descendant_of_collider_opens_path() -> None:
    # 0 → 2 ← 1, 2 → 3: conditioning on 3 (descendant of collider) also opens.
    dag = DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])
    oracle = DSeparationOracle(dag)
    assert oracle(0, 1, []) == 1.0
    assert oracle(0, 1, [3]) == 0.0
    assert oracle(0, 1, [2]) == 0.0
