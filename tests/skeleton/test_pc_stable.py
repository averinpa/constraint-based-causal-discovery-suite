"""PCStable recovers true skeleton with the d-separation oracle."""

from __future__ import annotations

import numpy as np

from cbcd.graph.marks import EndpointMark
from cbcd.skeleton import PCStable
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle


def _true_skeleton(dag) -> np.ndarray:  # type: ignore[no-untyped-def]
    adj = dag.endpoints != EndpointMark.NO_EDGE
    return adj | adj.T


def test_pc_stable_recovers_y_structure() -> None:
    dag, _ = ALL_FIXTURES["y_structure"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel.adj, _true_skeleton(dag))


def test_pc_stable_recovers_fork() -> None:
    dag, _ = ALL_FIXTURES["fork"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel.adj, _true_skeleton(dag))


def test_pc_stable_recovers_chain() -> None:
    dag, _ = ALL_FIXTURES["chain"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel.adj, _true_skeleton(dag))


def test_pc_stable_recovers_m_structure() -> None:
    dag, _ = ALL_FIXTURES["m_structure"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel.adj, _true_skeleton(dag))


def test_pc_stable_recovers_diamond() -> None:
    dag, _ = ALL_FIXTURES["diamond"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel.adj, _true_skeleton(dag))


def test_pc_stable_recovers_asia() -> None:
    dag, _ = ALL_FIXTURES["asia"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert np.array_equal(skel.adj, _true_skeleton(dag))


def test_pc_stable_records_sepsets() -> None:
    # Chain 0 → 1 → 2: PC must remove edge 0-2 with sepset {1}.
    from cbcd.graph import DAG

    dag = DAG.from_directed_edges(3, [(0, 1), (1, 2)])
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    assert frozenset({0, 2}) in skel.sepsets
    assert skel.sepsets[frozenset({0, 2})] == (1,)
