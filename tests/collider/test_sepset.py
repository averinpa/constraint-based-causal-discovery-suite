"""SepsetOrienter classifies unshielded triples correctly."""

from __future__ import annotations

from cbcd.collider import SepsetOrienter
from cbcd.skeleton import PCStable
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle


def test_y_structure_collider_at_2() -> None:
    dag, _ = ALL_FIXTURES["y_structure"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    # Triple (0, 2, 1) should be classified as a collider (canonical X<Y form).
    assert (0, 2, 1) in decisions.colliders
    assert decisions.non_colliders == frozenset()
    assert decisions.ambiguous == frozenset()


def test_chain_no_colliders() -> None:
    dag, _ = ALL_FIXTURES["chain"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    # 0—1—2 with sepset {1} for {0, 2} → triple (0, 1, 2) is a non-collider.
    assert decisions.colliders == frozenset()
    assert (0, 1, 2) in decisions.non_colliders


def test_fork_no_colliders() -> None:
    dag, _ = ALL_FIXTURES["fork"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    # 1—0—2 with sepset {0} for {1, 2}.
    assert decisions.colliders == frozenset()
    assert (1, 0, 2) in decisions.non_colliders


def test_m_structure_collider_at_2() -> None:
    dag, _ = ALL_FIXTURES["m_structure"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    # 0 and 1 are non-adjacent in skeleton, both adjacent to 2.
    # Sepset({0, 1}) does not contain 2 → collider at 2.
    assert (0, 2, 1) in decisions.colliders


def test_diamond_collider_at_3() -> None:
    dag, _ = ALL_FIXTURES["diamond"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    # 1 and 2 non-adjacent, both adjacent to 3 → collider at 3.
    assert (1, 3, 2) in decisions.colliders
