"""ColliderDecisions.apply_to_pag lays correct CIRCLE / ARROW marks."""

from __future__ import annotations

from cbcd.collider import SepsetOrienter
from cbcd.graph.marks import EndpointMark
from cbcd.skeleton import PCStable
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle

ARR = EndpointMark.ARROW
CIRC = EndpointMark.CIRCLE


def test_y_structure_pag_arrows_at_collider() -> None:
    dag, _ = ALL_FIXTURES["y_structure"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    pag = decisions.apply_to_pag(skel)

    # Skeleton: 0—2, 1—2 (no 0-1). Collider at 2.
    # Expected: 0 o→ 2 ←o 1, no edge 0-1.
    assert pag.endpoints[0, 2] == ARR  # arrow at 2 from 0's edge
    assert pag.endpoints[2, 0] == CIRC  # circle at 0
    assert pag.endpoints[1, 2] == ARR
    assert pag.endpoints[2, 1] == CIRC
    assert pag.endpoints[0, 1] == EndpointMark.NO_EDGE
    assert pag.endpoints[1, 0] == EndpointMark.NO_EDGE


def test_chain_pag_all_circles() -> None:
    dag, _ = ALL_FIXTURES["chain"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    pag = decisions.apply_to_pag(skel)

    # Skeleton: 0—1, 1—2 (no 0-2). No colliders. All present edges CIRCLE—CIRCLE.
    assert pag.endpoints[0, 1] == CIRC
    assert pag.endpoints[1, 0] == CIRC
    assert pag.endpoints[1, 2] == CIRC
    assert pag.endpoints[2, 1] == CIRC


def test_m_structure_pag_collider_at_2() -> None:
    dag, _ = ALL_FIXTURES["m_structure"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    pag = decisions.apply_to_pag(skel)

    # 0 o→ 2 ←o 1; 1 o-o 3; 1 o-o 4 (no v-structures involving 3, 4)
    assert pag.endpoints[0, 2] == ARR
    assert pag.endpoints[2, 0] == CIRC
    assert pag.endpoints[1, 2] == ARR
    assert pag.endpoints[2, 1] == CIRC
    # 1—3 is CIRCLE—CIRCLE (not a collider edge)
    assert pag.endpoints[1, 3] == CIRC
    assert pag.endpoints[3, 1] == CIRC


def test_apply_to_pag_carries_sepsets() -> None:
    dag, _ = ALL_FIXTURES["chain"]()
    oracle = DSeparationOracle(dag)
    skel = PCStable()(oracle, alpha=0.5)
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    pag = decisions.apply_to_pag(skel)
    assert pag.sepsets == skel.sepsets
