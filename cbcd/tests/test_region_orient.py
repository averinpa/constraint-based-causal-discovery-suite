"""Region-scoped orientation with boundary semantics (roadmap Phase 0c)."""

from __future__ import annotations

import numpy as np

from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark
from cbcd.region import orient_region
from cbcd.skeleton import PCStable, Skeleton
from tests.oracle import DSeparationOracle


def _collider_dag() -> DAG:
    """X0 -> X2 <- X1  and  X2 -> X3. One v-structure at 2; skeleton 0-2, 1-2, 2-3."""
    return DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])


def _directed(ep: np.ndarray, i: int, j: int) -> bool:
    return bool(ep[i, j] == EndpointMark.ARROW and ep[j, i] == EndpointMark.TAIL)


def _undirected(ep: np.ndarray, i: int, j: int) -> bool:
    return bool(ep[i, j] == EndpointMark.TAIL and ep[j, i] == EndpointMark.TAIL)


def _skeleton() -> tuple[Skeleton, DSeparationOracle]:
    oracle = DSeparationOracle(_collider_dag())
    return PCStable()(oracle, alpha=0.5), oracle


def test_interior_all_recovers_full_orientation() -> None:
    # interior = every node → no gating, no freeze → the global PC result.
    skel, oracle = _skeleton()
    cpdag = orient_region(skel, oracle, interior=[0, 1, 2, 3], alpha=0.5)
    ep = cpdag.endpoints
    assert _directed(ep, 0, 2) and _directed(ep, 1, 2)  # v-structure 0->2<-1
    assert _directed(ep, 2, 3)  # Meek R1: 2->3


def test_boundary_center_suppresses_v_structure() -> None:
    # Centre 2 is boundary → its v-structure is not trusted; everything stays undirected.
    skel, oracle = _skeleton()
    cpdag = orient_region(skel, oracle, interior=[0, 1, 3], alpha=0.5)
    ep = cpdag.endpoints
    assert _undirected(ep, 0, 2) and _undirected(ep, 1, 2) and _undirected(ep, 2, 3)


def test_interior_collider_kept_but_boundary_edge_frozen() -> None:
    # Centre 2 interior → v-structure oriented; but 3 is boundary, so Meek must NOT
    # propagate 2->3 (the boundary-incident edge is frozen undirected).
    skel, oracle = _skeleton()
    cpdag = orient_region(skel, oracle, interior=[2], alpha=0.5)
    ep = cpdag.endpoints
    assert _directed(ep, 0, 2) and _directed(ep, 1, 2)  # collider still sound
    assert _undirected(ep, 2, 3)  # frozen: 3 outside the interior


def test_interior_all_equals_ungated_orientation() -> None:
    # Parity: interior = full set behaves exactly like the ungated collider + Meek path.
    from cbcd.collider import SepsetOrienter
    from cbcd.rules import MeekRules

    skel, oracle = _skeleton()
    gated = orient_region(skel, oracle, interior=[0, 1, 2, 3], alpha=0.5).endpoints
    decisions = SepsetOrienter()(skel, oracle, alpha=0.5)
    ungated = MeekRules()(decisions.apply_to_cpdag(skel)).endpoints
    assert np.array_equal(gated, ungated)
