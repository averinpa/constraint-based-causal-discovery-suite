"""Region-scoped PCStable: the ``variables`` active-set restriction (roadmap Phase 0b)."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.skeleton import FAS, PCStable
from tests.oracle import DSeparationOracle


def _chain5() -> DAG:
    """X0 -> X1 -> X2 -> X3 -> X4; true skeleton is the 4-edge chain."""
    return DAG.from_directed_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])


def test_region_none_equals_full_range() -> None:
    oracle = DSeparationOracle(_chain5())
    full = PCStable()(oracle, alpha=0.5)
    explicit = PCStable()(oracle, alpha=0.5, variables=list(range(5)))
    assert np.array_equal(full.adj, explicit.adj)


def test_region_recovers_induced_subgraph_with_boundary() -> None:
    # Region {0,1,2} contains the separator {1} for the (0,2) pair, so it is a
    # sound active set: the region skeleton equals the induced chain 0-1-2.
    oracle = DSeparationOracle(_chain5())
    skel = PCStable()(oracle, alpha=0.5, variables=[0, 1, 2])

    expected = np.zeros((5, 5), dtype=bool)
    for u, v in [(0, 1), (1, 2)]:
        expected[u, v] = expected[v, u] = True
    assert np.array_equal(skel.adj, expected)
    # nodes outside the active set are isolated
    assert not skel.adj[3].any()
    assert not skel.adj[4].any()


def test_region_unsound_without_boundary_keeps_spurious_edges() -> None:
    # Region {0,2,4} omits the separators {1},{3}: PC cannot separate the pairs,
    # so spurious edges survive. This is option B's contract -- the caller
    # (region-grow) must include the boundary; the skeleton is only sound given
    # a sufficient active set.
    oracle = DSeparationOracle(_chain5())
    skel = PCStable()(oracle, alpha=0.5, variables=[0, 2, 4])
    # 0-2 and 2-4 remain although the *true* induced subgraph over {0,2,4} is empty.
    assert skel.adj[0, 2] and skel.adj[2, 4]
    assert not skel.adj[1].any() and not skel.adj[3].any()  # unlisted nodes isolated


def test_region_singleton_and_empty_have_no_edges() -> None:
    oracle = DSeparationOracle(_chain5())
    assert not PCStable()(oracle, alpha=0.5, variables=[2]).adj.any()
    assert not PCStable()(oracle, alpha=0.5, variables=[]).adj.any()


def test_region_out_of_range_raises() -> None:
    oracle = DSeparationOracle(_chain5())
    with pytest.raises(CBCDInputError):
        PCStable()(oracle, alpha=0.5, variables=[0, 1, 99])


def test_fas_threads_variables() -> None:
    oracle = DSeparationOracle(_chain5())
    pcs = PCStable()(oracle, alpha=0.5, variables=[0, 1, 2])
    fas = FAS()(oracle, alpha=0.5, variables=[0, 1, 2])
    assert np.array_equal(pcs.adj, fas.adj)
