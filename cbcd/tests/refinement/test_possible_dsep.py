"""PossibleDSepRefinement removes spurious edges using Possible-D-Sep witnesses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from cbcd.citest.protocol import CITestResult
from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PartialPAG
from cbcd.refinement import PossibleDSepRefinement

ARR = EndpointMark.ARROW
CIRC = EndpointMark.CIRCLE
TAIL = EndpointMark.TAIL


class StubCI:
    """CITest that answers from a lookup table; default ``p_value=0.0`` (dependent)."""

    def __init__(
        self,
        n_vars: int,
        sep_results: dict[tuple[int, int, frozenset[int]], float],
    ) -> None:
        self.n_vars = n_vars
        self._sep = sep_results

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        key = (min(x, y), max(x, y), frozenset(S))
        return self._sep.get(key, 0.0)

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        return CITestResult(p_value=self(x, y, S))


def _ep(n: int, edges: list[tuple[int, int, EndpointMark, EndpointMark]]) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        m[i, j] = mj
        m[j, i] = mi
    return m


def test_refinement_removes_edge_when_possible_dsep_separates() -> None:
    # 4-node PartialPAG: 0 ↔ 1 ↔ 2 ↔ 3 with extra edge 0 o─o 3 (the spurious one).
    # Possible-D-Sep(0, 3) = {1, 2} (colliders on the path 0-1-2-3 because 1 and 2
    # have arrows from both sides). Stub: only S = {1, 2} separates 0 and 3.
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, ARR),
            (2, 3, ARR, ARR),
            (0, 3, CIRC, CIRC),
        ],
    )
    g = PartialPAG(4, ep)
    ci = StubCI(
        4,
        {
            (0, 3, frozenset({1, 2})): 1.0,
        },
    )
    out = PossibleDSepRefinement()(g, ci, alpha=0.05)
    # Edge (0, 3) removed.
    assert out.endpoints[0, 3] == EndpointMark.NO_EDGE
    assert out.endpoints[3, 0] == EndpointMark.NO_EDGE
    # Witness recorded.
    assert out.sepsets is not None
    assert out.sepsets[frozenset({0, 3})] == (1, 2)
    # Non-targeted edges still present (orientations wiped to CIRCLE-CIRCLE).
    assert out.endpoints[0, 1] == CIRC
    assert out.endpoints[1, 0] == CIRC


def test_refinement_no_op_when_no_separation_found() -> None:
    # No subset separates 0 and 3 (stub returns p ≤ alpha for all queries).
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, ARR),
            (2, 3, ARR, ARR),
            (0, 3, CIRC, CIRC),
        ],
    )
    g = PartialPAG(4, ep)
    ci = StubCI(4, {})  # everything dependent
    out = PossibleDSepRefinement()(g, ci, alpha=0.05)
    # All edges present (orientations wiped to CIRCLE).
    assert out.endpoints[0, 3] == CIRC
    assert out.endpoints[3, 0] == CIRC


def test_refinement_honours_max_cond_set_cap() -> None:
    # Same fixture but cap max_cond_set=1 — only size-0 and size-1 subsets are
    # tested. The separating set is {1, 2} (size 2). Should NOT remove.
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, ARR),
            (2, 3, ARR, ARR),
            (0, 3, CIRC, CIRC),
        ],
    )
    g = PartialPAG(4, ep)
    ci = StubCI(4, {(0, 3, frozenset({1, 2})): 1.0})
    out = PossibleDSepRefinement(max_cond_set=1)(g, ci, alpha=0.05)
    assert out.endpoints[0, 3] == CIRC
    assert out.endpoints[3, 0] == CIRC


def test_refinement_rejects_n_jobs_gt_1() -> None:
    g = PartialPAG(2, _ep(2, [(0, 1, CIRC, CIRC)]))
    ci = StubCI(2, {})
    with pytest.raises(CBCDInputError):
        PossibleDSepRefinement()(g, ci, alpha=0.05, n_jobs=2)
