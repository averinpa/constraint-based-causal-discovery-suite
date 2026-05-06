"""PAG / PartialPAG construction, mark validation, and accessors."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.graph import PAG, EndpointMark, PartialPAG


def _ep(n: int, edges: list[tuple[int, int, EndpointMark, EndpointMark]]) -> np.ndarray:
    """Build endpoints from ``(i, j, mark_at_i, mark_at_j)`` triples."""
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        m[i, j] = mj
        m[j, i] = mi
    return m


CIRC = EndpointMark.CIRCLE
ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL


def test_pag_accepts_circle_marks() -> None:
    # Edge 1 o-> 2: CIRCLE at 1 (= endpoints[2, 1]), ARROW at 2 (= endpoints[1, 2]).
    g = PAG(3, _ep(3, [(0, 1, CIRC, CIRC), (1, 2, CIRC, ARR)]))
    assert g.endpoints[1, 0] == CIRC
    assert g.endpoints[1, 2] == ARR
    assert g.endpoints[2, 1] == CIRC


def test_pag_accepts_bidirected() -> None:
    g = PAG(3, _ep(3, [(0, 1, ARR, ARR)]))
    assert g.endpoints[0, 1] == ARR
    assert g.endpoints[1, 0] == ARR


def test_pag_rejects_unknown_mark() -> None:
    bad = np.zeros((3, 3), dtype=np.int8)
    bad[0, 1] = 99
    bad[1, 0] = 99
    with pytest.raises(CBCDInputError):
        PAG(3, bad)


def test_pag_rejects_asymmetric_no_edge() -> None:
    bad = np.zeros((3, 3), dtype=np.int8)
    bad[0, 1] = ARR  # one side has an edge, other side doesn't
    with pytest.raises(CBCDInputError):
        PAG(3, bad)


def test_partial_pag_carries_sepsets() -> None:
    sepsets = {frozenset({0, 2}): (1,)}
    g = PartialPAG(3, _ep(3, [(0, 1, CIRC, CIRC), (1, 2, CIRC, CIRC)]), sepsets=sepsets)
    assert g.sepsets == sepsets


def test_partial_pag_default_sepsets_is_none() -> None:
    g = PartialPAG(3, _ep(3, [(0, 1, CIRC, CIRC)]))
    assert g.sepsets is None


def test_pag_definite_edges_excludes_circle() -> None:
    # 0 → 1 (definite), 1 o-> 2 (has CIRCLE → not definite), 0 ↔ 2 (definite)
    g = PAG(3, _ep(3, [(0, 1, TAIL, ARR), (1, 2, CIRC, ARR), (0, 2, ARR, ARR)]))
    edges = g.definite_edges()
    keys = {(i, j) for i, j, _, _ in edges}
    assert keys == {(0, 1), (0, 2)}


def test_pag_possibly_directed() -> None:
    # 0 → 1 (mark at 1 = ARROW, mark at 0 = TAIL): possibly_directed(0, 1) = True
    # 0 → 1: possibly_directed(1, 0) = False (mark at 0 = TAIL, can't have arrow there)
    g = PAG(2, _ep(2, [(0, 1, TAIL, ARR)]))
    assert g.possibly_directed(0, 1) is True
    assert g.possibly_directed(1, 0) is False


def test_pag_possibly_directed_circle_endpoints() -> None:
    # 0 o-o 1: both directions are possible
    g = PAG(2, _ep(2, [(0, 1, CIRC, CIRC)]))
    assert g.possibly_directed(0, 1) is True
    assert g.possibly_directed(1, 0) is True


def test_pag_possibly_directed_bidirected() -> None:
    # 0 ↔ 1: TAIL not consistent with either side, so neither direction possibly directed
    g = PAG(2, _ep(2, [(0, 1, ARR, ARR)]))
    assert g.possibly_directed(0, 1) is False
    assert g.possibly_directed(1, 0) is False


def test_pag_possibly_directed_no_edge() -> None:
    g = PAG(2)
    assert g.possibly_directed(0, 1) is False


def test_pag_equality_distinguishes_marks() -> None:
    g1 = PAG(2, _ep(2, [(0, 1, CIRC, ARR)]))
    g2 = PAG(2, _ep(2, [(0, 1, CIRC, ARR)]))
    g3 = PAG(2, _ep(2, [(0, 1, TAIL, ARR)]))
    assert g1 == g2
    assert g1 != g3
