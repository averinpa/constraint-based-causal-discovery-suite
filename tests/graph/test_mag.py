"""MAG construction (minimal stub) and deferred-method behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.graph import MAG, EndpointMark


def _ep(n: int, edges: list[tuple[int, int, EndpointMark, EndpointMark]]) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        m[i, j] = mj
        m[j, i] = mi
    return m


ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL
CIRC = EndpointMark.CIRCLE


def test_mag_accepts_directed() -> None:
    g = MAG(3, _ep(3, [(0, 1, TAIL, ARR), (1, 2, TAIL, ARR)]))
    assert g.endpoints[0, 1] == ARR
    assert g.endpoints[1, 0] == TAIL


def test_mag_accepts_bidirected() -> None:
    g = MAG(2, _ep(2, [(0, 1, ARR, ARR)]))
    assert g.endpoints[0, 1] == ARR
    assert g.endpoints[1, 0] == ARR


def test_mag_rejects_circle() -> None:
    bad = _ep(2, [(0, 1, CIRC, ARR)])
    with pytest.raises(CBCDInputError):
        MAG(2, bad)


def test_mag_rejects_undirected_tail_tail() -> None:
    bad = _ep(2, [(0, 1, TAIL, TAIL)])
    with pytest.raises(CBCDInputError):
        MAG(2, bad)


def test_mag_methods_are_deferred() -> None:
    g = MAG(2, _ep(2, [(0, 1, TAIL, ARR)]))
    with pytest.raises(NotImplementedError):
        g.is_ancestor_of(0, 1)
    with pytest.raises(NotImplementedError):
        g.m_separated(0, 1, ())
    with pytest.raises(NotImplementedError):
        g.to_pag()
