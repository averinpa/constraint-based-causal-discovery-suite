"""Meek R1–R4 unit tests with hand-crafted PartialCPDAGs."""

from __future__ import annotations

import numpy as np

from cbcd.graph.cpdag import PartialCPDAG
from cbcd.graph.marks import EndpointMark
from cbcd.rules import MeekRules


def _ep(n: int, dirs: list[tuple[int, int]], unds: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int8)
    for u, v in dirs:
        m[u, v] = EndpointMark.ARROW
        m[v, u] = EndpointMark.TAIL
    for u, v in unds:
        m[u, v] = EndpointMark.TAIL
        m[v, u] = EndpointMark.TAIL
    return m


def _is_directed(cpdag, u: int, v: int) -> bool:  # type: ignore[no-untyped-def]
    return (
        cpdag.endpoints[u, v] == EndpointMark.ARROW
        and cpdag.endpoints[v, u] == EndpointMark.TAIL
    )


def _is_undirected(cpdag, u: int, v: int) -> bool:  # type: ignore[no-untyped-def]
    return (
        cpdag.endpoints[u, v] == EndpointMark.TAIL
        and cpdag.endpoints[v, u] == EndpointMark.TAIL
    )


def test_r1_fires() -> None:
    # 0 → 1 — 2, 0 not adjacent 2 ⟹ 1 → 2.
    g = PartialCPDAG(3, _ep(3, [(0, 1)], [(1, 2)]))
    out = MeekRules(rules=frozenset({"R1"}))(g)
    assert _is_directed(out, 1, 2)


def test_r1_does_not_fire_when_a_c_adjacent() -> None:
    # 0 → 1 — 2 with 0 — 2: R1 must not orient 1 → 2.
    g = PartialCPDAG(3, _ep(3, [(0, 1)], [(1, 2), (0, 2)]))
    out = MeekRules(rules=frozenset({"R1"}))(g)
    assert _is_undirected(out, 1, 2)


def test_r2_fires() -> None:
    # 0 → 1 → 2, 0 — 2 ⟹ 0 → 2.
    g = PartialCPDAG(3, _ep(3, [(0, 1), (1, 2)], [(0, 2)]))
    out = MeekRules(rules=frozenset({"R2"}))(g)
    assert _is_directed(out, 0, 2)


def test_r2_does_not_fire_without_chain() -> None:
    # 0 → 1, 0 — 2 with no 1 → 2: R2 inactive.
    g = PartialCPDAG(3, _ep(3, [(0, 1)], [(0, 2), (1, 2)]))
    out = MeekRules(rules=frozenset({"R2"}))(g)
    assert _is_undirected(out, 0, 2)


def test_r3_fires() -> None:
    # a — b, a — c, a — d, c → b, d → b, c not adj d ⟹ a → b.
    # n=4, a=0, b=1, c=2, d=3.
    g = PartialCPDAG(4, _ep(4, [(2, 1), (3, 1)], [(0, 1), (0, 2), (0, 3)]))
    out = MeekRules(rules=frozenset({"R3"}))(g)
    assert _is_directed(out, 0, 1)


def test_r3_does_not_fire_when_c_d_adjacent() -> None:
    g = PartialCPDAG(4, _ep(4, [(2, 1), (3, 1)], [(0, 1), (0, 2), (0, 3), (2, 3)]))
    out = MeekRules(rules=frozenset({"R3"}))(g)
    assert _is_undirected(out, 0, 1)


def test_meek_fixpoint_terminates() -> None:
    # 0 → 1 — 2 — 3 — 4: should propagate via R1 to 1→2→3→4.
    g = PartialCPDAG(5, _ep(5, [(0, 1)], [(1, 2), (2, 3), (3, 4)]))
    out = MeekRules()(g)
    assert _is_directed(out, 1, 2)
    assert _is_directed(out, 2, 3)
    assert _is_directed(out, 3, 4)
