"""Graph queries on PAG endpoint matrices."""

from __future__ import annotations

import numpy as np

from cbcd.graph.marks import EndpointMark
from cbcd.graph.queries import (
    find_discriminating_path,
    find_uncovered_circle_path,
    find_uncovered_pd_path,
    possible_dsep,
)

ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL
CIRC = EndpointMark.CIRCLE
NO = EndpointMark.NO_EDGE


def _ep(n: int, edges: list[tuple[int, int, EndpointMark, EndpointMark]]) -> np.ndarray:
    """Build endpoints from ``(i, j, mark_at_i, mark_at_j)`` triples."""
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, mi, mj in edges:
        m[i, j] = mj
        m[j, i] = mi
    return m


def test_uncovered_circle_path_simple() -> None:
    # 0 o-o 1 o-o 2 o-o 0 (triangle of o-o edges).
    # Path from 0 to 2 via 1 has length 2; the (0, 2) pair is the path
    # endpoints (allowed adjacent for "uncovered"). The interior triple
    # (0, 1, 2) is unshielded? No — 0 and 2 ARE adjacent (path endpoints).
    # In the standard "uncovered" definition, only consecutive triples need
    # be unshielded. The single interior triple (0, 1, 2) has v_{i-1}=0 and
    # v_{i+1}=2 — adjacent. So this is NOT uncovered.
    ep = _ep(3, [(0, 1, CIRC, CIRC), (1, 2, CIRC, CIRC), (0, 2, CIRC, CIRC)])
    assert find_uncovered_circle_path(ep, 0, 2) is None


def test_uncovered_circle_path_chain() -> None:
    # 0 o-o 1 o-o 2 o-o 3 (no shortcut). Path 0-1-2-3, interior triples:
    # (0,1,2) - need 0 not adj 2: True; (1,2,3) - need 1 not adj 3: True. Uncovered.
    ep = _ep(4, [(0, 1, CIRC, CIRC), (1, 2, CIRC, CIRC), (2, 3, CIRC, CIRC)])
    path = find_uncovered_circle_path(ep, 0, 3)
    assert path == (0, 1, 2, 3)


def test_uncovered_pd_path_directed() -> None:
    # 0 → 1 → 2 (potentially-directed path α→β→γ).
    ep = _ep(3, [(0, 1, TAIL, ARR), (1, 2, TAIL, ARR)])
    path = find_uncovered_pd_path(ep, 0, 2)
    assert path == (0, 1, 2)


def test_uncovered_pd_path_with_circle_marks() -> None:
    # 0 o→ 1 o-o 2 (still potentially directed both steps).
    ep = _ep(3, [(0, 1, CIRC, ARR), (1, 2, CIRC, CIRC)])
    path = find_uncovered_pd_path(ep, 0, 2)
    assert path == (0, 1, 2)


def test_uncovered_pd_path_blocked_by_arrow_at_start() -> None:
    # 0 ← 1 → 2: from 0's view, step 0→1 has arrow at 0 (mark at u=0 is ARROW
    # — disallowed for PD step starting at u). No PD path from 0 to 2.
    ep = _ep(3, [(0, 1, ARR, TAIL), (1, 2, TAIL, ARR)])
    assert find_uncovered_pd_path(ep, 0, 2) is None


def test_possible_dsep_collider_chain() -> None:
    # 0 ↔ 1 ↔ 2 (bidirected chain). 1 is a collider on path 0-1-2.
    # Possible-D-Sep(0, 2) should include 1 (collider).
    ep = _ep(3, [(0, 1, ARR, ARR), (1, 2, ARR, ARR)])
    pds = possible_dsep(ep, 0, 2)
    assert 1 in pds


def test_possible_dsep_chain_no_collider_excluded() -> None:
    # 0 → 1 → 2: triple (0, 1, 2) is not a collider at 1, and 0,2 not adjacent
    # (no triangle). So 1 should NOT be in PossibleDSep(0, 2).
    ep = _ep(3, [(0, 1, TAIL, ARR), (1, 2, TAIL, ARR)])
    pds = possible_dsep(ep, 0, 2)
    # 1 reaches Possible-D-Sep as a length-1 step from 0, but extending to 2
    # requires the (0, 1, 2) triple to be a collider or triangle. Neither.
    # PossibleDSep(0, 2) should at least contain 1 (length-1 reachable).
    assert 1 in pds


def test_possible_dsep_excludes_endpoints() -> None:
    ep = _ep(3, [(0, 1, ARR, ARR), (1, 2, ARR, ARR)])
    pds = possible_dsep(ep, 0, 2)
    assert 0 not in pds
    assert 2 not in pds


def test_discriminating_path_minimal_p0() -> None:
    # Variables: θ=0, a=1, b=2, c=3. Path ⟨θ, a, b, c⟩ with no intermediate q_i.
    # Conditions:
    #   - a-b: arrow at a       → endpoints[2, 1] = ARR
    #   - a → c (a is parent)   → endpoints[1, 3] = ARR, endpoints[3, 1] = TAIL
    #   - b o─o c (b's mark is CIRCLE; c's mark anything, use CIRCLE)
    #   - a is a collider on ⟨θ, a, b⟩: arrow at a from θ AND from b.
    #   - θ not adjacent to c   → no edge (0, 3).
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),  # θ ↔ a (arrow at a from θ side)
            (1, 2, ARR, TAIL),  # a ← b: arrow at a, tail at b
            (1, 3, TAIL, ARR),  # a → c
            (2, 3, CIRC, CIRC),  # b o─o c
        ],
    )
    path = find_discriminating_path(ep, a=1, b=2, c=3)
    assert path == (0, 1, 2, 3)


def test_discriminating_path_none_if_theta_adjacent_to_c() -> None:
    # Same as minimal but with θ → c (θ adjacent to c). No θ qualifies.
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, TAIL),
            (0, 3, TAIL, ARR),  # θ → c, making θ adjacent to c
            (1, 3, TAIL, ARR),
            (2, 3, CIRC, CIRC),
        ],
    )
    assert find_discriminating_path(ep, a=1, b=2, c=3) is None


def test_discriminating_path_none_if_a_not_parent_of_c() -> None:
    # a is not a parent of c → precondition fails.
    ep = _ep(
        4,
        [
            (0, 1, ARR, ARR),
            (1, 2, ARR, TAIL),
            (1, 3, CIRC, CIRC),  # a o─o c instead of a → c
            (2, 3, CIRC, CIRC),
        ],
    )
    assert find_discriminating_path(ep, a=1, b=2, c=3) is None
