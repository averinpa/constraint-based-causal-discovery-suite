"""Structural queries on a PAG / PartialPAG.

These are the graph-traversal helpers used by ``FCIRules`` (uncovered circle
paths for R5–R7, discriminating paths for R4 and R8–R10) and by
``PossibleDSepRefinement`` (Possible-D-Sep computation). All queries operate
directly on an endpoint matrix so they can run against either a closed
``PAG`` or a working ``PartialPAG``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from cbcd.graph.marks import EndpointMark


def _has_edge(endpoints: NDArray[np.int8], i: int, j: int) -> bool:
    return bool(endpoints[i, j] != EndpointMark.NO_EDGE)


def _adjacent_indices(endpoints: NDArray[np.int8], v: int) -> list[int]:
    return [int(j) for j in np.where(endpoints[v] != EndpointMark.NO_EDGE)[0]]


def _is_collider_on_path(endpoints: NDArray[np.int8], a: int, b: int, c: int) -> bool:
    """B is a collider on path A-B-C iff arrow at B from both A and C."""
    return bool(endpoints[a, b] == EndpointMark.ARROW and endpoints[c, b] == EndpointMark.ARROW)


def _is_uncovered_path(endpoints: NDArray[np.int8], path: list[int]) -> bool:
    """A path is *uncovered* iff every consecutive triple is unshielded.

    For each interior vertex ``v_i`` (1 ≤ i ≤ k-1), its two neighbours on the
    path ``v_{i-1}`` and ``v_{i+1}`` are non-adjacent. Endpoints of the path
    may themselves be adjacent (e.g. the o─o edge being oriented in R5).
    """
    k = len(path)
    return all(not _has_edge(endpoints, path[i - 1], path[i + 1]) for i in range(1, k - 1))


def possible_dsep(endpoints: NDArray[np.int8], x: int, y: int) -> frozenset[int]:
    """Possible-D-Sep(X, Y) excluding X and Y.

    A vertex V is in Possible-D-Sep(X, Y) iff V ≠ X, V ≠ Y, and there is a path
    π = ⟨X = v_0, v_1, ..., v_k = V⟩ such that for every consecutive triple
    ⟨v_{i-1}, v_i, v_{i+1}⟩ on π either:

    * v_i is a collider on π (arrows at v_i from both sides), or
    * ⟨v_{i-1}, v_i, v_{i+1}⟩ is a triangle (all three vertices mutually
      adjacent).

    Implemented via BFS over (current_vertex, predecessor) states; revisit a
    vertex through a different predecessor to allow fresh subpaths.
    """
    if x == y:
        return frozenset()
    visited: set[tuple[int, int]] = set()
    out: set[int] = set()
    queue: list[tuple[int, int]] = []
    for nbr in _adjacent_indices(endpoints, x):
        if nbr == x:
            continue
        state = (nbr, x)
        if state in visited:
            continue
        visited.add(state)
        queue.append(state)
        if nbr != y:
            out.add(nbr)

    while queue:
        v, prev = queue.pop()
        for w in _adjacent_indices(endpoints, v):
            if w in (prev, x):
                continue
            collider = _is_collider_on_path(endpoints, prev, v, w)
            triangle = _has_edge(endpoints, prev, w)
            if not (collider or triangle):
                continue
            state = (w, v)
            if state in visited:
                continue
            visited.add(state)
            queue.append(state)
            if w != y:
                out.add(w)
    return frozenset(out)


def _walk_paths(
    endpoints: NDArray[np.int8],
    start: int,
    end: int,
    edge_ok: Callable[[NDArray[np.int8], int, int], bool],
    *,
    require_uncovered: bool,
    max_length: int | None,
) -> Iterator[tuple[int, ...]]:
    """DFS over simple (no-repeat) paths from ``start`` to ``end``."""
    n = endpoints.shape[0]
    if max_length is None:
        max_length = n
    stack: list[tuple[int, list[int]]] = [(start, [start])]
    while stack:
        v, path = stack.pop()
        if v == end and len(path) >= 2:
            if require_uncovered and not _is_uncovered_path(endpoints, path):
                continue
            yield tuple(path)
            continue
        if len(path) - 1 >= max_length:
            continue
        for w in _adjacent_indices(endpoints, v):
            if w in path:
                continue
            if not edge_ok(endpoints, v, w):
                continue
            stack.append((w, path + [w]))


def _circle_circle_step(endpoints: NDArray[np.int8], u: int, v: int) -> bool:
    """An edge ``u — v`` qualifies as a circle-circle step iff both ends are CIRCLE."""
    return bool(endpoints[u, v] == EndpointMark.CIRCLE and endpoints[v, u] == EndpointMark.CIRCLE)


def _pd_step(endpoints: NDArray[np.int8], u: int, v: int) -> bool:
    """Step ``u → v`` is potentially directed iff mark at ``v`` ∈ {ARROW, CIRCLE}
    and mark at ``u`` ∈ {TAIL, CIRCLE}."""
    mark_at_v = int(endpoints[u, v])
    mark_at_u = int(endpoints[v, u])
    if mark_at_v == int(EndpointMark.NO_EDGE):
        return False
    if mark_at_v not in (int(EndpointMark.ARROW), int(EndpointMark.CIRCLE)):
        return False
    return mark_at_u in (int(EndpointMark.TAIL), int(EndpointMark.CIRCLE))


def find_uncovered_circle_path(
    endpoints: NDArray[np.int8], x: int, y: int, *, max_length: int | None = None
) -> tuple[int, ...] | None:
    """Return one uncovered circle path from ``x`` to ``y`` with ≥ 1 intermediate
    vertex, or None if none exists."""
    for path in _walk_paths(
        endpoints,
        x,
        y,
        edge_ok=_circle_circle_step,
        require_uncovered=True,
        max_length=max_length,
    ):
        if len(path) >= 3:
            return path
    return None


def find_uncovered_pd_path(
    endpoints: NDArray[np.int8], x: int, y: int, *, max_length: int | None = None
) -> tuple[int, ...] | None:
    """Return one uncovered potentially-directed path from ``x`` to ``y`` with
    ≥ 1 intermediate vertex, or None."""
    for path in _walk_paths(
        endpoints,
        x,
        y,
        edge_ok=_pd_step,
        require_uncovered=True,
        max_length=max_length,
    ):
        if len(path) >= 3:
            return path
    return None


def find_discriminating_path(
    endpoints: NDArray[np.int8], a: int, b: int, c: int
) -> tuple[int, ...] | None:
    """Return a discriminating path ``⟨θ, ..., a, b, c⟩`` for ``b`` between θ
    and ``c``, or None if none exists.

    Conditions (Zhang 2008):

    * The path ends ``... a — b — c``; ``a`` adjacent to ``b``, ``b`` to ``c``.
    * The edge ``a — b`` has an arrow at ``a`` (so ``a ←* b``).
    * ``a`` is a parent of ``c`` (``a → c`` with TAIL at ``a``).
    * Every vertex strictly between θ and ``b`` (so ``q_1, ..., q_p, a``) is a
      collider on the path **and** a parent of ``c``.
    * θ is not adjacent to ``c``.
    """
    if not (_has_edge(endpoints, a, b) and _has_edge(endpoints, b, c)):
        return None
    if endpoints[b, a] != EndpointMark.ARROW:
        return None
    if not (endpoints[a, c] == EndpointMark.ARROW and endpoints[c, a] == EndpointMark.TAIL):
        return None

    n = endpoints.shape[0]
    # ``path_back`` grows backwards from ``a`` toward θ. The full path is
    # reversed(path_back) + [b, c]. Each iteration either closes the path
    # (w becomes θ, non-adjacent to c) or extends it (w becomes a new
    # intermediate, requiring parent-of-c and collider conditions).
    stack: list[list[int]] = [[a]]
    while stack:
        path_back = stack.pop()
        front = path_back[-1]
        neighbour_toward_b = path_back[-2] if len(path_back) >= 2 else b
        for w in _adjacent_indices(endpoints, front):
            if w in path_back or w in (b, c):
                continue
            # ``front`` must be a collider on triple ⟨w, front, neighbour_toward_b⟩.
            if endpoints[w, front] != EndpointMark.ARROW:
                continue
            if endpoints[neighbour_toward_b, front] != EndpointMark.ARROW:
                continue
            # If w is non-adjacent to c → w is θ; return the path.
            if not _has_edge(endpoints, w, c):
                full = list(reversed(path_back + [w])) + [b, c]
                return tuple(full)
            # Otherwise w becomes a new intermediate — require parent-of-c.
            if not (endpoints[w, c] == EndpointMark.ARROW and endpoints[c, w] == EndpointMark.TAIL):
                continue
            if len(path_back) >= n:
                continue
            stack.append(path_back + [w])
    return None
