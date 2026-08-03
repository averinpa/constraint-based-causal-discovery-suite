"""LPCMCI conditioning search spaces (SM §S7): apds / napds sets + sepset-membership queries.

Clean-room from Definitions S4/S5. ``apds`` is the ancestral-phase search space (non-future
adjacencies of the target whose near endpoint is not an arrowhead — the potential ancestors). ``napds``
is the tighter non-ancestral-phase analog of FCI's Possible-D-Sep: ``napds1`` (apds minus the
tail-connected-to-source nodes) plus ``napds2`` (a path-reachable set with collider/ancestor
constraints). All accessors take grid-node ids and respect homology through :class:`LPCMCIPAG`.
"""

from __future__ import annotations

from cbcd.timeseries.lpcmci_pag import HEAD, LPCMCIPAG, TAIL, decode_grid


def apds(g: LPCMCIPAG, target: int, exclude: int) -> list[int]:
    """``apds_t(target, exclude)`` (Def S4): non-future adjacencies of ``target`` (other than
    ``exclude``) whose endpoint mark at the neighbour is not a head — i.e. neighbours not yet
    identified as non-ancestors of ``target``."""
    _, tl = decode_grid(target, g.max_lag)
    out: list[int] = []
    for c in g.neighbors(target):
        if c == exclude:
            continue
        _, cl = decode_grid(c, g.max_lag)
        if cl > tl:  # c is in the future of the target
            continue
        if g.mark(c, target) == HEAD:  # arrowhead at c -> c is a known non-ancestor
            continue
        out.append(c)
    return out


def _napds1(g: LPCMCIPAG, target: int, source: int) -> list[int]:
    """``napds1`` (Def S5): ``apds_t(target, source)`` minus neighbours ``c`` connected to ``source``
    by an edge with a tail at ``source``."""
    base = apds(g, target, source)
    out: list[int] = []
    for c in base:
        if g.edge_exists(c, source) and g.mark(source, c) == TAIL:
            continue
        out.append(c)
    return out


def _is_after(g: LPCMCIPAG, a: int, b: int) -> bool:
    """``a`` is strictly after ``b`` in time order (``b < a``)."""
    return g.before(b, a)


def _napds2(g: LPCMCIPAG, target: int, source: int) -> list[int]:
    """``napds2`` (Def S5): nodes reachable from ``target`` by a path whose interior unshielded triples
    are colliders and with the tail/ancestor constraints of the definition (SM §S7).

    Implemented as a **polynomial state-space reachability** over ``(current, previous)`` states rather
    than exponential simple-path enumeration: a node is collected when reached, and a state is expanded
    only if the just-passed node satisfies the interior constraints (i, ii, v). Allowing walks (a state
    is visited once) yields a superset of the strict simple-path ``napds2``; that is safe — a larger
    S3 search space can only find *valid* m-separators (and hence remove only truly non-adjacent pairs)
    at the oracle. Conditions: (i) no tail at any interior node; (ii) interior unshielded triples are
    colliders; (iii) the path avoids ``source``; (iv) the first node is not head-connected to /after
    ``source``; (v) interior nodes are not tail-connected to ``target``/``source``, not head-connected
    to both, and not after both.
    """
    result: set[int] = set()
    seen: set[tuple[int, int]] = set()
    stack: list[tuple[int, int, bool]] = []  # (current, previous, is_first_hop)
    for nb in g.neighbors(target):
        if nb == source:
            continue
        if g.edge_exists(nb, source) and g.mark(nb, source) == HEAD:  # (iv)
            continue
        if _is_after(g, nb, source):  # (iv)
            continue
        stack.append((nb, target, True))

    while stack:
        cur, prev, is_first = stack.pop()
        if (cur, prev) in seen:
            continue
        seen.add((cur, prev))
        result.add(cur)  # cur is a valid path endpoint X^k
        # Interior constraints (i, v) for `cur` once we extend past it.
        if not is_first:
            if g.edge_exists(cur, target) and g.mark(target, cur) == TAIL:
                continue
            if g.edge_exists(cur, source) and g.mark(source, cur) == TAIL:
                continue
            head_t = g.edge_exists(cur, target) and g.mark(cur, target) == HEAD
            head_s = g.edge_exists(cur, source) and g.mark(cur, source) == HEAD
            if head_t and head_s:
                continue
            if _is_after(g, cur, target) and _is_after(g, cur, source):
                continue
        for nxt in g.neighbors(cur):
            if nxt in (source, prev):
                continue
            # (ii) unshielded triple prev-cur-nxt => cur is a collider on p
            if not g.edge_exists(prev, nxt) and not (
                g.mark(prev, cur) == HEAD and g.mark(nxt, cur) == HEAD
            ):
                continue
            stack.append((nxt, cur, False))
    result.discard(target)
    result.discard(source)
    return sorted(result)


def napds(g: LPCMCIPAG, target: int, source: int) -> list[int]:
    """``napds_t(target, source)`` = ``napds1 ∪ napds2`` (Def S5)."""
    s = set(_napds1(g, target, source)) | set(_napds2(g, target, source))
    s.discard(target)
    s.discard(source)
    return sorted(s)


def order_by_imin(nodes: list[int], target: int, i_min: dict[frozenset[int], float]) -> list[int]:
    """Order a search set by ascending ``I_min(target, ·)`` (strongest association first) for
    order-independence (SM Alg-S2 line 7)."""
    return sorted(nodes, key=lambda c: (i_min.get(frozenset({target, c}), float("inf")), c))
