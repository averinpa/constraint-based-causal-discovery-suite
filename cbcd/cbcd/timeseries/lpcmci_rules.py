"""LPCMCI orientation rules + Alg-S4 driver (SM §S4, §S5).

Clean-room from the authors' published rule logic. Each rule is a pure function that PROPOSES endpoint
modifications on an :class:`LPCMCIPAG`; the Alg-S4 driver applies them, resolving a
propose-tail-and-head conflict at one endpoint to a ``CONFLICT`` mark, and restarts from the first
rule on any change. Middle-mark rules APR and MMR realise Lemma 1 / Lemma S8.

**Mark convention (used everywhere):** ``g.mark(U, V)`` is the endpoint mark *at U* on edge ``{U, V}``.
So ``U --> V`` means ``mark(U,V)=TAIL`` and ``mark(V,U)=HEAD``; ``U *-> V`` means ``mark(V,U)=HEAD``;
``U o-* V`` means ``mark(U,V)=CIRCLE``. A **proposal** ``(U, V, m)`` sets the mark *at U* to ``m``.

Sepset-membership ("is B in / not in S_AC") is answered from the ``SepSet`` memory populated by the
removal phases (a valid m-separating set per pair); at the CI oracle this is exact.
"""

from __future__ import annotations

from cbcd.timeseries.lpcmci_pag import (
    CIRCLE,
    CONFLICT,
    HEAD,
    LPCMCIPAG,
    MM_BANG,
    MM_EMPTY,
    MM_L,
    MM_Q,
    MM_R,
    TAIL,
)

SepSet = dict[frozenset[int], set[int]]
Proposal = tuple[int, int, int]  # (U, V, mark) -> set the mark AT U on edge {U, V}


def _unshielded_triples(g: LPCMCIPAG) -> list[tuple[int, int, int]]:
    """Unshielded triples ``(a, b, c)`` — ``a, c`` adjacent to ``b``, ``a`` not adjacent ``c``
    (canonicalised with ``a < c`` by grid id)."""
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for b in range(g.grid_n):
        nbrs = g.neighbors(b)
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                a, c = nbrs[x], nbrs[y]
                if g.edge_exists(a, c):
                    continue
                key = (min(a, c), b, max(a, c))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def _in_sepset(sepset: SepSet, a: int, c: int, b: int) -> bool:
    return b in sepset.get(frozenset({a, c}), set())


def _b_not_in_sepset(sepset: SepSet, a: int, c: int, b: int) -> bool:
    key = frozenset({a, c})
    return key in sepset and b not in sepset[key]


# --- collider rule (R0′ family) -----------------------------------------------------------------
def rule_collider(g: LPCMCIPAG, sepset: SepSet) -> list[Proposal]:
    """Unshielded ``a *-* b *-* c`` with ``b ∉ S_ac`` and no tail/conflict already at ``b`` -> collider
    ``a *-> b <-* c`` (head at ``b`` on both arms)."""
    props: list[Proposal] = []
    for a, b, c in _unshielded_triples(g):
        if not _b_not_in_sepset(sepset, a, c, b):
            continue
        if g.mark(b, a) in (TAIL, CONFLICT) or g.mark(b, c) in (TAIL, CONFLICT):
            continue
        props.append((b, a, HEAD))
        props.append((b, c, HEAD))
    return props


# --- R1′  (FCI R1): x *-> b o-* z, x not adj z, b ∈ S_xz  =>  b --> z (tail at b) ----------------
def rule_r1(g: LPCMCIPAG, sepset: SepSet) -> list[Proposal]:
    props: list[Proposal] = []
    for a, b, c in _unshielded_triples(g):
        for x, z in ((a, c), (c, a)):
            if g.mark(b, x) == HEAD and g.mark(b, z) == CIRCLE and _in_sepset(sepset, x, z, b):
                props.append((b, z, TAIL))  # tail at b => b --> z
    return props


# --- R2′ (FCI R2): a -> b *-> c or a *-> b -> c, with a o-* c  =>  head at c ----------------------
def rule_r2(g: LPCMCIPAG, sepset: SepSet) -> list[Proposal]:
    del sepset
    props: list[Proposal] = []
    for b in range(g.grid_n):
        nbrs = g.neighbors(b)
        for a in nbrs:
            for c in nbrs:
                if a == c or not g.edge_exists(a, c):
                    continue
                if g.mark(c, a) != CIRCLE:  # need circle at c on a-c
                    continue
                pat_a = (
                    g.mark(a, b) == TAIL and g.mark(b, a) == HEAD and g.mark(c, b) == HEAD
                )  # a --> b *-> c
                pat_b = (
                    g.mark(b, a) == HEAD and g.mark(b, c) == TAIL and g.mark(c, b) == HEAD
                )  # a *-> b --> c
                if pat_a or pat_b:
                    props.append((c, a, HEAD))
    return props


# --- R3′ (FCI R3): a *-> b <-* c, a o-o d o-o c, d o-o b, d ∈ S_ac  =>  head at b -----------------
def rule_r3(g: LPCMCIPAG, sepset: SepSet) -> list[Proposal]:
    props: list[Proposal] = []
    for a, b, c in _unshielded_triples(g):
        if not (g.mark(b, a) == HEAD and g.mark(b, c) == HEAD):
            continue
        for d in g.neighbors(b):
            if d in (a, c) or not (g.edge_exists(a, d) and g.edge_exists(c, d)):
                continue
            if (
                g.mark(d, a) == CIRCLE
                and g.mark(d, c) == CIRCLE
                and g.mark(b, d) == CIRCLE
                and _in_sepset(sepset, a, c, d)
            ):
                props.append((b, d, HEAD))  # head at b
    return props


# --- R8′ (FCI R8): a --> b --> c with a o-* c  =>  a --> c (tail at a) ----------------------------
def rule_r8(g: LPCMCIPAG, sepset: SepSet) -> list[Proposal]:
    del sepset
    props: list[Proposal] = []
    for b in range(g.grid_n):
        for a in g.neighbors(b):
            for c in g.neighbors(b):
                if a == c or not g.edge_exists(a, c):
                    continue
                if (
                    g.mark(a, b) == TAIL
                    and g.mark(b, a) == HEAD
                    and g.mark(b, c) == TAIL
                    and g.mark(c, b) == HEAD
                    and g.mark(a, c) == CIRCLE
                ):
                    props.append((a, c, TAIL))  # tail at a => a --> c
    return props


# --- APR (Lemma 1): definite directed edges get empty middle marks ------------------------------
def rule_apr(g: LPCMCIPAG) -> bool:
    changed = False
    for a in range(g.grid_n):
        for b in g.neighbors(a):
            if a >= b:
                continue
            if g.mark(a, b) == TAIL and g.mark(b, a) == HEAD:
                tail_node, head_node = a, b  # a --> b
            elif g.mark(b, a) == TAIL and g.mark(a, b) == HEAD:
                tail_node, head_node = b, a  # b --> a
            else:
                continue
            mm = g.middle(a, b)
            if mm == MM_EMPTY:
                continue
            demote = (
                mm == MM_BANG
                or (mm == MM_L and g.before(head_node, tail_node))
                or (mm == MM_R and g.before(tail_node, head_node))
            )
            if demote:
                g.set_middle(a, b, MM_EMPTY)
                changed = True
    return changed


# --- MMR (Lemma S8): order-directed middle-mark promotion ---------------------------------------
def rule_mmr(g: LPCMCIPAG) -> bool:
    changed = False
    for a in range(g.grid_n):
        for b in g.neighbors(a):
            if not g.before(a, b):  # apply once per edge, from the earlier endpoint (A < B)
                continue
            mm = g.middle(a, b)
            new = MM_L if mm in (MM_Q, MM_R) else mm  # A<B: ? -> L, R -> L
            if new != mm:
                g.set_middle(a, b, new)
                changed = True
    return changed


# --- R9′ (uncovered potentially-directed paths): a o-> c + uncovered PD path => tail at a --------
def _potentially_directed(g: LPCMCIPAG, u: int, v: int) -> bool:
    """Edge ``u — v`` can be oriented ``u --> v``: mark at u is tail/circle, mark at v is head/circle."""
    return g.mark(u, v) in (TAIL, CIRCLE) and g.mark(v, u) in (HEAD, CIRCLE)


def _uncovered_pd_path(g: LPCMCIPAG, start: int, end: int) -> list[int] | None:
    stack: list[list[int]] = [[start]]
    while stack:
        path = stack.pop()
        cur = path[-1]
        if cur == end and len(path) >= 3:
            return path
        for nb in g.neighbors(cur):
            if nb in path or not _potentially_directed(g, cur, nb):
                continue
            if len(path) >= 2 and g.edge_exists(nb, path[-2]):  # uncovered
                continue
            stack.append([*path, nb])
    return None


def rule_r9(g: LPCMCIPAG, sepset: SepSet) -> list[Proposal]:
    del sepset
    props: list[Proposal] = []
    for a in range(g.grid_n):
        for c in g.neighbors(a):
            # a o-> c : circle at a, head at c
            if not (g.mark(a, c) == CIRCLE and g.mark(c, a) == HEAD):
                continue
            path = _uncovered_pd_path(g, a, c)
            if path is not None and len(path) >= 3 and not g.edge_exists(path[1], c):
                props.append((a, c, TAIL))  # tail at a
    return props


# =================================================================================================
# Alg-S4 driver
# =================================================================================================
_PROPOSAL_RULES = {
    "collider": rule_collider,
    "R1": rule_r1,
    "R2": rule_r2,
    "R3": rule_r3,
    "R8": rule_r8,
    "R9": rule_r9,
}
_MIDDLE_RULES = {"APR": rule_apr, "MMR": rule_mmr}


def _apply_proposals(g: LPCMCIPAG, props: list[Proposal], only_lagged: bool = False) -> bool:
    """Apply endpoint proposals ``(U, V, mark)`` (set mark at U). A single endpoint proposed both
    tail and head, or a proposal contradicting an already-committed tail/head, becomes ``CONFLICT``.
    ``only_lagged`` restricts to lagged edges (SM Alg-S2 line 18)."""
    wanted: dict[tuple[int, int], set[int]] = {}
    for u, v, mark in props:
        if only_lagged:
            tau, _, _, _ = g._canonical(u, v)
            if tau == 0:
                continue
        wanted.setdefault((u, v), set()).add(mark)
    changed = False
    for (u, v), marks in wanted.items():
        if not g.edge_exists(u, v):
            continue
        cur = g.mark(u, v)
        if len(marks) > 1:  # both tail and head proposed at U
            if cur != CONFLICT:
                g.set_mark(u, v, CONFLICT)
                changed = True
            continue
        m = next(iter(marks))
        if cur in (m, CONFLICT):
            continue
        if cur in (TAIL, HEAD) and cur != m:  # contradicts a committed orientation
            g.set_mark(u, v, CONFLICT)
            changed = True
        elif cur == CIRCLE:
            g.set_mark(u, v, m)
            changed = True
    return changed


def orient(g: LPCMCIPAG, rule_names: list[str], sepset: SepSet, only_lagged: bool = False) -> None:
    """Alg-S4 driver: apply rules in order; restart from the first rule on any change; stop when a
    full pass changes nothing."""
    i = 0
    guard = 0
    max_guard = 300 * (g.grid_n + 1)
    while i < len(rule_names):
        guard += 1
        if guard > max_guard:
            break
        name = rule_names[i]
        if name in _MIDDLE_RULES:
            changed = _MIDDLE_RULES[name](g)
        else:
            props = _PROPOSAL_RULES[name](g, sepset)
            changed = _apply_proposals(g, props, only_lagged=only_lagged)
        i = 0 if changed else i + 1


# Rule lists per the SM (Alg-S2 line 18 = lagged-only prelim; line 22 / S3 = full).
RULES_PRELIM_LAGGED: list[str] = ["APR", "MMR", "R8", "R2", "R1", "R9"]
RULES_FULL: list[str] = ["APR", "MMR", "R8", "R2", "R1", "collider", "R3", "R9"]
