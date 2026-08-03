"""Stage 3 — LPCMCI orientation rules on hand-built LPCMCI-PAG fixtures (SM §S4)."""

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
    grid_id,
)
from cbcd.timeseries.lpcmci_rules import (
    _apply_proposals,
    orient,
    rule_apr,
    rule_collider,
    rule_mmr,
    rule_r1,
    rule_r2,
    rule_r8,
)


def _g(n: int = 3, ml: int = 0) -> LPCMCIPAG:
    return LPCMCIPAG(n, ml)


def test_collider_orients_heads_at_apex_when_not_in_sepset() -> None:
    # a o-o b o-o c, a not adj c, b NOT in sepset(a,c) -> a *-> b <-* c
    g = _g(3)
    a, b, c = grid_id(0, 0, 0), grid_id(1, 0, 0), grid_id(2, 0, 0)
    g.add_edge(a, b, CIRCLE, CIRCLE, MM_EMPTY)
    g.add_edge(b, c, CIRCLE, CIRCLE, MM_EMPTY)
    sepset = {frozenset({a, c}): set()}  # b not in it
    props = rule_collider(g, sepset)
    assert (b, a, HEAD) in props and (b, c, HEAD) in props
    _apply_proposals(g, props)
    assert g.mark(b, a) == HEAD and g.mark(b, c) == HEAD  # heads at the apex


def test_collider_skips_when_b_in_sepset() -> None:
    g = _g(3)
    a, b, c = grid_id(0, 0, 0), grid_id(1, 0, 0), grid_id(2, 0, 0)
    g.add_edge(a, b, CIRCLE, CIRCLE, MM_EMPTY)
    g.add_edge(b, c, CIRCLE, CIRCLE, MM_EMPTY)
    sepset = {frozenset({a, c}): {b}}  # b IS in sepset -> non-collider
    assert rule_collider(g, sepset) == []


def test_r1_orients_tail_at_apex() -> None:
    # a *-> b o-o c, a not adj c, b in sepset(a,c)  =>  b --> c (tail at b)
    g = _g(3)
    a, b, c = grid_id(0, 0, 0), grid_id(1, 0, 0), grid_id(2, 0, 0)
    g.add_edge(a, b, CIRCLE, HEAD, MM_EMPTY)  # a *-> b (head at b)
    g.add_edge(b, c, CIRCLE, CIRCLE, MM_EMPTY)  # b o-o c
    sepset = {frozenset({a, c}): {b}}
    props = rule_r1(g, sepset)
    assert (b, c, TAIL) in props
    _apply_proposals(g, props)
    assert g.mark(b, c) == TAIL  # b --> c


def test_r2_orients_head_at_c() -> None:
    # a --> b *-> c with a o-* c  =>  head at c
    g = _g(3)
    a, b, c = grid_id(0, 0, 0), grid_id(1, 0, 0), grid_id(2, 0, 0)
    g.add_edge(a, b, TAIL, HEAD, MM_EMPTY)  # a --> b
    g.add_edge(b, c, TAIL, HEAD, MM_EMPTY)  # b --> c (=> b *-> c)
    g.add_edge(a, c, CIRCLE, CIRCLE, MM_EMPTY)  # a o-o c
    props = rule_r2(g, {})
    assert (c, a, HEAD) in props


def test_r8_orients_tail_at_a() -> None:
    # a --> b --> c with a o-* c  =>  a --> c (tail at a)
    g = _g(3)
    a, b, c = grid_id(0, 0, 0), grid_id(1, 0, 0), grid_id(2, 0, 0)
    g.add_edge(a, b, TAIL, HEAD, MM_EMPTY)
    g.add_edge(b, c, TAIL, HEAD, MM_EMPTY)
    g.add_edge(a, c, CIRCLE, HEAD, MM_EMPTY)  # a o-> c
    props = rule_r8(g, {})
    assert (a, c, TAIL) in props


def test_conflict_when_both_tail_and_head_proposed() -> None:
    g = _g(2)
    a, b = grid_id(0, 0, 0), grid_id(1, 0, 0)
    g.add_edge(a, b, CIRCLE, CIRCLE, MM_EMPTY)
    _apply_proposals(g, [(a, b, TAIL), (a, b, HEAD)])  # contradictory at a
    assert g.mark(a, b) == CONFLICT


def test_conflict_when_proposal_contradicts_committed() -> None:
    g = _g(2)
    a, b = grid_id(0, 0, 0), grid_id(1, 0, 0)
    g.add_edge(a, b, TAIL, HEAD, MM_EMPTY)  # tail committed at a
    _apply_proposals(g, [(a, b, HEAD)])  # propose head at a -> conflict
    assert g.mark(a, b) == CONFLICT


def test_apr_demotes_bang_middle_of_directed_edge() -> None:
    g = _g(2)
    a, b = grid_id(0, 0, 0), grid_id(1, 0, 0)
    g.add_edge(a, b, TAIL, HEAD, MM_BANG)  # a -!-> b
    assert rule_apr(g)
    assert g.middle(a, b) == MM_EMPTY  # -!-> becomes -->


def test_mmr_promotes_question_to_l_from_earlier_endpoint() -> None:
    g = LPCMCIPAG(2, 1)
    p, x = grid_id(0, -1, 1), grid_id(0, 0, 1)  # p = X0_{t-1} < x = X0_t
    g.add_edge(p, x, TAIL, HEAD, MM_Q)
    assert rule_mmr(g)
    assert g.middle(p, x) == MM_L  # A<B: ? -> L


def test_orient_driver_reaches_fixpoint_and_restarts() -> None:
    # collider a*->b<-*c makes a head at apex b; the driver then restarts and R1 fires on b*->e.
    g = _g(4)
    a, b, c, e = (grid_id(v, 0, 0) for v in range(4))
    g.add_edge(a, b, CIRCLE, CIRCLE, MM_EMPTY)
    g.add_edge(b, c, CIRCLE, CIRCLE, MM_EMPTY)  # a-b-c unshielded (a not adj c) -> collider at b
    g.add_edge(b, e, CIRCLE, CIRCLE, MM_EMPTY)  # b-e, a not adj e -> R1 uses a*->b o-o e
    sepset = {frozenset({a, c}): set(), frozenset({a, e}): {b}}  # b∉sep(a,c); b∈sep(a,e)
    orient(g, ["collider", "R1"], sepset)
    assert g.mark(b, a) == HEAD and g.mark(b, c) == HEAD  # collider heads at b
    assert g.mark(b, e) == TAIL  # R1 restart: a *-> b o-o e, b∈sep(a,e) => b --> e


def test_middle_marks_absent_from_returned_pag() -> None:
    from cbcd.timeseries.lpcmci_pag import complete_lpcmci_pag

    g = complete_lpcmci_pag(2, 1)
    ts = g.to_timeseries_pag()  # conversion strips middle marks
    # the returned type carries only ordinary PAG endpoint marks
    seen = {int(m) for m in ts.window.endpoints.flatten()}
    assert seen <= {0, TAIL, HEAD, CIRCLE}
    _ = (MM_L, MM_R)  # symbols exist for the internal layer
