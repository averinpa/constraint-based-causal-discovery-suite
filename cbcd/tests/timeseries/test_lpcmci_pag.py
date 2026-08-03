"""Stage 1 — LPCMCI-PAG graph type: homology, marks, middle-mark algebra, conversion."""

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
    NO_EDGE,
    TAIL,
    _homologous_pairs,
    complete_lpcmci_pag,
    decode_grid,
    grid_id,
    mm_combine,
)


def test_grid_encode_decode_roundtrip() -> None:
    max_lag = 3
    for v in range(4):
        for lag in range(-max_lag, 1):
            node = grid_id(v, lag, max_lag)
            assert decode_grid(node, max_lag) == (v, lag)


def test_lemma_s8_middle_mark_algebra() -> None:
    # ? + * = *
    for m in (MM_EMPTY, MM_Q, MM_L, MM_R, MM_BANG):
        assert mm_combine(MM_Q, m) == m
        assert mm_combine(m, MM_Q) == m
    # * + empty = empty
    for m in (MM_EMPTY, MM_Q, MM_L, MM_R, MM_BANG):
        assert mm_combine(m, MM_EMPTY) == MM_EMPTY
        assert mm_combine(MM_EMPTY, m) == MM_EMPTY
    # L + R = !
    assert mm_combine(MM_L, MM_R) == MM_BANG
    assert mm_combine(MM_R, MM_L) == MM_BANG
    # idempotent
    assert mm_combine(MM_L, MM_L) == MM_L
    assert mm_combine(MM_BANG, MM_L) == MM_BANG


def test_homology_write_propagates_to_all_shifts() -> None:
    # max_lag=2: edge X^0_{t-1} — X^1_t has homologous copy X^0_{t-2} — X^1_{t-1}.
    g = LPCMCIPAG(n_series=2, max_lag=2)
    a = grid_id(0, -1, 2)  # X^0_{t-1}
    b = grid_id(1, 0, 2)  # X^1_t
    g.add_edge(a, b, TAIL, HEAD, MM_L)  # X^0_{t-1} --> X^1_t
    # Every time-shifted copy in-window sees the same edge + marks.
    for (pa, pb) in _homologous_pairs(a, b, 2, 2):
        assert g.edge_exists(pa, pb)
        assert g.mark(pa, pb) == TAIL  # tail at the X^0 (earlier) end
        assert g.mark(pb, pa) == HEAD  # head at the X^1 (present) end
        assert g.middle(pa, pb) == MM_L
    # The shifted copy is a genuinely different grid pair.
    a2, b2 = grid_id(0, -2, 2), grid_id(1, -1, 2)
    assert (a2, b2) != (a, b)
    assert g.edge_exists(a2, b2)


def test_set_mark_is_homology_consistent_and_directional() -> None:
    g = LPCMCIPAG(n_series=2, max_lag=1)
    a = grid_id(0, -1, 1)
    b = grid_id(1, 0, 1)
    g.add_edge(a, b, CIRCLE, CIRCLE, MM_Q)
    g.set_mark(b, a, HEAD)  # head at the present end
    assert g.mark(b, a) == HEAD
    assert g.mark(a, b) == CIRCLE  # other end unchanged
    # reading the mark at each end via the reversed argument order agrees
    assert g.mark(a, b) == CIRCLE and g.mark(b, a) == HEAD


def test_contemporaneous_canonicalization_symmetric() -> None:
    g = LPCMCIPAG(n_series=3, max_lag=1)
    x = grid_id(0, 0, 1)
    y = grid_id(2, 0, 1)
    g.add_edge(x, y, CIRCLE, HEAD, MM_Q)  # X^0_t o-?-> X^2_t
    assert g.mark(x, y) == CIRCLE
    assert g.mark(y, x) == HEAD
    # order of arguments does not matter
    assert g.mark(y, x) == HEAD and g.mark(x, y) == CIRCLE
    assert g.edge_exists(y, x)


def test_neighbors_and_parents_respect_homology() -> None:
    g = LPCMCIPAG(n_series=2, max_lag=1)
    # X^0_{t-1} --> X^0_t (autodependency) and X^0_t o-o X^1_t
    g.add_edge(grid_id(0, -1, 1), grid_id(0, 0, 1), TAIL, HEAD, MM_EMPTY)
    g.add_edge(grid_id(0, 0, 1), grid_id(1, 0, 1), CIRCLE, CIRCLE, MM_Q)
    x0t = grid_id(0, 0, 1)
    assert set(g.neighbors(x0t)) == {grid_id(0, -1, 1), grid_id(1, 0, 1)}
    assert g.parents(x0t) == [grid_id(0, -1, 1)]  # the autodependency, tail->head


def test_before_total_order() -> None:
    g = LPCMCIPAG(n_series=2, max_lag=1)
    # earlier lag before later; ties by index
    assert g.before(grid_id(0, -1, 1), grid_id(0, 0, 1))
    assert g.before(grid_id(1, -1, 1), grid_id(0, 0, 1))  # any past before any present
    assert g.before(grid_id(0, 0, 1), grid_id(1, 0, 1))  # τ=0 tie: index
    assert not g.before(grid_id(1, 0, 1), grid_id(0, 0, 1))


def test_complete_initialisation() -> None:
    g = complete_lpcmci_pag(n_series=2, max_lag=1)
    # contemporaneous X^0_t o-?-o X^1_t
    x0, x1 = grid_id(0, 0, 1), grid_id(1, 0, 1)
    assert g.mark(x0, x1) == CIRCLE and g.mark(x1, x0) == CIRCLE
    assert g.middle(x0, x1) == MM_Q
    # lagged X^0_{t-1} -L-> X^0_t
    p = grid_id(0, -1, 1)
    assert g.mark(p, x0) == TAIL and g.mark(x0, p) == HEAD
    assert g.middle(p, x0) == MM_L


def test_to_timeseries_pag_strips_middle_and_maps_conflict() -> None:
    g = LPCMCIPAG(n_series=2, max_lag=1)
    g.add_edge(grid_id(0, 0, 1), grid_id(1, 0, 1), HEAD, HEAD, MM_EMPTY)  # bidirected <->
    g.add_edge(grid_id(0, -1, 1), grid_id(0, 0, 1), CONFLICT, HEAD, MM_Q)
    ts = g.to_timeseries_pag()
    ep = ts.window.endpoints
    x0, x1, p = grid_id(0, 0, 1), grid_id(1, 0, 1), grid_id(0, -1, 1)
    assert ep[x0, x1] == HEAD and ep[x1, x0] == HEAD  # bidirected preserved
    assert ep[p, x0] == HEAD and ep[x0, p] == CIRCLE  # CONFLICT -> CIRCLE on output


def test_stop_test_flags() -> None:
    g = complete_lpcmci_pag(2, 1)  # has ? (contemp) and L (lagged) middle marks
    assert g.has_nonempty_middle()
    assert g.has_nonempty_middle_incl_bang()
    # collapse all to empty
    for a in range(g.grid_n):
        for b in range(a + 1, g.grid_n):
            if g.edge_exists(a, b):
                g.set_middle(a, b, MM_EMPTY)
    assert not g.has_nonempty_middle()
    assert not g.has_nonempty_middle_incl_bang()


def test_edge_removal_clears_marks() -> None:
    g = LPCMCIPAG(2, 1)
    a, b = grid_id(0, 0, 1), grid_id(1, 0, 1)
    g.add_edge(a, b, CIRCLE, CIRCLE, MM_Q)
    assert g.edge_exists(a, b)
    g.remove_edge(a, b)
    assert not g.edge_exists(a, b)
    assert g.mark(a, b) == NO_EDGE and g.middle(a, b) == MM_EMPTY
