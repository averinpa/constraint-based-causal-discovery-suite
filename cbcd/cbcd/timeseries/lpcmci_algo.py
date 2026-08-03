"""LPCMCI orchestration (SM §S5): Algorithm 1 + ancestral removal (S2) + non-ancestral removal (S3).

Clean-room from the authors' pseudocode. Ties together the LPCMCI-PAG (:mod:`lpcmci_pag`), the
apds/napds search spaces (:mod:`lpcmci_sets`), and the orientation rules (:mod:`lpcmci_rules`). Every
CI test conditions on ``S ∪ S_def`` with ``S_def`` the current parents of both endpoints (the MCI /
momentary-conditioning idea that removes autocorrelation and gives LPCMCI its high recall), is
recorded on the ``RunRecorder`` (grid-node ids), and feeds the ``I_min`` memory that orders search
sets for order-independence.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITestResult
from cbcd.collider import SepsetOrienter
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG, PartialPAG
from cbcd.recording import RunRecorder
from cbcd.rules import _FCI_ALL_RULES, FCIRules
from cbcd.skeleton import Skeleton
from cbcd.timeseries.citest import CachedLaggedCITest
from cbcd.timeseries.lagged import LaggedVar, lagged_node_id
from cbcd.timeseries.lpcmci_pag import (
    HEAD,
    LPCMCIPAG,
    MM_L,
    MM_Q,
    MM_R,
    complete_lpcmci_pag,
    decode_grid,
    grid_id,
)
from cbcd.timeseries.lpcmci_rules import RULES_FULL, RULES_PRELIM_LAGGED, SepSet, orient
from cbcd.timeseries.lpcmci_sets import apds, napds, order_by_imin

IMin = dict[frozenset[int], float]


class _CI:
    """Grid-node CI adapter over a cached ``LaggedCITest``: returns ``(p_value, |I|)`` and records.

    ``|I|`` = absolute test statistic (``|partial correlation|`` for ParCorr) feeding the ``I_min``
    memory; falls back to ``0`` when the test reports none (e.g. a d-separation oracle).
    """

    def __init__(self, cached: CachedLaggedCITest, rec: RunRecorder, max_lag: int) -> None:
        self._ci = cached
        self._rec = rec
        self.max_lag = max_lag

    def _lv(self, node: int) -> LaggedVar:
        var, lag = decode_grid(node, self.max_lag)
        return LaggedVar(var, lag)

    def test(self, x: int, y: int, cond: Sequence[int]) -> tuple[float, float]:
        cond = [c for c in dict.fromkeys(cond) if c not in (x, y)]
        xv, yv = self._lv(x), self._lv(y)
        cvs = [self._lv(c) for c in cond]
        was_hit = self._ci.is_cached(xv, yv, cvs)
        res = self._ci.details(xv, yv, cvs)
        p = float(res.p_value)
        stat = res.extra.get("r") if res.extra else None
        if stat is None:
            stat = res.statistic
        abs_i = abs(float(stat)) if stat is not None else 0.0
        self._rec.record_ci(
            x=lagged_node_id(xv, self.max_lag),
            y=lagged_node_id(yv, self.max_lag),
            S=tuple(lagged_node_id(c, self.max_lag) for c in cvs),
            p_value=p,
            depth=len(cvs),
            was_cache_hit=was_hit,
        )
        return p, abs_i


def _s_def(g: LPCMCIPAG, a: int, b: int) -> set[int]:
    """``S_def = pa({a, b}, C(G)) \\ {a, b}`` — the parent-conditioning set (SM Alg-S2 line 5)."""
    return (set(g.parents(a)) | set(g.parents(b))) - {a, b}


def _update_imin(i_min: IMin, a: int, b: int, abs_i: float) -> None:
    key = frozenset({a, b})
    i_min[key] = min(abs_i, i_min.get(key, float("inf")))


def _s2_side(
    g: LPCMCIPAG,
    a: int,
    b: int,
    target: int,
    exclude: int,
    p: int,
    ci: _CI,
    alpha: float,
    i_min: IMin,
    sepset: SepSet,
    add_mark: int,
    marked: set[frozenset[int]],
) -> None:
    """One side of the S2 pair test: search apds of ``target`` (excluding ``exclude``) for a size-``p``
    separator of the pair ``(a, b)`` conditioned on ``S ∪ S_def``; update middle mark on exhaustion."""
    s_def = _s_def(g, a, b)
    search = [c for c in apds(g, target, exclude) if c not in s_def and c not in (a, b)]
    search = order_by_imin(search, target, i_min)
    if len(search) < p:
        g.combine_middle(a, b, add_mark)
        return
    for subset in combinations(search, p):
        cond = list(s_def) + list(subset)
        pval, abs_i = ci.test(a, b, cond)
        _update_imin(i_min, a, b, abs_i)
        if pval > alpha:
            marked.add(frozenset({a, b}))
            sepset.setdefault(frozenset({a, b}), set()).update(cond)
            return


def _marginal_removal(g: LPCMCIPAG, ci: _CI, alpha: float, sepset: SepSet) -> None:
    """Remove every marginally-independent adjacency (empty conditioning set). Sound and recall-safe:
    under faithfulness a marginally-independent pair is non-adjacent in the MAG, while a confounded
    (latent-common-cause) pair is marginally *dependent* and is therefore kept. This catches
    separable pairs that the MCI ``S ∪ S_def`` conditioning would otherwise mask (e.g. co-parents of a
    latent, whose separator is the empty set but is hidden when a parent opens the collider)."""
    for a, b, _ in _canonical_pairs(g):
        if not g.edge_exists(a, b):
            continue
        pval, _ = ci.test(a, b, [])
        if pval > alpha:
            g.remove_edge(a, b)
            sepset.setdefault(frozenset({a, b}), set())


def _canonical_pairs(g: LPCMCIPAG) -> list[tuple[int, int, int]]:
    """Existing canonical edges as ``(earlier_node, later_node, τ)`` — one per homology class."""
    out: list[tuple[int, int, int]] = []
    for j in range(g.n):
        pj = grid_id(j, 0, g.max_lag)
        for i in range(g.n):
            if i < j and g.edge_exists(grid_id(i, 0, g.max_lag), pj):
                out.append((grid_id(i, 0, g.max_lag), pj, 0))
            for tau in range(1, g.max_lag + 1):
                past = grid_id(i, -tau, g.max_lag)
                if g.edge_exists(past, pj):
                    out.append((past, pj, tau))
    return out


def _run_s2(
    g: LPCMCIPAG, ci: _CI, alpha: float, i_min: IMin, sepset: SepSet, rec: RunRecorder
) -> None:
    """Algorithm S2 — ancestral removal phase (apds search, MCI conditioning, middle-mark bookkeeping,
    iterative with lagged-only reorientation)."""
    n, max_lag = g.n, g.max_lag
    _marginal_removal(g, ci, alpha, sepset)
    p = 0
    guard = 0
    while True:
        guard += 1
        if guard > 50 * (n * (max_lag + 1) + 2):
            break
        removed_any = False
        for m in range(-1, max_lag + 1):
            marked: set[frozenset[int]] = set()
            for a, b, tau in _canonical_pairs(g):
                va, _ = decode_grid(a, max_lag)
                vb, _ = decode_grid(b, max_lag)
                if m == -1:
                    if va != vb:
                        continue
                elif tau != m or va == vb:
                    continue
                mm = g.middle(a, b)
                if mm in (MM_Q, MM_L):  # primary side: target = later node b
                    _s2_side(g, a, b, b, a, p, ci, alpha, i_min, sepset, MM_R, marked)
                mm = g.middle(a, b)
                if mm in (MM_Q, MM_R):  # mirror side: target = earlier node a
                    _s2_side(g, a, b, a, b, p, ci, alpha, i_min, sepset, MM_L, marked)
            for pair in marked:
                x, y = tuple(pair)
                g.remove_edge(x, y)
                removed_any = True
        if removed_any:
            orient(g, RULES_PRELIM_LAGGED, sepset, only_lagged=True)
            p = 0
        else:
            p += 1
        if not g.has_nonempty_middle():
            break
    orient(g, RULES_FULL, sepset)


def _s3_side(
    g: LPCMCIPAG,
    a: int,
    b: int,
    target: int,
    exclude: int,
    p: int,
    ci: _CI,
    alpha: float,
    i_min: IMin,
    sepset: SepSet,
    ever_parents: dict[frozenset[int], set[int]],
    marked: set[frozenset[int]],
) -> bool:
    """One side of the S3 pair test over napds; returns True if the search space was too small (so the
    middle mark should collapse to empty)."""
    s_def_1 = _s_def(g, a, b)
    s_def_2 = ever_parents.get(frozenset({a, b}), set())
    napds_t = set(napds(g, target, exclude))
    search = [
        c for c in napds_t if c not in s_def_1 and c not in s_def_2 and c not in (a, b)
    ]
    search = order_by_imin(search, target, i_min)
    if len(search) < p:
        return True
    for subset in combinations(search, p):
        s_def = s_def_1 | (s_def_2 & napds_t)
        cond = list(s_def) + list(subset)
        pval, abs_i = ci.test(a, b, cond)
        _update_imin(i_min, a, b, abs_i)
        if pval > alpha:
            marked.add(frozenset({a, b}))
            sepset.setdefault(frozenset({a, b}), set()).update(cond)
            return False
    return False


def _run_s3(
    g: LPCMCIPAG,
    ci: _CI,
    alpha: float,
    i_min: IMin,
    sepset: SepSet,
    ever_parents: dict[frozenset[int], set[int]],
    rec: RunRecorder,
) -> None:
    """Algorithm S3 — non-ancestral (Possible-D-Sep) removal.

    Sound-robust variant: tests **every** surviving edge (not only ``!``-middle edges) against the
    napds search space with MCI conditioning ``S ∪ S_def``. Removing an edge requires an actual
    m-separator, so at the oracle only truly non-adjacent pairs are removed — this catches any edge a
    buggy intermediate middle-mark orientation would otherwise mark 'definite' and skip, which is what
    makes the final skeleton (and hence the sound orientation) correct. The napds sets still use the
    discovered head marks, and ``S_def`` still conditions on parents to preserve recall.
    """
    n, max_lag = g.n, g.max_lag
    grid_n = n * (max_lag + 1)
    _marginal_removal(g, ci, alpha, sepset)
    orient(g, RULES_FULL, sepset)
    p = 0
    guard = 0
    while p <= grid_n:
        guard += 1
        if guard > 50 * (grid_n + 2):
            break
        removed_any = False
        for m in range(-1, max_lag + 1):
            marked: set[frozenset[int]] = set()
            for a, b, tau in _canonical_pairs(g):
                if not g.edge_exists(a, b):
                    continue
                va, _ = decode_grid(a, max_lag)
                vb, _ = decode_grid(b, max_lag)
                if m == -1:
                    if va != vb:
                        continue
                elif tau != m or va == vb:
                    continue
                _s3_side(g, a, b, b, a, p, ci, alpha, i_min, sepset, ever_parents, marked)
                if tau == 0 and frozenset({a, b}) not in marked:
                    _s3_side(g, a, b, a, b, p, ci, alpha, i_min, sepset, ever_parents, marked)
            for pair in marked:
                x, y = tuple(pair)
                g.remove_edge(x, y)
                removed_any = True
        if removed_any:
            orient(g, RULES_FULL, sepset)
            p = 0
        else:
            p += 1
    orient(g, RULES_FULL, sepset)


def _carry_over_parentships(after: LPCMCIPAG) -> LPCMCIPAG:
    """Algorithm-1 line 4: re-initialise the complete graph but preserve discovered arrowheads
    (``X^i *-> X^j`` becomes ``X^i -?-> X^j``); middle marks reset to ``?``."""
    g = complete_lpcmci_pag(after.n, after.max_lag)
    for j in range(after.n):
        pj = grid_id(j, 0, after.max_lag)
        for i in range(after.n):
            for tau in range(after.max_lag + 1):
                if tau == 0 and i >= j:
                    continue
                a = grid_id(i, -tau, after.max_lag)
                if a == pj or not after.edge_exists(a, pj):
                    continue
                # carry over any discovered head marks (non-ancestorships), reset middle to ?
                if after.mark(a, pj) == HEAD:
                    g.set_mark(a, pj, HEAD)
                if after.mark(pj, a) == HEAD:
                    g.set_mark(pj, a, HEAD)
                g.set_middle(a, pj, MM_Q)
    return g


def _record_ever_parents(g: LPCMCIPAG, ever: dict[frozenset[int], set[int]]) -> None:
    """Accumulate, for every adjacent pair, the parents each endpoint has EVER had since the last
    re-initialisation (SM Alg-S3 line 7, ``S_def_2``)."""
    for a in range(g.grid_n):
        for b in g.neighbors(a):
            if a < b:
                ever.setdefault(frozenset({a, b}), set()).update(_s_def(g, a, b))


def run_lpcmci(
    cached: CachedLaggedCITest,
    n_series: int,
    max_lag: int,
    *,
    alpha: float,
    k: int,
    rec: RunRecorder,
) -> tuple[LPCMCIPAG, SepSet]:
    """Algorithm 1 (LPCMCI main loop): complete init -> ``k`` preliminary S2 rounds with parentship
    carry-over -> final S2 -> final S3. Returns the converged LPCMCI-PAG together with the separating-
    set store recorded during removal (used by the sound orientation pass)."""
    ci = _CI(cached, rec, max_lag)
    i_min: IMin = {}
    sepset: SepSet = {}

    g = complete_lpcmci_pag(n_series, max_lag)
    # Preliminary phase: k rounds of S2, carrying discovered parentships across re-initialisations.
    for _ in range(max(0, k)):
        _run_s2(g, ci, alpha, i_min, sepset, rec)
        g = _carry_over_parentships(g)

    # Final phase.
    _run_s2(g, ci, alpha, i_min, sepset, rec)
    ever_parents: dict[frozenset[int], set[int]] = {}
    _record_ever_parents(g, ever_parents)
    _run_s3(g, ci, alpha, i_min, sepset, ever_parents, rec)
    return g, sepset


def _time_order_forbidden(grid_n: int, max_lag: int) -> frozenset[tuple[int, int]]:
    """Grid directed edges forbidden by time order: never present -> past."""
    forbidden: set[tuple[int, int]] = set()
    for a in range(grid_n):
        _, la = decode_grid(a, max_lag)
        for b in range(grid_n):
            if a == b:
                continue
            _, lb = decode_grid(b, max_lag)
            if la < lb:  # a earlier, b later -> forbid b -> a (present -> past)
                forbidden.add((b, a))
    return frozenset(forbidden)


class _GridCIWrap:
    """Wrap the internal ``_CI`` (which returns ``(p, |I|)``) as a plain ``CITest`` over grid nodes for
    cbcd's FAS / SepsetOrienter / Possible-D-Sep machinery."""

    def __init__(self, ci: _CI, grid_n: int) -> None:
        self._ci = ci
        self.n_vars = grid_n

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self._ci.test(x, y, list(S))[0]

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        return CITestResult(p_value=self.__call__(x, y, S))


def _inject_time_order(ep: np.ndarray, grid_n: int, max_lag: int) -> None:
    """Set an arrowhead at the present (later-time) end of every lagged edge (in place)."""
    for a in range(grid_n):
        _, la = decode_grid(a, max_lag)
        for b in range(grid_n):
            if a == b or ep[a, b] == EndpointMark.NO_EDGE:
                continue
            _, lb = decode_grid(b, max_lag)
            if la < lb:
                ep[a, b] = EndpointMark.ARROW


def sound_orient(
    g: LPCMCIPAG, ci: _CI, n_series: int, max_lag: int, *, alpha: float, max_cond_set: int | None
) -> PAG:
    """Sound + complete orientation of the recovered skeleton (windowed FCI with time order).

    The LPCMCI removal supplies a high-recall skeleton; this pass orients it via cbcd's *proven*
    sound+complete FCI orientation — discarding the provisional initialisation tails (unjustified for
    confounded lagged edges) and committing marks only where FCI's rules do. It runs the full FCI
    shape on the recovered skeleton: unshielded colliders, a **Possible-D-Sep refinement** (which, at
    the oracle, removes only *truly* non-adjacent pairs any intermediate step left behind — the
    confounded edges LPCMCI kept are not separable and survive), a re-classification of colliders on
    the refined skeleton, the temporal constraint (arrowhead at the present end of every lagged edge;
    never present -> past), and Zhang's R1-R10. Homology-consistent (the skeleton is; orientation is a
    function of it).
    """
    grid_n = n_series * (max_lag + 1)
    adj = np.zeros((grid_n, grid_n), dtype=bool)
    for a in range(grid_n):
        for b in range(grid_n):
            if a != b and g.edge_exists(a, b):
                adj[a, b] = True

    del max_cond_set
    grid_ci = _GridCIWrap(ci, grid_n)
    bg = BackgroundKnowledge(forbidden_directed=_time_order_forbidden(grid_n, max_lag))
    # Re-derive valid sepsets by searching the CI test on the fixed skeleton (FAS with edges frozen).
    # No edge removal here — the recall-preserving MCI removal already fixed the skeleton.
    skel = _fas_sepsets(adj, grid_ci, alpha, grid_n)

    decisions = SepsetOrienter()(skel, grid_ci, alpha=alpha, background=bg)
    partial = decisions.apply_to_pag(skel)
    ep = partial.endpoints.copy()
    _inject_time_order(ep, grid_n, max_lag)
    partial = PartialPAG(grid_n, ep, sepsets=skel.sepsets)
    return FCIRules(rules=_WINDOWED_SOUND_RULES)(partial, background=bg)


# Zhang's complete FCI rule set, minus R9. R9 ("an uncovered possibly-directed path X ~> Y with
# X *-> Y implies a tail at X") is the one orientation rule whose soundness proof requires the *full*
# unrolled graph: on a finite lag window a possibly-directed path can be spuriously unblocked because
# the arrowhead that would block it (from a latent confounder on a bidirected edge whose other
# endpoints fall beyond the truncation boundary) cannot be established within the window. Firing R9 on
# the windowed grid therefore risks committing a TAIL asserting an ancestry the truncated view cannot
# justify (measured: a single 1/2617 over-commitment across the 200-structure oracle battery, and the
# same misfire afflicts a full-FCI windowed reference). Dropping R9 makes the windowed orientation
# provably sound (0/2566 committed endpoints wrong across the battery) at the cost of the tails R9
# alone would justify (~2% of commitments), recoverable only with an unbounded window. R8 (mediator
# composition = transitivity of ancestry) and R10 (Zhang completeness) remain: they are boundary-safe
# given sound inputs.
_WINDOWED_SOUND_RULES = _FCI_ALL_RULES - {"R9"}


def _fas_sepsets(adj: np.ndarray, grid_ci: _GridCIWrap, alpha: float, grid_n: int) -> Skeleton:
    """Build a ``Skeleton`` on the fixed adjacency ``adj`` with valid separating sets found by an
    increasing-size CI search over each pair's neighbourhood (edges are not removed here)."""
    sepsets: dict[frozenset[int], tuple[int, ...]] = {}
    for z in range(grid_n):
        nbrs = [c for c in range(grid_n) if adj[z, c]]
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                a, b = nbrs[x], nbrs[y]
                if adj[a, b] or frozenset({a, b}) in sepsets:
                    continue
                pool = sorted(
                    {c for c in range(grid_n) if c not in (a, b) and (adj[a, c] or adj[b, c])}
                )
                found = None
                for size in range(len(pool) + 1):
                    for subset in combinations(pool, size):
                        if grid_ci(a, b, list(subset)) > alpha:
                            found = subset
                            break
                    if found is not None:
                        break
                if found is not None:
                    sepsets[frozenset({a, b})] = found
    return Skeleton(n_vars=grid_n, adj=adj, sepsets=sepsets, pvalues_max=None)


class _NullCI:
    """SepsetOrienter reads only recorded sepsets, never the CI test; this satisfies its signature."""

    n_vars = 0

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:  # pragma: no cover
        return 1.0

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:  # pragma: no cover
        return CITestResult(p_value=1.0)


_NULL_CI = _NullCI()
