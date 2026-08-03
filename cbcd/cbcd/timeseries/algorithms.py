"""Time-series algorithm composition: pcmci(), pcmci_plus()."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cbcd._run import lagged_run
from cbcd.citest.protocol import CITestResult
from cbcd.collider import MajorityColliderOrienter
from cbcd.exceptions import CBCDInputError
from cbcd.graph.cpdag import PartialCPDAG
from cbcd.graph.marks import EndpointMark
from cbcd.recording import RunRecorder
from cbcd.rules import MeekRules
from cbcd.skeleton import Skeleton
from cbcd.timeseries.citest import CachedLaggedCITest, LaggedCITest
from cbcd.timeseries.graph import TimeSeriesCPDAG
from cbcd.timeseries.lagged import (
    LaggedBackgroundKnowledge,
    LaggedDataset,
    LaggedVar,
    lagged_node_id,
)
from cbcd.timeseries.skeleton import LaggedSkeletonAlgorithm, PC1Skeleton


def pcmci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | Literal["parcorr"] = "parcorr",
    alpha: float = 0.05,
    pc_alpha: float | None = None,
    skeleton: LaggedSkeletonAlgorithm | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
) -> TimeSeriesCPDAG:
    """PCMCI (Runge et al. 2019) — vanilla, lagged-only.

    Two-stage:

    1. **PC₁** — for each target ``Y_t``, prune candidate lagged parents
       via per-pair CI tests with growing conditioning sets.
    2. **MCI** — for each candidate ``(X_{t-τ}, Y_t)`` with τ ∈ [1, max_lag],
       condition on ``̂P(Y_t) ∪ {shifted parents of X_{t-τ}}`` and test;
       the edge exists iff the test rejects independence at ``alpha``.

    Returns a ``TimeSeriesCPDAG`` with all lagged edges directed
    past→present and no contemporaneous edges (decision: vanilla PCMCI
    assumes contemporaneous independence; use ``pcmci_plus`` for
    contemporaneous discovery — deferred).

    ``pc_alpha`` defaults to ``alpha`` when ``None``. Automatic ``pc_alpha``
    grid selection (open question O4) is deferred.
    """
    if pc_alpha is None:
        pc_alpha = alpha
    if not (0.0 < pc_alpha < 1.0):
        raise CBCDInputError(f"pc_alpha must be in (0, 1), got {pc_alpha}")
    if n_jobs != 1:
        raise CBCDInputError("n_jobs != 1 not yet implemented in this slice; pass n_jobs=1")

    with lagged_run(
        data,
        ci_test=ci_test,
        algorithm="pcmci",
        params={"alpha": alpha, "pc_alpha": pc_alpha, "n_jobs": n_jobs},
        alpha=alpha,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n, max_lag, cached = ctx.n_series, ctx.max_lag, ctx.ci
        skel_algo = skeleton if skeleton is not None else PC1Skeleton()
        skel = skel_algo(cached, alpha=pc_alpha, background=background, recorder=ctx.rec, n_jobs=n_jobs)

        endpoints = np.zeros((max_lag + 1, n, n), dtype=np.int8)

        # MCI step: test every candidate lagged edge (X_{t-τ}, Y_t) using the
        # conditioning set ̂P(Y) ∪ shifted ̂P(X). Add the edge iff the test
        # rejects independence at α.
        for y_var in range(n):
            target = LaggedVar(y_var, 0)
            py = skel.parents.get(target, frozenset())
            for x_var in range(n):
                x_parents = skel.parents.get(LaggedVar(x_var, 0), frozenset())
                for tau in range(1, max_lag + 1):
                    candidate = LaggedVar(x_var, -tau)
                    if background is not None and background.is_forbidden_lagged(candidate, target):
                        continue
                    # Shifted parents of X at lag -τ: each (Z, -σ) ∈ ̂P(X)
                    # becomes (Z, -(τ + σ)) in Y's frame.
                    shifted = {
                        LaggedVar(p.var, p.lag - tau) for p in x_parents if -p.lag + tau <= max_lag
                    }
                    S_set = (py | shifted) - {candidate, target}
                    S_sorted = sorted(S_set, key=lambda lv: (lv.var, lv.lag))
                    was_hit = cached.is_cached(candidate, target, S_sorted)
                    p = cached(candidate, target, S_sorted)
                    ctx.rec.record_ci(
                        x=lagged_node_id(candidate, max_lag),
                        y=lagged_node_id(target, max_lag),
                        S=tuple(lagged_node_id(s, max_lag) for s in S_sorted),
                        p_value=p,
                        depth=len(S_sorted),
                        was_cache_hit=was_hit,
                    )
                    if p <= alpha:
                        endpoints[tau, x_var, y_var] = EndpointMark.ARROW

        ctx.summary = {
            "n_series": n,
            "max_lag": max_lag,
            "n_edges": int((endpoints == EndpointMark.ARROW).sum()),
        }
        ctx.result = TimeSeriesCPDAG(
            n_vars=n, max_lag=max_lag, endpoints=endpoints, var_names=data.var_names
        )
    return ctx.result


# ==================================================================================================
# PCMCI+ (Runge, UAI 2020) — lagged + contemporaneous discovery, cbcd-native.
#
# The working graph is cbcd's own ``EndpointMark`` representation over the lagged grid (grid node
# ``var*(max_lag+1) + (-lag)``); contemporaneous edges start undirected (TAIL–TAIL), lagged edges are
# pre-oriented by time order (arrowhead at the present endpoint). Orientation reuses cbcd's shared
# rule engine: ``MajorityColliderOrienter`` (majority-rule R0) + ``MeekRules`` (R1–R3). The only
# time-series-specific piece is the momentary-CI (MCI) conditioning, wrapped as a ``CITest`` below.
# ==================================================================================================


def _grid_id(var: int, tau: int, max_lag: int) -> int:
    """Grid-node index of ``var`` at lag ``-tau`` (tau >= 0)."""
    return var * (max_lag + 1) + tau


class _MCICITest:
    """cbcd-native momentary-CI (MCI) test over the lagged grid.

    Presents the ``CITest`` contract (integer grid nodes) to cbcd's shared orienters, and internally
    augments every conditioning set with both endpoints' estimated lagged parents (shifted into each
    endpoint's time frame, Runge 2020) before deferring to the cached ``LaggedCITest``. Conditioning
    lags outside ``[-max_lag, 0]`` are dropped (cbcd's fixed-window lagged design). Every call is
    recorded on the ``RunRecorder``.
    """

    def __init__(
        self,
        cached: CachedLaggedCITest,
        rec: RunRecorder,
        lagged_parents: dict[int, list[LaggedVar]],
        max_lag: int,
        n_series: int,
    ) -> None:
        self._cached = cached
        self._rec = rec
        self._lp = lagged_parents
        self.max_lag = max_lag
        self.n_vars = n_series * (max_lag + 1)

    def _decode(self, node: int) -> LaggedVar:
        var, rem = divmod(int(node), self.max_lag + 1)
        return LaggedVar(var, -rem)

    def _mci_conditions(
        self, x: int, y: int, S: Sequence[int]
    ) -> tuple[LaggedVar, LaggedVar, list[LaggedVar]]:
        xnode, ynode = self._decode(x), self._decode(y)
        Z: list[LaggedVar] = [self._decode(s) for s in S]
        for node in (xnode, ynode):
            # ``.get`` (not ``[]``): the local-temporal builder materialises parents only for present-
            # region series, so a lagged-parent-only endpoint has no entry -> no extra conditioning
            # from that side. Sound: the *present* endpoint's parents d-separate it from non-parents.
            # For pcmci/pcmci_plus the dict covers every series, so this is behaviour-preserving there.
            for parent in self._lp.get(node.var, ()):
                shifted = LaggedVar(parent.var, parent.lag + node.lag)
                if shifted not in (xnode, ynode) and shifted not in Z:
                    Z.append(shifted)
        Z = [z for z in Z if -self.max_lag <= z.lag <= 0]
        return xnode, ynode, Z

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        xnode, ynode, Z = self._mci_conditions(x, y, S)
        was_hit = self._cached.is_cached(xnode, ynode, Z)
        p = float(self._cached(xnode, ynode, Z))
        self._rec.record_ci(
            x=lagged_node_id(xnode, self.max_lag),
            y=lagged_node_id(ynode, self.max_lag),
            S=tuple(lagged_node_id(z, self.max_lag) for z in Z),
            p_value=p,
            depth=len(Z),
            was_cache_hit=was_hit,
        )
        return p

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        return CITestResult(p_value=self.__call__(x, y, S))

    def is_cached(self, x: int, y: int, S: Sequence[int]) -> bool:
        xnode, ynode, Z = self._mci_conditions(x, y, S)
        return self._cached.is_cached(xnode, ynode, Z)


def _pcmci_plus_skeleton(
    adj: NDArray[np.bool_],
    present: list[int],
    mci: _MCICITest,
    n: int,
    max_lag: int,
    alpha: float,
    max_cond_set: int | None,
) -> None:
    """MCI skeleton phase (Runge 2020, Alg 2): prune lagged (τ≥1) and contemporaneous (τ=0)
    adjacencies, iterating conditioning subsets over *contemporaneous* neighbours of the present
    target only. Mutates ``adj`` (grid adjacency) in place; order-independent (PC-stable: the
    contemporaneous-neighbour snapshot is frozen per cardinality)."""
    cap = n if max_cond_set is None else max_cond_set
    p = 0
    while p <= cap:
        cadj = {b: [w for w in present if w != b and adj[w, b]] for b in present}
        remaining = [
            (a, b)
            for b in present
            for a in range(adj.shape[0])
            if a != b and adj[a, b] and len([w for w in cadj[b] if w != a]) >= p
        ]
        if not remaining:
            break
        for a, b in remaining:
            if not adj[a, b]:  # removed earlier this round
                continue
            pool = [w for w in cadj[b] if w != a]
            for subset in combinations(pool, p):
                if mci(a, b, list(subset)) > alpha:  # independent
                    adj[a, b] = adj[b, a] = False
                    break
        p += 1


def _pcmci_plus_apply_colliders(
    ep: NDArray[np.int8],
    colliders: frozenset[tuple[int, int, int]],
    max_lag: int,
) -> None:
    """Orient each majority-rule collider's *contemporaneous* arms into the centre (arrowhead at the
    centre), resolving order conflicts (an edge an earlier collider oriented the other way) to
    undirected. Lagged arms are already directed by time order, so they are left untouched."""
    directed: set[tuple[int, int]] = set()
    conflicted: set[frozenset[int]] = set()
    for x, z, y in colliders:
        for endpoint in (x, y):
            # Contemporaneous arm only: both endpoints at lag 0 (grid id divisible by max_lag+1).
            if endpoint % (max_lag + 1) != 0 or z % (max_lag + 1) != 0:
                continue
            key = frozenset({endpoint, z})
            if key in conflicted:
                continue
            if (z, endpoint) in directed:  # opposite orientation already committed -> conflict
                conflicted.add(key)
                ep[endpoint, z] = ep[z, endpoint] = EndpointMark.TAIL
                directed.discard((z, endpoint))
                continue
            ep[endpoint, z] = EndpointMark.ARROW
            ep[z, endpoint] = EndpointMark.TAIL
            directed.add((endpoint, z))


def _grid_to_ts_endpoints(
    grid_ep: NDArray[np.int8], n: int, max_lag: int
) -> NDArray[np.int8]:
    """Project the oriented lagged-grid endpoint matrix onto the ``(max_lag+1, n, n)``
    ``TimeSeriesCPDAG`` layout (present-anchored edges only)."""
    ep = np.zeros((max_lag + 1, n, n), dtype=np.int8)
    for i in range(n):
        pi = _grid_id(i, 0, max_lag)
        for j in range(n):
            pj = _grid_id(j, 0, max_lag)
            if i != j and grid_ep[pi, pj] != EndpointMark.NO_EDGE:  # contemporaneous edge
                ep[0, i, j] = grid_ep[pi, pj]
                ep[0, j, i] = grid_ep[pj, pi]
            # Lagged edges include autoregressive ones (i == j, different lags).
            for tau in range(1, max_lag + 1):
                if grid_ep[_grid_id(i, tau, max_lag), pj] == EndpointMark.ARROW:
                    ep[tau, i, j] = EndpointMark.ARROW
    return ep


def pcmci_plus(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | Literal["parcorr"] = "parcorr",
    tau_max: int | None = None,
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    var_names: tuple[str, ...] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
    skeleton: LaggedSkeletonAlgorithm | None = None,
) -> TimeSeriesCPDAG:
    """PCMCI+ (Runge, UAI 2020) — lagged **and** contemporaneous causal discovery.

    Upgrades :func:`pcmci` (lagged-only) with contemporaneous link discovery and orientation, the
    completeness gain for the causally-sufficient time-series cell. Three phases:

    1. **Lagged (PC₁):** estimate each variable's lagged parents ``B̂⁻`` (reuses :class:`PC1Skeleton`,
       the same phase ``pcmci`` uses).
    2. **Contemporaneous + MCI:** initialise all contemporaneous adjacencies plus the ``B̂⁻`` lagged
       edges, then prune with the momentary-CI test — conditioning on contemporaneous subsets *plus
       both endpoints' lagged parents* — which blocks lagged paths and removes autocorrelation for
       better-calibrated tests.
    3. **Orientation:** majority-rule collider phase (:class:`MajorityColliderOrienter`, MCI re-tests)
       + Meek R1–R3 (:class:`MeekRules`) — cbcd's own shared rule engine — applied on the lagged grid
       with lagged edges pre-oriented by time order. Contemporaneous edges get collider/Meek marks.

    The working graph throughout is cbcd's :class:`EndpointMark` representation on the lagged grid;
    the result is a :class:`TimeSeriesCPDAG` (lagged directed past→present; contemporaneous directed
    or undirected). Causal sufficiency assumed (no latents; the latent case is :func:`lpcmci`).
    ``tau_max`` must equal the dataset's ``max_lag``. Background knowledge is threaded into the lagged
    phase and forbids contemporaneous adjacencies (``no_contemporaneous``) / orientations it prohibits.
    """
    max_lag = data.max_lag if tau_max is None else int(tau_max)
    if max_lag != data.max_lag:
        raise CBCDInputError(f"tau_max={max_lag} must equal the dataset max_lag={data.max_lag}")

    with lagged_run(
        data,
        ci_test=ci_test,
        algorithm="pcmci_plus",
        params={"alpha": alpha, "tau_max": max_lag, "max_cond_set": max_cond_set},
        alpha=alpha,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n, cached, rec = ctx.n_series, ctx.ci, ctx.rec
        grid_n = n * (max_lag + 1)
        present = [_grid_id(v, 0, max_lag) for v in range(n)]

        # Phase 1 — lagged parents B̂⁻ via PC1 (shared with pcmci).
        skel_algo = skeleton if skeleton is not None else PC1Skeleton()
        skel = skel_algo(
            cached, alpha=alpha, max_cond_set=max_cond_set, background=background, recorder=rec
        )
        lagged_parents: dict[int, list[LaggedVar]] = {
            j: sorted(
                skel.parents.get(LaggedVar(j, 0), frozenset()), key=lambda lv: (lv.var, -lv.lag)
            )
            for j in range(n)
        }
        mci = _MCICITest(cached, rec, lagged_parents, max_lag, n)

        # Initialise the grid adjacency: all contemporaneous pairs + the lagged B̂⁻ edges.
        adj = np.zeros((grid_n, grid_n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (
                    background is not None
                    and background.is_forbidden_lagged(LaggedVar(i, 0), LaggedVar(j, 0))
                    and background.is_forbidden_lagged(LaggedVar(j, 0), LaggedVar(i, 0))
                ):
                    continue
                adj[_grid_id(i, 0, max_lag), _grid_id(j, 0, max_lag)] = True
        for j in range(n):
            for pv in lagged_parents[j]:
                a, b = _grid_id(pv.var, -pv.lag, max_lag), _grid_id(j, 0, max_lag)
                adj[a, b] = adj[b, a] = True

        # Phase 2 — MCI skeleton over contemporaneous conditioning subsets.
        _pcmci_plus_skeleton(adj, present, mci, n, max_lag, alpha, max_cond_set)

        # Build the EndpointMark grid: contemporaneous edges undirected, lagged edges arrow-at-present.
        ep = np.zeros((grid_n, grid_n), dtype=np.int8)
        present_set = set(present)
        for a in range(grid_n):
            for b in range(grid_n):
                if a >= b or not adj[a, b]:
                    continue
                if a in present_set and b in present_set:  # contemporaneous: undirected
                    ep[a, b] = ep[b, a] = EndpointMark.TAIL
                else:  # lagged: arrowhead at the present (later-time) endpoint
                    past, pres = (a, b) if a not in present_set else (b, a)
                    ep[past, pres] = EndpointMark.ARROW
                    ep[pres, past] = EndpointMark.TAIL

        # Phase 3a — majority-rule colliders (R0), centres restricted to present nodes, conditioning
        # subsets drawn from contemporaneous neighbours only (cbcd's shared MajorityColliderOrienter).
        contemp_adj = np.zeros((grid_n, grid_n), dtype=bool)
        for a in present:
            for b in present:
                if a != b and adj[a, b]:
                    contemp_adj[a, b] = True
        decisions = MajorityColliderOrienter()(
            Skeleton(n_vars=grid_n, adj=adj, sepsets={}, pvalues_max=None),
            mci,
            alpha=alpha,
            interior=frozenset(present),
            conditioning_adj=contemp_adj,
            recorder=rec,
        )
        _pcmci_plus_apply_colliders(ep, decisions.colliders, max_lag)

        # Phase 3b — Meek R1–R3 closure on the grid (cbcd's shared MeekRules), lagged edges fixed.
        # Ambiguous triples are carried on the PartialCPDAG so R1 does not treat them as evidence.
        partial = PartialCPDAG(
            n_vars=grid_n, endpoints=ep, ambiguous_triples=decisions.ambiguous
        )
        grid_cpdag = MeekRules(rules=frozenset({"R1", "R2", "R3"}))(partial, recorder=rec)
        grid_ep = grid_cpdag.endpoints.copy()

        # Background: relax any contemporaneous orientation the tiers/rules prohibit to undirected.
        if background is not None:
            for i in range(n):
                for j in range(n):
                    a, b = _grid_id(i, 0, max_lag), _grid_id(j, 0, max_lag)
                    if (
                        grid_ep[a, b] == EndpointMark.ARROW
                        and grid_ep[b, a] == EndpointMark.TAIL
                        and background.is_forbidden_lagged(LaggedVar(i, 0), LaggedVar(j, 0))
                    ):
                        grid_ep[a, b] = grid_ep[b, a] = EndpointMark.TAIL

        endpoints = _grid_to_ts_endpoints(grid_ep, n, max_lag)
        ctx.summary = {
            "n_series": n,
            "max_lag": max_lag,
            "n_lagged_edges": int((endpoints[1:] == EndpointMark.ARROW).sum()),
            "n_contemp_edges": len(TimeSeriesCPDAG(n, max_lag, endpoints).contemporaneous_edges()),
        }
        ctx.result = TimeSeriesCPDAG(
            n_vars=n, max_lag=max_lag, endpoints=endpoints, var_names=data.var_names
        )
    return ctx.result
