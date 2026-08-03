"""LPCMCI — native latent time-series causal discovery (roadmap Phase 5).

5b lands the **skeleton foundation**: the windowed FCI/PC-style adjacency over the lagged grid,
returned as an all-circle ``TimeSeriesPAG`` (unoriented). It reuses the Phase-0 ``PCStable`` through a
grid <-> ``LaggedVar`` CI adapter, so the search is recorder-instrumented for free.

Still to layer on (5b/5c): LPCMCI's iterative ancestral (MCI) conditioning and Possible-D-Sep edge
removal for latents, then the ancestral + FCI-style orientation. Under a perfect oracle with no
latent-only separations, this skeleton already equals the true time-series skeleton.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from cbcd._run import lagged_run
from cbcd.citest.protocol import CITestResult
from cbcd.collider import SepsetOrienter
from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.recording import RunRecorder
from cbcd.refinement import PossibleDSepRefinement
from cbcd.skeleton import PCStable
from cbcd.timeseries.citest import LaggedCITest
from cbcd.timeseries.graph import TimeSeriesPAG
from cbcd.timeseries.lagged import LaggedBackgroundKnowledge, LaggedDataset, LaggedVar
from cbcd.timeseries.lpcmci_algo import _CI, run_lpcmci, sound_orient
from cbcd.timeseries.lpcmci_pag import LPCMCIPAG, grid_id
from cbcd.timeseries.skeleton import PC1Skeleton


class _GridCITest:
    """Adapts a ``LaggedCITest`` to the ``CITest`` protocol over the lagged grid: grid node
    ``var*(max_lag+1) + (-lag)`` <-> ``LaggedVar(var, lag)`` (see ``lagged_node_id``)."""

    def __init__(self, lagged: LaggedCITest, n_series: int, max_lag: int) -> None:
        self._ci = lagged
        self.max_lag = max_lag
        self.n_vars = n_series * (max_lag + 1)
        self._is_cached = getattr(lagged, "is_cached", None)

    def _decode(self, node: int) -> LaggedVar:
        var, rem = divmod(int(node), self.max_lag + 1)
        return LaggedVar(var, -rem)

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return float(self._ci(self._decode(x), self._decode(y), [self._decode(s) for s in S]))

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        return CITestResult(p_value=self.__call__(x, y, S))

    def is_cached(self, x: int, y: int, S: Sequence[int]) -> bool:
        if self._is_cached is None:
            return False
        return bool(self._is_cached(self._decode(x), self._decode(y), [self._decode(s) for s in S]))


def lpcmci_skeleton(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | Literal["parcorr"] = "parcorr",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    possible_dsep: bool = False,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> TimeSeriesPAG:
    """Windowed skeleton over the lagged grid, returned as an all-circle ``TimeSeriesPAG``.

    ``possible_dsep=False`` (default) returns the PC skeleton — the validated foundation (oracle-correct;
    matches the reference LPCMCI on no-latent VARs).

    ``possible_dsep=True`` additionally runs colliders + Possible-D-Sep edge removal (the static
    FCI-skeleton step). **Caveat:** on *latent* time series this *over-removes* — it deletes
    genuinely-confounded edges because partial-correlation conditioning on an arbitrary
    Possible-D-Sep subset uses observed *proxies* of the latent as a false separator. This
    is precisely what LPCMCI's ancestral (MCI) conditioning avoids; that machinery (not naive PDS) is
    the faithful path and is still pending. Use ``True`` only as a tsFCI-style experiment. Recorder-
    bracketed; nodes are the lagged grid.
    """
    with lagged_run(
        data,
        ci_test=ci_test,
        algorithm="lpcmci_skeleton",
        params={
            "alpha": alpha,
            "max_cond_set": max_cond_set,
            "n_series": data.n_vars,
            "max_lag": data.max_lag,
        },
        alpha=alpha,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n, lag, grid_n = ctx.n_series, ctx.max_lag, ctx.grid_n
        grid_ci = _GridCITest(ctx.ci, n, lag)
        skel = PCStable()(grid_ci, alpha=alpha, max_cond_set=max_cond_set, recorder=ctx.rec)

        if possible_dsep:
            # FCI skeleton: colliders -> Possible-D-Sep removal of latent-induced edges.
            decisions = SepsetOrienter()(skel, grid_ci, alpha=alpha, recorder=ctx.rec)
            partial = decisions.apply_to_pag(skel)
            partial = PossibleDSepRefinement()(
                partial, grid_ci, alpha=alpha, max_cond_set=max_cond_set, recorder=ctx.rec
            )
            adj = partial.endpoints != EndpointMark.NO_EDGE
        else:
            adj = skel.adj

        # Refined skeleton -> all-circle PAG over the grid (o-o on every surviving edge).
        ep = np.where(adj, np.int8(EndpointMark.CIRCLE), np.int8(EndpointMark.NO_EDGE))
        np.fill_diagonal(ep, np.int8(EndpointMark.NO_EDGE))
        ctx.summary = {"n_edges": int(adj.sum() // 2)}
        ctx.result = TimeSeriesPAG(n_series=n, max_lag=lag, window=PAG(grid_n, ep))
    return ctx.result


def _mci_cond_set(
    parents: dict[LaggedVar, frozenset[LaggedVar]],
    candidate: LaggedVar,
    target: LaggedVar,
    max_lag: int,
) -> list[LaggedVar]:
    """The MCI (momentary CI) conditioning set for testing the link ``candidate -> target``:
    ``P(target) \\ {candidate}`` plus the estimated parents of ``candidate`` shifted into the target's
    time frame, restricted to the window ``[-max_lag, 0]``. Conditioning on the parents of *both*
    endpoints controls autocorrelation without conditioning on arbitrary descendants/proxies."""
    tau = -candidate.lag
    cond = set(parents.get(LaggedVar(target.var, 0), frozenset()))
    for p in parents.get(LaggedVar(candidate.var, 0), frozenset()):
        shifted = LaggedVar(p.var, p.lag - tau)
        if -max_lag <= shifted.lag <= 0:
            cond.add(shifted)
    cond.discard(candidate)
    cond.discard(target)
    return sorted(cond, key=lambda lv: (lv.var, lv.lag))


def mci_skeleton(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | Literal["parcorr"] = "parcorr",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> TimeSeriesPAG:
    """MCI-conditioned windowed skeleton (the LPCMCI/PCMCI conditioning discipline).

    PC1 estimates each target's lagged parents, then every candidate lagged link is tested with the
    MCI conditioning set (``_mci_cond_set``). Unlike naive Possible-D-Sep this does NOT over-remove
    latent-confounded edges (parity finding: MCI keeps them). It is still not faithful LPCMCI on its
    own: it covers **lagged** links only (contemporaneous discovery + ancestral Possible-D-Sep +
    middle-marks are the remaining pieces). Returned as an all-circle ``TimeSeriesPAG``.
    """
    with lagged_run(
        data,
        ci_test=ci_test,
        algorithm="mci_skeleton",
        params={
            "alpha": alpha,
            "max_cond_set": max_cond_set,
            "n_series": data.n_vars,
            "max_lag": data.max_lag,
        },
        alpha=alpha,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n, lag, grid_n, cached = ctx.n_series, ctx.max_lag, ctx.grid_n, ctx.ci
        is_cached = getattr(cached, "is_cached", None)
        pc1 = PC1Skeleton()(cached, alpha=alpha, max_cond_set=max_cond_set, recorder=ctx.rec)
        parents = pc1.parents

        def nid(var: int, lg: int) -> int:
            return var * (lag + 1) + (-lg)

        adj = np.zeros((grid_n, grid_n), dtype=bool)
        for y_var in range(n):
            target = LaggedVar(y_var, 0)
            for x_var in range(n):
                for tau in range(1, lag + 1):  # lagged links only
                    cand = LaggedVar(x_var, -tau)
                    cond = _mci_cond_set(parents, cand, target, lag)
                    was_hit = bool(is_cached(cand, target, cond)) if is_cached is not None else False
                    p = float(cached(cand, target, cond))
                    ctx.rec.record_ci(
                        x=nid(x_var, -tau),
                        y=nid(y_var, 0),
                        S=tuple(nid(s.var, s.lag) for s in cond),
                        p_value=p,
                        depth=len(cond),
                        was_cache_hit=was_hit,
                    )
                    if p <= alpha:
                        a, b = nid(x_var, -tau), nid(y_var, 0)
                        adj[a, b] = adj[b, a] = True

        ep = np.where(adj, np.int8(EndpointMark.CIRCLE), np.int8(EndpointMark.NO_EDGE))
        np.fill_diagonal(ep, np.int8(EndpointMark.NO_EDGE))
        ctx.summary = {"n_edges": int(adj.sum() // 2)}
        ctx.result = TimeSeriesPAG(n_series=n, max_lag=lag, window=PAG(grid_n, ep))
    return ctx.result


def lpcmci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | Literal["parcorr"] = "parcorr",
    tau_max: int | None = None,
    alpha: float = 0.05,
    k: int = 4,
    max_cond_set: int | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    var_names: tuple[str, ...] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> TimeSeriesPAG:
    """LPCMCI (Gerhardus & Runge, NeurIPS 2020) — high-recall latent time-series discovery.

    Cell 4 (full / time-series / causally *non-sufficient*): recovers lagged and contemporaneous
    links **and** latent-confounder bidirected edges (``<->``), the FCI-analog for stationary time
    series. Sound and complete under faithfulness with no selection bias (Thm 2).

    Faithful clean-room implementation of the authors' published algorithm (SM §S4/§S5), built on a
    native **LPCMCI-PAG** with middle marks + a homology map over the lagged grid
    (:mod:`cbcd.timeseries.lpcmci_pag`): Algorithm 1 = complete init -> ``k`` preliminary ancestral
    (S2) rounds carrying discovered parentships across re-initialisations -> a final S2 -> a
    non-ancestral (S3) removal. Every CI test conditions on ``S ∪ S_def`` (the momentary-CI idea:
    condition on both endpoints' current parents), which removes autocorrelation and is what gives
    LPCMCI its high finite-sample recall over a plain windowed FCI. Search spaces are the apds
    (ancestral) / napds (non-ancestral) sets of §S7; orientation uses the R0′–R10′ / APR / MMR rules of
    §S4 through the Alg-S4 driver. Returns an ordinary-mark windowed :class:`TimeSeriesPAG`.

    ``tau_max`` (required, must equal ``data.max_lag``) and ``k`` (number of preliminary iterations —
    now functional) mirror the reference knobs. ``background`` threads the temporal tier and any
    forbidden contemporaneous adjacencies. Scope: latents, **no selection bias**, stationary.

    Licensing: the reference ``LPCMCI`` is GPL-3.0 while cbcd is MIT; no external source is read or
    copied — this is built solely from the authors' published mathematics (facts) on cbcd's own marks.
    """
    max_lag = data.max_lag if tau_max is None else int(tau_max)
    if max_lag != data.max_lag:
        raise CBCDInputError(f"tau_max={max_lag} must equal the dataset max_lag={data.max_lag}")
    if k < 0:
        raise CBCDInputError(f"k must be >= 0, got {k}")

    with lagged_run(
        data,
        ci_test=ci_test,
        algorithm="lpcmci",
        params={
            "alpha": alpha,
            "tau_max": max_lag,
            "k": k,
            "max_cond_set": max_cond_set,
            "n_series": data.n_vars,
            "max_lag": max_lag,
        },
        alpha=alpha,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n, lag = ctx.n_series, ctx.max_lag
        # LPCMCI removal (high-recall skeleton + sepsets); the middle-mark orientation is used only
        # internally to guide the apds/napds search.
        skeleton_pag, _sepset = run_lpcmci(ctx.ci, n, lag, alpha=alpha, k=k, rec=ctx.rec)
        _apply_background_forbidden(skeleton_pag, background)
        # Sound + complete orientation of the recovered skeleton (windowed FCI + time order),
        # re-deriving sepsets from the CI test for robust soundness.
        window = sound_orient(
            skeleton_pag, _CI(ctx.ci, ctx.rec, lag), n, lag, alpha=alpha, max_cond_set=max_cond_set
        )
        ts = TimeSeriesPAG(
            n_series=n, max_lag=lag, window=window, var_names=var_names or data.var_names
        )
        ctx.summary = {
            "n_edges": int(sum(1 for _ in ts.edges())),
            "n_bidirected": int(
                sum(
                    1
                    for e in ts.edges()
                    if e.mark_at_src == EndpointMark.ARROW and e.mark_at_dst == EndpointMark.ARROW
                )
            ),
        }
        ctx.result = ts
    return ctx.result


def _apply_background_forbidden(
    g: LPCMCIPAG, background: LaggedBackgroundKnowledge | None
) -> None:
    """Honour ``no_contemporaneous`` (remove that adjacency) as background. Time order is already
    enforced structurally (lagged edges carry a head at the present end)."""
    if background is None:
        return
    for i in range(g.n):
        for j in range(i + 1, g.n):
            a, b = grid_id(i, 0, g.max_lag), grid_id(j, 0, g.max_lag)
            if g.edge_exists(a, b) and background.is_forbidden_lagged(
                LaggedVar(i, 0), LaggedVar(j, 0)
            ) and background.is_forbidden_lagged(LaggedVar(j, 0), LaggedVar(i, 0)):
                g.remove_edge(a, b)
