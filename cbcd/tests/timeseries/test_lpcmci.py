"""LPCMCI skeleton foundation (5b) + faithful latent LPCMCI oracle gate (cell 4)."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np

from cbcd.citest.protocol import CITestResult
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark
from cbcd.timeseries import lpcmci, lpcmci_skeleton, mci_skeleton, pcmci, pcmci_plus
from cbcd.timeseries.lagged import LaggedDataset, LaggedVar
from cbcd.timeseries.lpcmci import _mci_cond_set
from tests.oracle import DSeparationOracle


class _TSLatentOracle:
    """Time-series m-separation oracle over an unrolled DAG with hidden (latent) series. The CI
    surface exposes only the ``n_obs`` observed variables (indices ``0..n_obs-1``); latent series
    (higher indices) participate in d-separation but are never queried or conditioned on."""

    def __init__(
        self,
        n_obs: int,
        n_lat: int,
        max_lag: int,
        lagged: list[tuple[int, int, int]],
        contemp: list[tuple[int, int]],
    ) -> None:
        self.n_vars = n_obs
        self.max_lag = max_lag
        n_total = n_obs + n_lat
        self._th = max(4 * max_lag, 8)
        self._tref = max(2 * max_lag, 4)
        g = nx.DiGraph()
        for v in range(n_total):
            for t in range(self._th + 1):
                g.add_node((v, t))
        for i, j in contemp:
            for t in range(self._th + 1):
                g.add_edge((i, t), (j, t))
        for i, j, tau in lagged:
            for t in range(tau, self._th + 1):
                g.add_edge((i, t - tau), (j, t))
        self._g = g

    def _node(self, lv: LaggedVar) -> tuple[int, int]:
        return (lv.var, self._tref + lv.lag)

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> CITestResult:
        cond = {self._node(s) for s in S}
        sep = nx.is_d_separator(self._g, {self._node(x)}, {self._node(y)}, cond)
        return CITestResult(p_value=1.0 if sep else 0.0)


def _grid(var: int, lag: int, max_lag: int) -> int:
    return var * (max_lag + 1) + (-lag)


class _LaggedOracle:
    """d-separation oracle over an unrolled time-series DAG, exposed as a LaggedCITest. Grid node
    ``var*(max_lag+1) + (-lag)`` indexes the unrolled DAG."""

    def __init__(self, unrolled: DSeparationOracle, n_vars: int, max_lag: int) -> None:
        self._o = unrolled
        self.n_vars = n_vars
        self.max_lag = max_lag

    def _id(self, lv: LaggedVar) -> int:
        return lv.var * (self.max_lag + 1) + (-lv.lag)

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return float(self._o(self._id(x), self._id(y), [self._id(s) for s in S]))

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> CITestResult:
        return CITestResult(p_value=self.__call__(x, y, S))


def _adjacency(tspag) -> set[frozenset[int]]:  # type: ignore[no-untyped-def]
    ep = tspag.window.endpoints
    n = ep.shape[0]
    return {
        frozenset({i, j})
        for i in range(n)
        for j in range(i + 1, n)
        if ep[i, j] != EndpointMark.NO_EDGE
    }


def test_windowed_skeleton_recovered() -> None:
    # n_series=2 (X=0, Y=1), max_lag=1. Grid nodes: (0,0)=0 (0,-1)=1 (1,0)=2 (1,-1)=3.
    # Structure (within window): X_{t-1}->X_t, X_t->Y_t, X_{t-1}->Y_{t-1}.
    unrolled = DAG.from_directed_edges(4, [(1, 0), (0, 2), (1, 3)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)  # data ignored by the oracle

    tspag = lpcmci_skeleton(data, ci_test=oracle, alpha=0.5)
    # True window skeleton: X autocorrelation (0-1), X-Y contemporaneous at t (0-2) and t-1 (1-3).
    assert _adjacency(tspag) == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 3})}
    # every surviving edge is circle-circle (unoriented)
    for e in tspag.edges():
        assert e.mark_at_src == EndpointMark.CIRCLE and e.mark_at_dst == EndpointMark.CIRCLE


def test_possible_dsep_invariant_on_no_latent() -> None:
    # With no latents, Possible-D-Sep removes nothing: the FCI skeleton == the PC skeleton.
    unrolled = DAG.from_directed_edges(4, [(1, 0), (0, 2), (1, 3)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)
    with_pds = _adjacency(lpcmci_skeleton(data, ci_test=oracle, alpha=0.5, possible_dsep=True))
    without = _adjacency(lpcmci_skeleton(data, ci_test=oracle, alpha=0.5, possible_dsep=False))
    assert with_pds == without == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 3})}


def test_possible_dsep_does_extra_ci_work() -> None:
    from cbcd import InMemoryRecorder

    unrolled = DAG.from_directed_edges(4, [(1, 0), (0, 2), (1, 3)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)
    rc_with, rc_without = InMemoryRecorder(), InMemoryRecorder()
    lpcmci_skeleton(data, ci_test=oracle, alpha=0.5, possible_dsep=True, recorder=rc_with)
    lpcmci_skeleton(data, ci_test=oracle, alpha=0.5, possible_dsep=False, recorder=rc_without)
    # the PDS pass issues additional CI tests on top of the PC skeleton
    assert rc_with.metrics()["n_ci_total"] >= rc_without.metrics()["n_ci_total"]


def test_mci_cond_set() -> None:
    # P(X0_t)={X0_{t-1}}, P(X1_t)={X0_{t-1}, X1_{t-1}}. For link X0_{t-1} -> X1_t the MCI set is
    # P(X1_t)\{X0_{t-1}} plus X0_{t-1}'s parents shifted by -1 (X0_{t-2}, out of window -> dropped).
    parents = {
        LaggedVar(0, 0): frozenset({LaggedVar(0, -1)}),
        LaggedVar(1, 0): frozenset({LaggedVar(0, -1), LaggedVar(1, -1)}),
    }
    cond = _mci_cond_set(parents, LaggedVar(0, -1), LaggedVar(1, 0), max_lag=1)
    assert cond == [LaggedVar(1, -1)]


def test_mci_skeleton_recovers_lagged() -> None:
    # Lagged-only structure: X0_{t-1}->X0_t, X0_{t-1}->X1_t, X1_{t-1}->X1_t.
    # grid: (0,0)=0 (0,-1)=1 (1,0)=2 (1,-1)=3
    unrolled = DAG.from_directed_edges(4, [(1, 0), (1, 2), (3, 2)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)
    tspag = mci_skeleton(data, ci_test=oracle, alpha=0.5)
    assert _adjacency(tspag) == {frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})}


def test_lpcmci_orientation_respects_time_order() -> None:
    # Lagged-only structure: X0_{t-1}->X0_t, X0_{t-1}->X1_t, X1_{t-1}->X1_t.
    unrolled = DAG.from_directed_edges(4, [(1, 0), (1, 2), (3, 2)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)
    pag = lpcmci(data, ci_test=oracle, alpha=0.5)

    lagged = [e for e in pag.edges() if e.src.lag < e.dst.lag]  # src is the earlier endpoint
    assert len(lagged) == 3
    for e in lagged:
        assert e.dst.lag == 0  # anchored at present
        # time order: arrowhead at the present endpoint, never a present->past tail
        assert e.mark_at_dst == EndpointMark.ARROW


def test_lpcmci_records_run() -> None:
    from cbcd import InMemoryRecorder

    unrolled = DAG.from_directed_edges(4, [(1, 0), (1, 2), (3, 2)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)
    rec = InMemoryRecorder()
    lpcmci(data, ci_test=oracle, alpha=0.5, recorder=rec, run_id="lp")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "lpcmci"
    assert run["run_id"] == "lp"


def test_records_run() -> None:
    from cbcd import InMemoryRecorder

    unrolled = DAG.from_directed_edges(4, [(1, 0), (0, 2), (1, 3)])
    oracle = _LaggedOracle(DSeparationOracle(unrolled), n_vars=2, max_lag=1)
    data = LaggedDataset(np.zeros((10, 2)), max_lag=1)
    rec = InMemoryRecorder()
    lpcmci_skeleton(data, ci_test=oracle, alpha=0.5, recorder=rec, run_id="sk")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "lpcmci_skeleton"
    assert run["run_id"] == "sk"
    assert rec.metrics()["n_ci_total"] >= 1


# ==================================================================================================
# Cell-4 structural gate: LPCMCI recovers the true ts-PAG (incl. bidirected latent edges) at oracle.
# ==================================================================================================


def test_lpcmci_recovers_bidirected_latent_edge_at_oracle() -> None:
    """The structural gate. A latent series L confounds observed X0, X1 contemporaneously (each also
    autoregressive). Marginalising L, the true ts-PAG has X0_t <-> X1_t (bidirected: neither is an
    ancestor of the other, they share a hidden common cause). LPCMCI must recover exactly that."""
    ml = 1
    # observed 0,1; latent 2. lagged X0->X0, X1->X1; contemp L->X0, L->X1.
    oracle = _TSLatentOracle(
        n_obs=2, n_lat=1, max_lag=ml,
        lagged=[(0, 0, 1), (1, 1, 1)], contemp=[(2, 0), (2, 1)],
    )
    ds = LaggedDataset(np.zeros((10, 2)), max_lag=ml)
    pag = lpcmci(ds, ci_test=oracle, tau_max=ml, alpha=0.5)
    ep = pag.window.endpoints

    x0t, x1t = _grid(0, 0, ml), _grid(1, 0, ml)
    # X0_t <-> X1_t : arrowhead at BOTH ends (bidirected latent-confounded edge).
    assert ep[x0t, x1t] == EndpointMark.ARROW and ep[x1t, x0t] == EndpointMark.ARROW

    # Present-anchored edges: the bidirected contemp edge + the two autoregressive lagged edges.
    present = pag.present_edges()
    assert len(present) == 3
    lagged_edges = [e for e in present if e.src.lag < e.dst.lag]
    assert len(lagged_edges) == 2
    for e in lagged_edges:
        assert e.mark_at_dst == EndpointMark.ARROW  # arrowhead at present (time order)
        # Faithful LPCMCI resolves the past end of a true (unconfounded) autoregressive edge to a
        # tail (definite ancestor); it may remain a circle if unresolved. Never a head here.
        assert e.mark_at_src in (EndpointMark.TAIL, EndpointMark.CIRCLE)


def test_lpcmci_no_latent_matches_expected_pag_at_oracle() -> None:
    """Sanity: with no latents, LPCMCI's windowed FCI recovers the ordinary (bidirected-free) PAG.
    Chain X0_{t-1} -> X1_t -> ... here just X0_{t-1} -> X1_t and X1_{t-1} -> X1_t (autoreg)."""
    ml = 1
    oracle = _TSLatentOracle(
        n_obs=2, n_lat=0, max_lag=ml, lagged=[(0, 1, 1), (1, 1, 1)], contemp=[],
    )
    ds = LaggedDataset(np.zeros((10, 2)), max_lag=ml)
    pag = lpcmci(ds, ci_test=oracle, tau_max=ml, alpha=0.5)
    # No contemporaneous edge, and no bidirected marks anywhere.
    ep = pag.window.endpoints
    x0t, x1t = _grid(0, 0, ml), _grid(1, 0, ml)
    assert ep[x0t, x1t] == EndpointMark.NO_EDGE
    # No edge carries arrowheads at both ends (no latent confounding).
    n = ep.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            assert not (ep[a, b] == EndpointMark.ARROW and ep[b, a] == EndpointMark.ARROW)


def test_lpcmci_recovers_latent_edge_that_sufficient_methods_cannot() -> None:
    """Latent-coverage demo: the bidirected edge that pcmci / pcmci_plus structurally cannot
    represent (their TimeSeriesCPDAG has no bidirected marks) but lpcmci recovers."""
    ml = 1
    oracle = _TSLatentOracle(
        n_obs=2, n_lat=1, max_lag=ml,
        lagged=[(0, 0, 1), (1, 1, 1)], contemp=[(2, 0), (2, 1)],
    )
    ds = LaggedDataset(np.zeros((10, 2)), max_lag=ml)

    lp = lpcmci(ds, ci_test=oracle, tau_max=ml, alpha=0.5)
    x0t, x1t = _grid(0, 0, ml), _grid(1, 0, ml)
    # lpcmci: bidirected latent edge present.
    assert lp.window.endpoints[x0t, x1t] == EndpointMark.ARROW
    assert lp.window.endpoints[x1t, x0t] == EndpointMark.ARROW

    # pcmci / pcmci_plus: their output type (TimeSeriesCPDAG) admits only TAIL/ARROW directed or
    # undirected marks -- no bidirected edge is representable. Both still run and see the adjacency,
    # but cannot express the confounding.
    cp = pcmci_plus(ds, ci_test=oracle, tau_max=ml, alpha=0.5)
    contemp = cp.contemporaneous_edges()
    # pcmci_plus sees the X0-X1 contemporaneous adjacency but represents it as directed/undirected.
    assert len(contemp) == 1
    for e in contemp:
        assert not (e.mark_at_src == EndpointMark.ARROW and e.mark_at_dst == EndpointMark.ARROW)
    # pcmci (lagged only) cannot represent the contemporaneous edge at all.
    assert pcmci(ds, ci_test=oracle, alpha=0.5).contemporaneous_edges() == ()


def test_lpcmci_signature_and_validation() -> None:
    from cbcd.exceptions import CBCDInputError

    ml = 1
    oracle = _TSLatentOracle(2, 0, ml, [(0, 1, 1)], [])
    ds = LaggedDataset(np.zeros((10, 2)), max_lag=ml)
    # k accepted; tau_max must match dataset.
    lpcmci(ds, ci_test=oracle, tau_max=ml, alpha=0.5, k=2)
    with __import__("pytest").raises(CBCDInputError):
        lpcmci(ds, ci_test=oracle, tau_max=2, alpha=0.5)


# ==================================================================================================
# RECALL GATE: faithful lpcmci recovers MORE true adjacencies than a windowed-FCI baseline on a
# strongly-autocorrelated latent DGP, and recovers the bidirected edge pcmci_plus cannot represent.
# ==================================================================================================


def _sim_autocorr_latent(T: int, seed: int, phi: float = 0.9, noise: float = 0.3) -> np.ndarray:
    """4 observed vars, strong autocorrelation ``phi``; a latent confounds X0,X1 contemporaneously;
    X0_{t-1}->X2_t and X1_{t-1}->X3_t cross-links."""
    rng = np.random.default_rng(seed)
    burn = 300
    X = np.zeros((T + burn, 5))
    L = 4
    for t in range(1, T + burn):
        X[t, L] = rng.normal(scale=noise)
        X[t, 0] = phi * X[t - 1, 0] + 0.6 * X[t, L] + rng.normal(scale=noise)
        X[t, 1] = phi * X[t - 1, 1] + 0.6 * X[t, L] + rng.normal(scale=noise)
        X[t, 2] = phi * X[t - 1, 2] + 0.5 * X[t - 1, 0] + rng.normal(scale=noise)
        X[t, 3] = phi * X[t - 1, 3] + 0.5 * X[t - 1, 1] + rng.normal(scale=noise)
    return X[burn:, :4]


def _windowed_fci_adjacency(X: np.ndarray, ml: int, alpha: float) -> set[frozenset[int]]:
    """The old windowed-FCI baseline: FAS -> colliders -> Possible-D-Sep -> time order -> Zhang rules
    over the lagged grid. Returns its grid adjacency."""
    from cbcd import fci
    from cbcd.background import BackgroundKnowledge
    from cbcd.citest.protocol import CITestResult
    from cbcd.timeseries.citest import CachedLaggedCITest, ParCorr

    n = X.shape[1]
    gn = n * (ml + 1)

    class _GridCI:
        def __init__(self, inner):
            self._ci = inner
            self.max_lag = ml
            self.n_vars = gn

        def _d(self, node):
            var, rem = divmod(int(node), ml + 1)
            return LaggedVar(var, -rem)

        def __call__(self, x, y, S):
            return float(self._ci(self._d(x), self._d(y), [self._d(z) for z in S]))

        def details(self, x, y, S):
            return CITestResult(p_value=self.__call__(x, y, S))

        def is_cached(self, x, y, S):
            f = getattr(self._ci, "is_cached", None)
            return bool(f(self._d(x), self._d(y), [self._d(z) for z in S])) if f else False

    ds = LaggedDataset(X, max_lag=ml)
    grid = _GridCI(CachedLaggedCITest(ParCorr(ds)))

    def dec(x):
        v, r = divmod(x, ml + 1)
        return -r

    forbid = {(b, a) for a in range(gn) for b in range(gn) if a != b and dec(a) < dec(b)}
    bg = BackgroundKnowledge(forbidden_directed=frozenset(forbid))
    pag = fci(np.zeros((5, gn)), ci_test=grid, alpha=alpha, background=bg)
    return {
        frozenset({a, b})
        for a in range(gn)
        for b in range(a + 1, gn)
        if pag.endpoints[a, b] != 0
    }


def _true_adjacency(ml: int) -> set[frozenset[int]]:
    t: set[frozenset[int]] = set()
    for v in range(4):
        t.add(frozenset({_grid(v, -1, ml), _grid(v, 0, ml)}))  # autoregressive
    t.add(frozenset({_grid(0, 0, ml), _grid(1, 0, ml)}))  # latent-confounded contemporaneous
    t.add(frozenset({_grid(0, -1, ml), _grid(2, 0, ml)}))
    t.add(frozenset({_grid(1, -1, ml), _grid(3, 0, ml)}))
    return t


def _lpcmci_adjacency(pag) -> set[frozenset[int]]:  # type: ignore[no-untyped-def]
    ep = pag.window.endpoints
    gn = ep.shape[0]
    return {frozenset({a, b}) for a in range(gn) for b in range(a + 1, gn) if ep[a, b] != 0}


def test_lpcmci_higher_recall_than_windowed_fci_on_autocorrelated_latents() -> None:
    """The reason LPCMCI exists: on strongly-autocorrelated latent data its MCI conditioning
    (``S ∪ S_def``) recovers more true adjacencies than a windowed-FCI baseline. Aggregated over
    seeds, faithful lpcmci's true-adjacency recall strictly exceeds the windowed-FCI baseline's."""
    from cbcd import pcmci_plus

    ml = 1
    true = _true_adjacency(ml)
    lp_recall = fci_recall = 0
    n_seeds = 8
    bidirected_found = False
    for seed in range(n_seeds):
        X = _sim_autocorr_latent(400, seed, phi=0.9)
        ds = LaggedDataset(X, max_lag=ml)
        lp = lpcmci(ds, ci_test="parcorr", tau_max=ml, alpha=0.05, k=4)
        lp_recall += len(_lpcmci_adjacency(lp) & true)
        fci_recall += len(_windowed_fci_adjacency(X, ml, 0.05) & true)
        # the confounded contemporaneous pair recovered as a bidirected edge
        ep = lp.window.endpoints
        x0t, x1t = _grid(0, 0, ml), _grid(1, 0, ml)
        if ep[x0t, x1t] == EndpointMark.ARROW and ep[x1t, x0t] == EndpointMark.ARROW:
            bidirected_found = True

    # (1) Strictly higher true-adjacency recall than the windowed-FCI stand-in.
    assert lp_recall > fci_recall, f"lpcmci recall {lp_recall} !> windowed-FCI {fci_recall}"
    # (2) lpcmci recovered the latent edge as bidirected on at least some seeds.
    assert bidirected_found
    # (3) pcmci_plus (causally sufficient) cannot represent that bidirected edge at all.
    X = _sim_autocorr_latent(400, 0, phi=0.9)
    ds = LaggedDataset(X, max_lag=ml)
    cp = pcmci_plus(ds, ci_test="parcorr", tau_max=ml, alpha=0.05)
    for e in cp.contemporaneous_edges():
        assert not (e.mark_at_src == EndpointMark.ARROW and e.mark_at_dst == EndpointMark.ARROW)
