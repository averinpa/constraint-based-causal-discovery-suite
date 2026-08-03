"""Randomized oracle soundness + completeness battery for LPCMCI (committed Stage-1 stress).

This locks the *soundness* of the LPCMCI orientation into the suite rather than trusting a
throwaway script. Over a fixed-seed ensemble of random latent SVAR structures we assert, under the
m-separation oracle, that **every committed endpoint mark is ancestrally consistent with the true
unrolled DAG**: a TAIL at ``X`` on edge ``{X, Y}`` requires ``X`` to be an ancestor of ``Y``; a
HEAD (arrowhead) at ``X`` requires ``X`` NOT to be an ancestor of ``Y``. These are exactly the
invariant PAG semantics, so the check needs no PAG reference. Zero violations == the output commits
no wrong orientation.

Completeness is characterized (not gated hard) against a sound windowed-FCI reference (FAS +
Possible-D-Sep + time order + Zhang R1-R10 minus the windowing-unsound R9 -- see
``cbcd.timeseries.lpcmci_algo._WINDOWED_SOUND_RULES``): we report the exact-PAG-recovery rate and
guard it with a loose lower bound so a real regression trips the test.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np

from cbcd import lpcmci
from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITestResult
from cbcd.collider import SepsetOrienter
from cbcd.graph.pag import PartialPAG
from cbcd.refinement import PossibleDSepRefinement
from cbcd.rules import _FCI_ALL_RULES, FCIRules
from cbcd.skeleton import FAS, Skeleton
from cbcd.timeseries.lagged import LaggedDataset, LaggedVar
from cbcd.timeseries.lpcmci_pag import decode_grid

TAIL, HEAD, CIRCLE, NO = 1, 2, 3, 0
_SOUND_RULES = _FCI_ALL_RULES - {"R9"}


def _rand_structure(
    rng: np.random.Generator, n_obs: int, n_lat: int, tau_max: int, density: float
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], int]:
    """Random acyclic-in-time DAG with latents over the window: contemporaneous edges follow a
    random topological order (acyclic); lagged edges always point forward in time."""
    n = n_obs + n_lat
    order = list(rng.permutation(n))
    pos = {v: i for i, v in enumerate(order)}
    contemp = [
        (a, b)
        for a in range(n)
        for b in range(n)
        if a != b and pos[a] < pos[b] and rng.random() < density
    ]
    lagged = [
        (i, j, tau)
        for i in range(n)
        for j in range(n)
        for tau in range(1, tau_max + 1)
        if rng.random() < density
    ]
    return lagged, contemp, n


class _Oracle:
    """m-separation oracle over an unrolled latent SVAR; only the first ``n_obs`` series are queried."""

    def __init__(
        self,
        n_obs: int,
        n_total: int,
        tau_max: int,
        lagged: list[tuple[int, int, int]],
        contemp: list[tuple[int, int]],
    ) -> None:
        self.n_vars = n_obs
        self.max_lag = tau_max
        self._th = max(6 * tau_max, 12)
        self._tref = max(3 * tau_max, 6)
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

    def __call__(self, x: LaggedVar, y: LaggedVar, s: Sequence[LaggedVar]) -> float:
        return self.details(x, y, s).p_value

    def details(self, x: LaggedVar, y: LaggedVar, s: Sequence[LaggedVar]) -> CITestResult:
        sep = nx.is_d_separator(
            self._g, {self._node(x)}, {self._node(y)}, {self._node(z) for z in s}
        )
        return CITestResult(p_value=1.0 if sep else 0.0)

    def is_ancestor(self, a_grid: int, b_grid: int) -> bool:
        av, al = decode_grid(a_grid, self.max_lag)
        bv, bl = decode_grid(b_grid, self.max_lag)
        an = (av, self._tref + al)
        bn = (bv, self._tref + bl)
        if an == bn:
            return True
        return bn in nx.descendants(self._g, an)


class _GridCI:
    """The oracle re-exposed on flat grid-node ids for the windowed-FCI reference."""

    def __init__(self, oracle: _Oracle, n_series: int, max_lag: int) -> None:
        self._o = oracle
        self.max_lag = max_lag
        self.n_vars = n_series * (max_lag + 1)

    def _d(self, node: int) -> LaggedVar:
        v, r = divmod(int(node), self.max_lag + 1)
        return LaggedVar(v, -r)

    def __call__(self, x: int, y: int, s: Sequence[int]) -> float:
        return float(self._o(self._d(x), self._d(y), [self._d(z) for z in s]))

    def details(self, x: int, y: int, s: Sequence[int]) -> CITestResult:
        return CITestResult(p_value=self.__call__(x, y, s))


def _present(a: int, b: int, ml: int) -> bool:
    return a % (ml + 1) == 0 or b % (ml + 1) == 0


def _sound_reference(oracle: _Oracle, n_series: int, ml: int) -> np.ndarray:
    """Sound + complete windowed PAG at the oracle: FAS + Possible-D-Sep + time order + Zhang
    R1-R10 minus the windowing-unsound R9 (matches the orientation LPCMCI uses)."""
    grid = _GridCI(oracle, n_series, ml)
    gn = n_series * (ml + 1)

    def dec(x: int) -> int:
        return -(x % (ml + 1))

    forbid = frozenset(
        (b, a) for a in range(gn) for b in range(gn) if a != b and dec(a) < dec(b)
    )
    bg = BackgroundKnowledge(forbidden_directed=forbid)
    skel = FAS()(grid, alpha=0.5, background=bg)
    partial = SepsetOrienter()(skel, grid, alpha=0.5, background=bg).apply_to_pag(skel)
    partial = PossibleDSepRefinement()(partial, grid, alpha=0.5)
    radj = (partial.endpoints != 0).astype(bool)
    rskel = Skeleton(gn, radj, dict(partial.sepsets) if partial.sepsets else {}, None)
    partial = SepsetOrienter()(rskel, grid, alpha=0.5, background=bg).apply_to_pag(rskel)
    ep = partial.endpoints.copy()
    for a in range(gn):
        for b in range(gn):
            if a != b and ep[a, b] != 0 and dec(a) < dec(b):
                ep[a, b] = 2  # arrowhead at the later endpoint (time order)
    partial = PartialPAG(gn, ep, sepsets=partial.sepsets)
    return FCIRules(rules=_SOUND_RULES)(partial, background=bg).endpoints


def _draw(rng: np.random.Generator) -> tuple:
    n_obs = int(rng.integers(2, 5))
    n_lat = int(rng.integers(0, 3))
    tau_max = int(rng.integers(1, 3))
    density = float(rng.uniform(0.25, 0.6))
    lagged, contemp, n_total = _rand_structure(rng, n_obs, n_lat, tau_max, density)
    return n_obs, n_lat, tau_max, lagged, contemp, n_total


def test_randomized_oracle_soundness_zero_violations() -> None:
    """Over 100 random latent SVAR structures the oracle LPCMCI commits ZERO wrong orientations:
    every present-anchored TAIL/HEAD is ancestrally consistent with the true unrolled DAG."""
    rng = np.random.default_rng(20260716)
    n_struct = 100
    total_committed = 0
    total_viol = 0
    offenders: list[tuple[int, int]] = []
    for s in range(n_struct):
        n_obs, n_lat, tau_max, lagged, contemp, n_total = _draw(rng)
        oracle = _Oracle(n_obs, n_total, tau_max, lagged, contemp)
        ds = LaggedDataset(np.zeros((30, n_obs)), max_lag=tau_max)
        out = lpcmci(ds, ci_test=oracle, tau_max=tau_max, alpha=0.5, k=4)
        ep = out.window.endpoints
        gn = n_obs * (tau_max + 1)
        viol = 0
        for a in range(gn):
            for b in range(gn):
                if a == b or ep[a, b] == NO or not _present(a, b, tau_max):
                    continue
                m = int(ep[a, b])  # mark at b on edge {a, b}
                if m == TAIL:
                    total_committed += 1
                    if not oracle.is_ancestor(b, a):  # tail at b => b ancestor of a
                        viol += 1
                elif m == HEAD:
                    total_committed += 1
                    if oracle.is_ancestor(b, a):  # head at b => b NOT ancestor of a
                        viol += 1
        if viol:
            offenders.append((s, viol))
        total_viol += viol
    assert total_committed > 500, f"battery too weak: only {total_committed} committed marks"
    assert total_viol == 0, (
        f"{total_viol} wrong-orientation violations across {total_committed} committed endpoints; "
        f"offending structures: {offenders[:8]}"
    )


def test_oracle_completeness_characterization() -> None:
    """Characterize (and loosely gate) exact windowed-PAG recovery vs the sound FCI reference."""
    rng = np.random.default_rng(20260716)
    n_struct = 40
    exact = 0
    pair_match = 0
    pair_total = 0
    for _ in range(n_struct):
        n_obs, n_lat, tau_max, lagged, contemp, n_total = _draw(rng)
        oracle = _Oracle(n_obs, n_total, tau_max, lagged, contemp)
        ds = LaggedDataset(np.zeros((30, n_obs)), max_lag=tau_max)
        ep = lpcmci(ds, ci_test=oracle, tau_max=tau_max, alpha=0.5, k=4).window.endpoints
        ref = _sound_reference(oracle, n_obs, tau_max)
        gn = n_obs * (tau_max + 1)
        match = True
        for a in range(gn):
            for b in range(a + 1, gn):
                if not _present(a, b, tau_max):
                    continue
                pair_total += 1
                if ep[a, b] == ref[a, b] and ep[b, a] == ref[b, a]:
                    pair_match += 1
                else:
                    match = False
        if match:
            exact += 1
    pair_rate = pair_match / max(pair_total, 1)
    # Characterization guard: the LPCMCI orientation reproduces the sound windowed FCI reference on
    # the large majority of endpoint pairs. Loose bounds so a real regression trips the test.
    assert pair_rate > 0.85, f"endpoint-pair match rate regressed to {pair_rate:.3f}"
    assert exact >= n_struct // 2, f"exact-PAG recovery regressed to {exact}/{n_struct}"
