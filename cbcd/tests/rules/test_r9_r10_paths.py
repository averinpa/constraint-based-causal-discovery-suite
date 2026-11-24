"""R9 and R10 search over every candidate first vertex, with uncovered pruning during search.

Before 2026-11-24 R9 took whichever uncovered potentially directed path the DFS met first and
gave up when its second vertex was adjacent to gamma, and the path search filtered uncovered
paths only at the end. The first missed tails, the second took minutes per call on 40-node
graphs inside MMB-by-MMB.
"""

from __future__ import annotations

import numpy as np

from cbcd import fci
from cbcd.graph.marks import EndpointMark as M
from cbcd.rules import _apply_zhang_r9
from tests.oracle_pag import DSeparationOracleProjected
from tests.test_locals import _rand_latent_case
from tests.test_mmb_by_mmb import _true_mag

ARR, TAIL, CIRC = int(M.ARROW), int(M.TAIL), int(M.CIRCLE)


def _r9_case(bad_first: bool) -> tuple[np.ndarray, int, int]:
    """alpha o-> gamma with two uncovered p.d. paths from alpha to gamma.

    Path A = alpha, a1, a2, a3, gamma with a1 adjacent to gamma (does not license R9).
    Path B = alpha, b1, b2, gamma with b1 not adjacent to gamma (licenses R9).
    ``bad_first`` swaps vertex labels so either path can be the one a DFS meets first.
    """
    names = ["alpha", "gamma", "a1", "a2", "a3", "b1", "b2"]
    if bad_first:
        names = ["alpha", "gamma", "b1", "b2", "a1", "a2", "a3"]
    ix = {v: i for i, v in enumerate(names)}
    ep = np.zeros((7, 7), dtype=np.int8)

    def edge(u: str, v: str, at_u: int, at_v: int) -> None:
        ep[ix[u], ix[v]], ep[ix[v], ix[u]] = at_v, at_u

    edge("alpha", "gamma", CIRC, ARR)
    for u, v in [("alpha", "a1"), ("a1", "a2"), ("a2", "a3"), ("a3", "gamma"), ("a1", "gamma"),
                 ("alpha", "b1"), ("b1", "b2"), ("b2", "gamma")]:
        edge(u, v, TAIL, ARR)
    return ep, ix["alpha"], ix["gamma"]


def test_r9_finds_qualifying_path_in_either_order() -> None:
    for bad_first in (False, True):
        ep, a, g = _r9_case(bad_first)
        assert _apply_zhang_r9(ep, ep.shape[0], None)
        assert ep[g, a] == TAIL, f"R9 missed alpha -> gamma (bad_first={bad_first})"


def test_r9_does_not_fire_without_qualifying_path() -> None:
    ep, a, g = _r9_case(False)
    b1 = 5
    ep[b1, g] = ep[g, b1] = ARR  # b1 now adjacent to gamma, so neither path qualifies
    _apply_zhang_r9(ep, ep.shape[0], None)
    assert ep[g, a] == CIRC


def test_fci_marks_agree_with_mag_on_random_latent_graphs() -> None:
    """Soundness of the whole rule set at an oracle, including graphs where the repaired R9
    and R10 commit tails the old search missed (seed 2026, graphs 47 and 153)."""
    rng = np.random.default_rng(2026)
    for _ in range(200):
        dag, n_obs = _rand_latent_case(rng)
        ep = np.asarray(fci(np.zeros((5, n_obs)), ci_test=DSeparationOracleProjected(dag, n_obs),
                            alpha=0.5).endpoints)
        mag = _true_mag(dag, n_obs)
        for (i, j), (mi, mj) in mag.items():
            for u, v, m in ((i, j, mj), (j, i, mi)):  # mark at v on edge u-v
                got = int(ep[u, v])
                if got != CIRC:
                    assert got == {"t": TAIL, "a": ARR}[m], (i, j)
