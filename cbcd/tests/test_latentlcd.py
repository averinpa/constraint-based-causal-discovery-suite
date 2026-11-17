"""LatentLCD (Ling et al., 2024) at a d-separation oracle.

The claim that distinguishes this method is EL(T), the neighbours joined to the target by a
bidirected edge. A method assuming causal sufficiency returns them and calls them causes, so the
tests here are about whether EL(T) is found and whether it is found correctly.

Theorem 3 is also checked in its own right: put sufficiency back and EL(T) is empty, so the
object must collapse to the parents and children.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import InMemoryRecorder, fci, latentlcd
from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark as M
from tests.oracle_pag import DSeparationOracleProjected
from tests.test_locals import _rand_latent_case

ARROW, TAIL, CIRCLE, NO_EDGE = int(M.ARROW), int(M.TAIL), int(M.CIRCLE), int(M.NO_EDGE)


def _groups(ep: np.ndarray, t: int) -> tuple[set[int], set[int], set[int]]:
    """LocalMAG(T) read off a PAG — (parents, children, EL)."""
    n = ep.shape[0]
    pa, ch, el = set(), set(), set()
    for w in range(n):
        if w == t or ep[t, w] == NO_EDGE:
            continue
        near, far = int(ep[w, t]), int(ep[t, w])
        if near == ARROW and far == ARROW:
            el.add(w)
        elif near == ARROW and far == TAIL:
            pa.add(w)
        elif near == TAIL and far == ARROW:
            ch.add(w)
    return pa, ch, el


# --------------------------------------------------------------- the Cancer example
# Cancer 0, Smoking 1, Immune 2, Coughing 3, Attention 4, Stress 5; Genotype is last and latent.
_BASE = [(1, 0), (2, 0), (0, 3)]


def _cancer(p: int, n_obs: int, extra: list[tuple[int, int]]):
    return DSeparationOracleProjected(DAG.from_directed_edges(p, _BASE + extra), n_obs)


def test_paper_figure_one_cannot_identify_el() -> None:
    """The paper's motivating figure is not a case the method can solve, and that is a limit of
    the data rather than of the method. Figure 1 draws the MAG, where Cancer <-> Attention, but
    the PAG leaves the Attention end a circle, because Attention has no collider of its own to
    make the arrowhead invariant. Global FCI leaves the same circle, which is the point: no
    observational method recovers EL(T) there."""
    oracle = _cancer(6, 5, [(5, 0), (5, 4)])  # Genotype -> Cancer, Genotype -> Attention
    ep = latentlcd(np.zeros((5, 5)), 0, ci_test=oracle, alpha=0.5).endpoints
    glob = fci(np.zeros((5, 5)), ci_test=oracle, alpha=0.5).endpoints
    assert ep[4, 0] == ARROW, "the arrowhead at Cancer should still be found"
    assert ep[0, 4] == CIRCLE, "EL cannot be identified here, so the far end must stay a circle"
    assert glob[0, 4] == CIRCLE, "global fci cannot identify it either"
    pa, _, el = _groups(ep, 0)
    assert not el and 4 not in pa, "Attention must be neither EL nor a committed parent"


def test_el_is_found_when_the_neighbour_has_its_own_collider() -> None:
    """Add one edge, Stress -> Attention, and Attention gains an unshielded collider of its own.
    The second arrowhead becomes invariant, FindBI fires, and the edge comes back bidirected —
    Attention is now identified as an effect of a latent cause rather than a cause of Cancer."""
    oracle = _cancer(7, 6, [(6, 0), (6, 4), (5, 4)])
    ep = latentlcd(np.zeros((5, 6)), 0, ci_test=oracle, alpha=0.5).endpoints
    glob = fci(np.zeros((5, 6)), ci_test=oracle, alpha=0.5).endpoints
    pa, _, el = _groups(ep, 0)
    assert 4 in el, f"Attention not in EL; marks are {int(ep[4, 0])}-{int(ep[0, 4])}"
    assert 4 not in pa
    assert ep[4, 0] == glob[4, 0] and ep[0, 4] == glob[0, 4]


# ------------------------------------------------------------------------- Theorem 3
def test_el_is_empty_under_causal_sufficiency() -> None:
    """Theorem 3 — with no latent variable the third group vanishes and LocalMAG collapses to
    the parents and children, which is what a sufficiency-assuming method returns."""
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(10):
        p = int(rng.integers(5, 8))
        order = rng.permutation(p)
        eps = np.zeros((p, p), np.int8)
        for _ in range(p + 2):
            a, b = (int(v) for v in rng.integers(0, p, 2))
            if a == b:
                continue
            i, j = (a, b) if np.where(order == a)[0][0] < np.where(order == b)[0][0] else (b, a)
            eps[i, j], eps[j, i] = M.ARROW, M.TAIL
        oracle = DSeparationOracleProjected(DAG(p, eps), p)  # nothing hidden
        for t in range(p):
            ep = latentlcd(np.zeros((5, p)), t, ci_test=oracle, alpha=0.5).endpoints
            _, _, el = _groups(ep, t)
            assert not el, f"target {t}: EL={sorted(el)} with no latent variable present"
            checked += 1
    assert checked > 40


# ------------------------------------------------ soundness against global discovery
def test_committed_marks_at_target_match_global_fci() -> None:
    """No committed mark at the target contradicts global FCI. LatentLCD is the least complete
    of the latent local methods — it goes one step out and stops — so circles where FCI commits
    are expected; a committed mark that differs would be unsound."""
    rng = np.random.default_rng(11)
    wrong = committed = 0
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            ep = latentlcd(data, t, ci_test=oracle, alpha=0.5).endpoints
            for w in range(n_obs):
                if w == t or ep[t, w] == NO_EDGE:
                    continue
                for i, j in ((t, w), (w, t)):
                    if ep[i, j] == CIRCLE:
                        continue
                    committed += 1
                    wrong += int(ep[i, j] != int(glob[i, j]))
    assert committed > 0, "nothing was committed, the test proves nothing"
    assert wrong == 0, f"{wrong} of {committed} committed marks at the target disagree with fci"


# ------------------------------------------------------------------------- plumbing
def test_records_and_validates() -> None:
    oracle = _cancer(7, 6, [(6, 0), (6, 4), (5, 4)])
    rec = InMemoryRecorder()
    latentlcd(np.zeros((5, 6)), 0, ci_test=oracle, alpha=0.5, recorder=rec, run_id="X")
    frames = rec.to_frames()
    run = frames["runs"].iloc[0]
    assert run["algorithm"] == "latentlcd" and run["run_id"] == "X"
    phases = set(frames["ci_tests"]["phase"])
    assert "latentlcd_collider" in phases, "no collider test was logged"
    assert len(frames["regions"]) > 0
    with pytest.raises(CBCDInputError):
        latentlcd(np.zeros((5, 6)), 9, ci_test=oracle)
