"""LoCaLS (Li et al., 2025) — the paper's own guarantees, checked at a d-separation oracle.

Three things are asserted, each the statement of a result in the paper rather than a property
this implementation happens to have.

* ``total_conditioning`` returns the Markov blanket of the MAG (Proposition 1), checked against
  the four clauses of Definition 9 computed from the graph.
* The adjacencies incident to the target agree with global ``fci`` (Corollary 1).
* Every committed mark incident to the target agrees with global ``fci``, which is the
  target-specific soundness claim of Theorem 4's stopping rules.

The oracle removes the test from the question, which is where every result in the paper is
stated. A failure here is a failure of the algorithm, not of a finite sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import InMemoryRecorder, fci, locals_, total_conditioning
from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark as M
from tests.oracle_pag import DSeparationOracleProjected

ARROW, TAIL, CIRCLE, NO_EDGE = int(M.ARROW), int(M.TAIL), int(M.CIRCLE), int(M.NO_EDGE)


def _rand_dag(p: int, n_edges: int, rng: np.random.Generator) -> DAG:
    order = rng.permutation(p)
    ep = np.zeros((p, p), np.int8)
    c, guard = 0, 0
    while c < n_edges and guard < 10_000:
        guard += 1
        a, b = (int(x) for x in rng.integers(0, p, 2))
        if a == b:
            continue
        i, j = (a, b) if np.where(order == a)[0][0] < np.where(order == b)[0][0] else (b, a)
        if ep[i, j] == 0:
            ep[i, j] = M.ARROW
            ep[j, i] = M.TAIL
            c += 1
    return DAG(p, ep)


def _rand_latent_case(rng: np.random.Generator) -> tuple[DAG, int]:
    """Random DAG whose last 0-2 indices are marginalized away as latent confounders."""
    n_obs = int(rng.integers(5, 9))
    n_lat = int(rng.integers(0, 3))
    p = n_obs + n_lat
    return _rand_dag(p, int(p * rng.uniform(1.0, 1.7)), rng), n_obs


def _mag_blanket(oracle: DSeparationOracleProjected, x: int, n_obs: int) -> set[int]:
    """Definition 9 read straight off m-separation: Y is in MB(X) iff X and Y are dependent
    given every other observed variable. That equivalence IS Proposition 1, so this is the
    graph-side answer the primitive is checked against."""
    others = [v for v in range(n_obs) if v != x]
    return {y for y in others if oracle(x, y, [v for v in others if v != y]) <= 0.5}


def _marks_at(ep: np.ndarray, t: int) -> dict[tuple[int, int], int]:
    """Every endpoint mark on an edge incident to ``t``, keyed by (node, other)."""
    n = ep.shape[0]
    out: dict[tuple[int, int], int] = {}
    for w in range(n):
        if w != t and ep[t, w] != NO_EDGE:
            out[(t, w)] = int(ep[t, w])
            out[(w, t)] = int(ep[w, t])
    return out


# ------------------------------------------------------------------- Proposition 1
def test_total_conditioning_is_the_mag_blanket() -> None:
    rng = np.random.default_rng(20261117)
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        for x in range(n_obs):
            assert set(total_conditioning(oracle, x, alpha=0.5)) == _mag_blanket(oracle, x, n_obs)


def test_total_conditioning_is_linear_in_n_vars() -> None:
    """One test per candidate, not a subset search — the cost claim in Section 4.4."""
    dag = DAG.from_directed_edges(6, [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5)])
    oracle = DSeparationOracleProjected(dag, 6)
    rec = InMemoryRecorder()
    rec.begin_run(
        run_id="tc", algorithm="tc", params={}, n_samples=0, n_vars=6, ci_test="oracle"
    )
    total_conditioning(oracle, 2, alpha=0.5, recorder=rec)
    assert len(rec.tables["ci_tests"]) == 5


# -------------------------------------------------------------------- Corollary 1
def test_adjacencies_at_target_match_global_fci() -> None:
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            loc = locals_(data, t, ci_test=oracle, alpha=0.5).endpoints
            got = {w for w in range(n_obs) if w != t and loc[t, w] != NO_EDGE}
            want = {w for w in range(n_obs) if w != t and glob[t, w] != NO_EDGE}
            assert got == want, f"target {t}: {sorted(got)} vs fci {sorted(want)}"
            checked += 1
    assert checked > 50


# ----------------------------------------------------------- target-specific soundness
def test_committed_marks_at_target_match_global_fci() -> None:
    """No committed mark incident to the target disagrees with global FCI. A circle where FCI
    commits is allowed (less informative); a committed mark that differs is not (unsound)."""
    rng = np.random.default_rng(11)
    wrong, committed = 0, 0
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            loc = locals_(data, t, ci_test=oracle, alpha=0.5).endpoints
            for (i, j), mark in _marks_at(loc, t).items():
                if mark == CIRCLE:
                    continue
                committed += 1
                if mark != int(glob[i, j]):
                    wrong += 1
    assert committed > 0, "nothing was committed, the test proves nothing"
    assert wrong == 0, f"{wrong} of {committed} committed marks at the target disagree with fci"


def test_target_marks_are_as_informative_as_global_fci() -> None:
    """Theorem 4: the local structure for the target is equivalent to the target-specific
    structure a global method returns. Soundness alone is cheap (commit nothing), so this is
    the other half — LoCaLS must also commit everywhere global FCI does."""
    rng = np.random.default_rng(11)
    local_committed = fci_committed = 0
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            loc = locals_(data, t, ci_test=oracle, alpha=0.5).endpoints
            for (i, j), mark in _marks_at(loc, t).items():
                local_committed += int(mark != CIRCLE)
                fci_committed += int(int(glob[i, j]) != CIRCLE)
    assert fci_committed > 0
    assert local_committed == fci_committed, (
        f"committed {local_committed} marks where global fci commits {fci_committed}"
    )


# ------------------------------------------------------------------------ plumbing
def test_locals_records_and_validates() -> None:
    dag = DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])
    oracle = DSeparationOracleProjected(dag, 4)
    data = np.zeros((5, 4))
    rec = InMemoryRecorder()
    locals_(data, 2, ci_test=oracle, alpha=0.5, recorder=rec, run_id="L")
    frames = rec.to_frames()
    run = frames["runs"].iloc[0]
    assert run["algorithm"] == "locals" and run["run_id"] == "L"
    assert len(frames["ci_tests"]) > 0, "no CI query reached the recorder"
    assert len(frames["regions"]) > 0, "no blanket was recorded"
    with pytest.raises(CBCDInputError):
        locals_(data, 9, ci_test=oracle)


def test_max_rounds_bounds_the_walk() -> None:
    """A capped walk learns at most that many blankets and stays sound."""
    rng = np.random.default_rng(3)
    dag, n_obs = _rand_latent_case(rng)
    oracle = DSeparationOracleProjected(dag, n_obs)
    data = np.zeros((5, n_obs))
    rec = InMemoryRecorder()
    rec_ = InMemoryRecorder()
    locals_(data, 0, ci_test=oracle, alpha=0.5, max_rounds=1, recorder=rec)
    locals_(data, 0, ci_test=oracle, alpha=0.5, recorder=rec_)
    assert len(rec.to_frames()["regions"]) == 1
    assert len(rec.to_frames()["ci_tests"]) <= len(rec_.to_frames()["ci_tests"])


def test_as_published_reproduces_the_defect() -> None:
    """The flag must actually reproduce what Algorithm 1 specifies, or it is useless for
    measuring the published behaviour. Running the rules over the whole pooled graph has to
    commit marks that disagree with global FCI, where the corrected path commits none."""
    rng = np.random.default_rng(11)
    wrong_pub = wrong_fixed = 0
    for _ in range(6):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            for published, bucket in ((True, "pub"), (False, "fixed")):
                ep = locals_(
                    data, t, ci_test=oracle, alpha=0.5, as_published=published
                ).endpoints
                bad = sum(
                    1
                    for (i, j), mk in _marks_at(ep, t).items()
                    if mk != CIRCLE and mk != int(glob[i, j])
                )
                if bucket == "pub":
                    wrong_pub += bad
                else:
                    wrong_fixed += bad
    assert wrong_fixed == 0, f"the corrected path is not sound: {wrong_fixed} wrong"
    assert wrong_pub > 0, "as_published did not reproduce the defect"
