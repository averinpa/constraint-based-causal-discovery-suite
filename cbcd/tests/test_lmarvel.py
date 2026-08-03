"""L-MARVEL structural parity with FCI + CI-efficiency regression (latents, no selection bias).

L-MARVEL recovers the skeleton + separating-set store by recursive Markov-boundary elimination
(Theorem 2 removability) and orients it with the *same* FCI rules ``fci`` uses, so under the
m-separation oracle it must return the identical PAG to ``cbcd.fci`` -- on every PAG fixture and on
random DAGs-with-latents. That is the sound+complete gate (Theorem 3). On the same runs its recorded
CI-test count must be no larger than FCI's, and strictly lower in aggregate on sparse graphs (the
reason to add it). Background knowledge is threaded exactly as ``fci`` threads it, so
``lmarvel(background=bg) == fci(background=bg)`` too.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import InMemoryRecorder, fci, lmarvel
from cbcd.background import BackgroundKnowledge
from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark as M
from tests.fixtures_pag import ALL_PAG_FIXTURES
from tests.oracle_pag import DSeparationOracleProjected


def _shd(a: np.ndarray, b: np.ndarray) -> int:
    """PAG endpoint mismatches (every endpoint counted, matching the fci oracle test)."""
    return int(np.sum(a != b))


def _rand_dag(p: int, n_edges: int, rng: np.random.Generator) -> DAG:
    """Random DAG on ``p`` nodes, acyclic by a random topological order."""
    order = rng.permutation(p)
    ep = np.zeros((p, p), np.int8)
    c = 0
    tries = 0
    while c < n_edges and tries < n_edges * 50:
        tries += 1
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
    """A random DAG-with-latents: the last ``n_latent`` (0-2) indices are hidden confounders."""
    n_obs = int(rng.integers(4, 9))
    n_lat = int(rng.integers(0, 3))
    p = n_obs + n_lat
    dag = _rand_dag(p, int(p * rng.uniform(1.0, 1.6)), rng)
    return dag, n_obs


# --------------------------------------------------------------------------------------------------
# Gate 1 — structural parity with FCI
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_PAG_FIXTURES))
def test_lmarvel_matches_fci_on_pag_fixtures(name: str) -> None:
    """On every DAG-with-latent fixture, L-MARVEL recovers exactly the FCI (and hand-derived) PAG."""
    full_dag, n_observed, expected = ALL_PAG_FIXTURES[name]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    data = np.zeros((10, n_observed))
    m = lmarvel(data, ci_test=oracle, alpha=0.5)
    f = fci(data, ci_test=oracle, alpha=0.5)
    assert _shd(m.endpoints, f.endpoints) == 0, (
        f"{name}: L-MARVEL != FCI\nL-MARVEL:\n{m.endpoints}\nFCI:\n{f.endpoints}"
    )
    assert _shd(m.endpoints, expected.endpoints) == 0, f"{name}: L-MARVEL != expected PAG"


def test_lmarvel_matches_fci_random_latents() -> None:
    """L-MARVEL == FCI on a broad sweep of random DAGs-with-latents under the m-separation oracle.
    Any single mismatch is a bug (the sound+complete gate)."""
    rng = np.random.default_rng(20260716)
    trials = 0
    for _ in range(200):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        m = lmarvel(data, ci_test=oracle, alpha=0.5)
        f = fci(data, ci_test=oracle, alpha=0.5)
        assert _shd(m.endpoints, f.endpoints) == 0, (
            f"n_obs={n_obs}: L-MARVEL != FCI\n{m.endpoints}\n{f.endpoints}"
        )
        trials += 1
    assert trials == 200


@pytest.mark.parametrize("mb_algo", ["grow_shrink", "iamb", "inter_iamb"])
def test_lmarvel_mb_algo_variants_match_fci(mb_algo: str) -> None:
    """Every supported initial Markov-boundary routine yields the same (correct) PAG."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        m = lmarvel(data, ci_test=oracle, alpha=0.5, mb_algo=mb_algo)
        f = fci(data, ci_test=oracle, alpha=0.5)
        assert _shd(m.endpoints, f.endpoints) == 0


def test_lmarvel_marks_are_pag_marks() -> None:
    """Output uses only PAG endpoint marks."""
    full_dag, n_obs, _ = ALL_PAG_FIXTURES["confounded_chain_through_collider"]()
    oracle = DSeparationOracleProjected(full_dag, n_obs)
    out = lmarvel(np.zeros((5, n_obs)), ci_test=oracle, alpha=0.5)
    seen = {int(v) for v in np.unique(out.endpoints)}
    assert seen <= {int(M.NO_EDGE), int(M.TAIL), int(M.ARROW), int(M.CIRCLE)}


# --------------------------------------------------------------------------------------------------
# Gate 2 — CI-test efficiency, recorded
# --------------------------------------------------------------------------------------------------


def test_lmarvel_cheaper_than_fci(capsys: pytest.CaptureFixture[str]) -> None:
    """On sparse graphs L-MARVEL issues fewer CI tests than FCI, total and unique -- the recorded
    efficiency win. Prints the aggregate ratios."""
    rng = np.random.default_rng(4242)
    lm_tot = f_tot = lm_uni = f_uni = 0
    per_trial_wins = 0
    n = 0
    for n_obs in (7, 8, 9, 10):
        for _ in range(20):
            n_lat = int(rng.integers(0, 3))
            p = n_obs + n_lat
            dag = _rand_dag(p, int(p * 1.2), rng)  # sparse
            oracle = DSeparationOracleProjected(dag, n_obs)
            data = np.zeros((5, n_obs))
            rl, rf = InMemoryRecorder(), InMemoryRecorder()
            m = lmarvel(data, ci_test=oracle, alpha=0.5, recorder=rl)
            f = fci(data, ci_test=oracle, alpha=0.5, recorder=rf)
            assert _shd(m.endpoints, f.endpoints) == 0  # parity holds here too
            mm, fm = rl.metrics(), rf.metrics()
            lm_tot += mm["n_ci_total"]
            f_tot += fm["n_ci_total"]
            lm_uni += mm["n_ci_unique"]
            f_uni += fm["n_ci_unique"]
            per_trial_wins += mm["n_ci_total"] <= fm["n_ci_total"]
            n += 1

    with capsys.disabled():
        print(
            f"\n[L-MARVEL vs FCI, {n} sparse DAGs-with-latents]  "
            f"total CI: L-MARVEL={lm_tot} FCI={f_tot} ({f_tot / lm_tot:.2f}x)  |  "
            f"unique CI: L-MARVEL={lm_uni} FCI={f_uni} ({f_uni / lm_uni:.2f}x)  |  "
            f"per-trial total<=FCI: {per_trial_wins}/{n}"
        )

    assert lm_tot < f_tot
    assert lm_uni < f_uni
    assert per_trial_wins >= int(0.85 * n)


# --------------------------------------------------------------------------------------------------
# Gate 3 — background-knowledge parity
# --------------------------------------------------------------------------------------------------


def test_lmarvel_matches_fci_with_background() -> None:
    """``lmarvel(bg) == fci(bg)`` for background consistent with the truth. Consistency is guaranteed
    by deriving the background from ``fci``'s own output (identified orientations + a non-edge)."""
    rng = np.random.default_rng(77)
    checks = 0
    facets = set()
    for _ in range(120):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        ep = fci(data, ci_test=oracle, alpha=0.5).endpoints
        directed = [
            (u, v)
            for u in range(n_obs)
            for v in range(n_obs)
            if ep[u, v] == M.ARROW and ep[v, u] == M.TAIL
        ]
        nonadj = [
            (i, j) for i in range(n_obs) for j in range(i + 1, n_obs) if ep[i, j] == M.NO_EDGE
        ]
        bks: list[tuple[str, BackgroundKnowledge]] = []
        if directed:
            bks.append(("required", BackgroundKnowledge(required_directed=frozenset(directed[:2]))))
            bks.append(
                (
                    "forbidden",
                    BackgroundKnowledge(forbidden_directed=frozenset((v, u) for u, v in directed[:2])),
                )
            )
        if nonadj:
            bks.append(
                (
                    "forbidden_adjacent",
                    BackgroundKnowledge(forbidden_adjacent=frozenset([frozenset(nonadj[0])])),
                )
            )
        for name, bg in bks:
            m = lmarvel(data, ci_test=oracle, alpha=0.5, background=bg)
            f = fci(data, ci_test=oracle, alpha=0.5, background=bg)
            assert _shd(m.endpoints, f.endpoints) == 0, f"[{name}] n_obs={n_obs}: lmarvel(bg) != fci(bg)"
            checks += 1
            facets.add(name)
    assert checks > 0
    assert {"required", "forbidden", "forbidden_adjacent"} <= facets


def test_lmarvel_background_none_is_regression() -> None:
    """``background=None`` changes nothing: identical to the default call and to ``fci``."""
    rng = np.random.default_rng(9)
    for _ in range(20):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        a = lmarvel(data, ci_test=oracle, alpha=0.5)
        b = lmarvel(data, ci_test=oracle, alpha=0.5, background=None)
        f = fci(data, ci_test=oracle, alpha=0.5)
        assert _shd(a.endpoints, b.endpoints) == 0
        assert _shd(b.endpoints, f.endpoints) == 0


# --------------------------------------------------------------------------------------------------
# Bookkeeping / input validation
# --------------------------------------------------------------------------------------------------


def test_lmarvel_records_run() -> None:
    """Run bracketed under the ``lmarvel`` algorithm name with the caller's run id."""
    full_dag, n_obs, _ = ALL_PAG_FIXTURES["y_structure"]()
    oracle = DSeparationOracleProjected(full_dag, n_obs)
    rec = InMemoryRecorder()
    lmarvel(np.zeros((5, n_obs)), ci_test=oracle, alpha=0.5, recorder=rec, run_id="lm")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "lmarvel"
    assert run["run_id"] == "lm"
    assert run["n_ci_total"] > 0


def test_lmarvel_rejects_unknown_mb_algo() -> None:
    full_dag, n_obs, _ = ALL_PAG_FIXTURES["y_structure"]()
    with pytest.raises(CBCDInputError):
        lmarvel(np.zeros((5, n_obs)), ci_test=DSeparationOracleProjected(full_dag, n_obs), mb_algo="nope")  # type: ignore[arg-type]


def test_lmarvel_single_and_edgeless() -> None:
    """Degenerate inputs: one variable, and an edge-free graph."""
    d1 = DAG.from_directed_edges(1, [])
    m1 = lmarvel(np.zeros((5, 1)), ci_test=DSeparationOracleProjected(d1, 1), alpha=0.5)
    assert m1.n_vars == 1 and not m1.adjacency().any()
    d3 = DAG.from_directed_edges(3, [])
    m3 = lmarvel(np.zeros((5, 3)), ci_test=DSeparationOracleProjected(d3, 3), alpha=0.5)
    assert m3.n_vars == 3 and not m3.adjacency().any()
