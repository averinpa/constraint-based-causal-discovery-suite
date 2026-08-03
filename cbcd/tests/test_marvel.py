"""MARVEL structural parity + CI-efficiency regression.

Two independent gates (the reason MARVEL exists):

1. **Correctness = PC.** With the d-separation oracle (statistical noise removed), ``marvel`` must
   return the *identical* CPDAG to ``cbcd.pc`` on every fixture and on random DAGs. Any endpoint
   mismatch is a wiring bug, not sampling error. This is the sound+complete gate (Thm 13).
2. **Efficiency win, measured.** On the same oracle runs the recorded CI-test count (``RunRecorder``)
   must be no larger than PC's, and strictly lower in aggregate on sparse graphs -- the whole point
   of the recursive Markov-boundary elimination. The actual counts are printed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import InMemoryRecorder, marvel, pc
from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark as M
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle


def _shd(a: np.ndarray, b: np.ndarray) -> int:
    """Endpoint mismatches, counted once per unordered pair."""
    n = a.shape[0]
    diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] != b[i, j] or a[j, i] != b[j, i]:
                diff += 1
    return diff


def _rand_dag(p: int, n_edges: int, rng: np.random.Generator) -> DAG:
    """Random DAG on ``p`` nodes with ``n_edges`` edges, acyclic by a random topological order."""
    order = rng.permutation(p)
    ep = np.zeros((p, p), np.int8)
    c = 0
    while c < n_edges:
        a, b = (int(x) for x in rng.integers(0, p, 2))
        if a == b:
            continue
        i, j = (a, b) if np.where(order == a)[0][0] < np.where(order == b)[0][0] else (b, a)
        if ep[i, j] == 0:
            ep[i, j] = M.ARROW
            ep[j, i] = M.TAIL
            c += 1
    return DAG(p, ep)


# --------------------------------------------------------------------------------------------------
# Gate 1 — structural parity with PC
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_FIXTURES.keys()))
def test_marvel_matches_pc_on_fixtures(name: str) -> None:
    """On every reference DAG, MARVEL recovers exactly the PC (and hand-derived) CPDAG."""
    dag, expected = ALL_FIXTURES[name]()
    oracle = DSeparationOracle(dag)
    data = np.zeros((10, dag.n_vars))
    m = marvel(data, ci_test=oracle, alpha=0.5)
    g = pc(data, ci_test=oracle, alpha=0.5)
    assert _shd(m.endpoints, g.endpoints) == 0, (
        f"{name}: MARVEL != PC\nMARVEL:\n{m.endpoints}\nPC:\n{g.endpoints}"
    )
    assert _shd(m.endpoints, expected.endpoints) == 0, f"{name}: MARVEL != expected CPDAG"


def test_marvel_matches_pc_random_dags() -> None:
    """MARVEL == PC on a broad sweep of random sparse/moderately-dense DAGs (the sound+complete
    gate). Any single mismatch is a bug."""
    rng = np.random.default_rng(12345)
    trials = 0
    for p, dens in [(6, 1.0), (8, 1.0), (10, 1.2), (12, 1.2), (15, 1.3), (10, 2.0), (14, 2.0)]:
        for _ in range(20):
            dag = _rand_dag(p, int(p * dens), rng)
            oracle = DSeparationOracle(dag)
            data = np.zeros((5, p))
            m = marvel(data, ci_test=oracle, alpha=0.5)
            g = pc(data, ci_test=oracle, alpha=0.5)
            assert _shd(m.endpoints, g.endpoints) == 0, (
                f"p={p} edges={int(p * dens)}: MARVEL != PC\n{m.endpoints}\n{g.endpoints}"
            )
            trials += 1
    assert trials == 140


@pytest.mark.parametrize("mb_algo", ["grow_shrink", "iamb", "inter_iamb"])
def test_marvel_mb_algo_variants_match_pc(mb_algo: str) -> None:
    """Every supported initial Markov-boundary routine yields the same (correct) CPDAG."""
    rng = np.random.default_rng(7)
    for _ in range(15):
        dag = _rand_dag(10, 12, rng)
        oracle = DSeparationOracle(dag)
        data = np.zeros((5, 10))
        m = marvel(data, ci_test=oracle, alpha=0.5, mb_algo=mb_algo)
        g = pc(data, ci_test=oracle, alpha=0.5)
        assert _shd(m.endpoints, g.endpoints) == 0


# --------------------------------------------------------------------------------------------------
# Gate 2 — CI-test efficiency, recorded
# --------------------------------------------------------------------------------------------------


def test_marvel_cheaper_than_pc(capsys: pytest.CaptureFixture[str]) -> None:
    """On sparse graphs MARVEL issues strictly fewer CI tests than PC, both total and unique --
    the recorded efficiency win. Prints the aggregate counts."""
    rng = np.random.default_rng(2024)
    m_tot = p_tot = m_uni = p_uni = 0
    per_trial_wins = 0
    n = 0
    for p in (10, 12, 15, 18, 20):
        for _ in range(15):
            dag = _rand_dag(p, int(p * 1.3), rng)  # sparse: ~1.3 edges/node
            oracle = DSeparationOracle(dag)
            data = np.zeros((5, p))
            rm, rp = InMemoryRecorder(), InMemoryRecorder()
            m = marvel(data, ci_test=oracle, alpha=0.5, recorder=rm)
            g = pc(data, ci_test=oracle, alpha=0.5, recorder=rp)
            assert _shd(m.endpoints, g.endpoints) == 0  # parity must hold on these too
            mm, pm = rm.metrics(), rp.metrics()
            m_tot += mm["n_ci_total"]
            p_tot += pm["n_ci_total"]
            m_uni += mm["n_ci_unique"]
            p_uni += pm["n_ci_unique"]
            per_trial_wins += mm["n_ci_total"] <= pm["n_ci_total"]
            n += 1

    with capsys.disabled():
        print(
            f"\n[MARVEL vs PC, {n} sparse DAGs]  "
            f"total CI: MARVEL={m_tot} PC={p_tot} ({p_tot / m_tot:.2f}x)  |  "
            f"unique CI: MARVEL={m_uni} PC={p_uni} ({p_uni / m_uni:.2f}x)  |  "
            f"per-trial total<=PC: {per_trial_wins}/{n}"
        )

    # Aggregate efficiency win: MARVEL must be well below PC on both axes.
    assert m_tot < p_tot
    assert m_uni < p_uni
    # And it should win on the large majority of individual sparse instances.
    assert per_trial_wins >= int(0.9 * n)


# --------------------------------------------------------------------------------------------------
# Bookkeeping / input validation
# --------------------------------------------------------------------------------------------------


def test_marvel_records_run() -> None:
    """The run is bracketed under the ``marvel`` algorithm name with the caller's run id."""
    dag = DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])
    oracle = DSeparationOracle(dag)
    rec = InMemoryRecorder()
    marvel(np.zeros((5, 4)), ci_test=oracle, alpha=0.5, recorder=rec, run_id="mv")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "marvel"
    assert run["run_id"] == "mv"
    assert run["n_ci_total"] > 0


def test_marvel_rejects_unknown_mb_algo() -> None:
    with pytest.raises(CBCDInputError):
        marvel(np.zeros((5, 3)), ci_test=DSeparationOracle(DAG(3)), mb_algo="nope")  # type: ignore[arg-type]


def test_marvel_single_and_edgeless() -> None:
    """Degenerate graphs: one variable, and an edge-free graph, both return an empty CPDAG."""
    m1 = marvel(np.zeros((5, 1)), ci_test=DSeparationOracle(DAG(1)), alpha=0.5)
    assert m1.n_vars == 1 and m1.directed_edges() == () and m1.undirected_edges() == ()
    m3 = marvel(np.zeros((5, 3)), ci_test=DSeparationOracle(DAG(3)), alpha=0.5)
    assert m3.n_vars == 3 and not m3.adjacency().any()
