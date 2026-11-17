"""CML (Coordinated Multi-Neighborhood Learning) — native efficient local discovery."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import InMemoryRecorder, cml, pc
from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark as M
from tests.oracle import DSeparationOracle


def _rand_dag(p: int, n_edges: int, rng: np.random.Generator) -> DAG:
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


def _target_adj(ep: np.ndarray, t: int, p: int) -> set[frozenset[int]]:
    return {frozenset({t, j}) for j in range(p) if j != t and ep[t, j] != M.NO_EDGE}


def test_cml_matches_global_pc_locally() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = int(rng.integers(7, 11))
        dag = _rand_dag(p, p + 2, rng)
        oracle = DSeparationOracle(dag)
        data = np.zeros((5, p))
        t = int(rng.integers(0, p))
        c = cml(data, [t], ci_test=oracle, alpha=0.5)
        g = pc(data, ci_test=oracle, alpha=0.5)
        assert _target_adj(c.endpoints, t, p) == _target_adj(g.endpoints, t, p)


def test_cml_multi_target() -> None:
    # 0 -> 2 <- 1, 2 -> 3. Query both 0 and 3; recover edges around each.
    dag = DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])
    oracle = DSeparationOracle(dag)
    c = cml(np.zeros((5, 4)), [0, 3], ci_test=oracle, alpha=0.5)
    assert c.endpoints[0, 2] != M.NO_EDGE
    assert c.endpoints[2, 3] != M.NO_EDGE


def test_cml_records_and_validates() -> None:
    dag = DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])
    oracle = DSeparationOracle(dag)
    data = np.zeros((5, 4))
    rec = InMemoryRecorder()
    cml(data, [2], ci_test=oracle, alpha=0.5, recorder=rec, run_id="c")
    run = rec.to_frames()["runs"].iloc[0]
    assert run["algorithm"] == "cml" and run["run_id"] == "c"
    with pytest.raises(CBCDInputError):
        cml(data, [9], ci_test=oracle)
    with pytest.raises(CBCDInputError):
        cml(data, [], ci_test=oracle)
