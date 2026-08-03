"""Markov-blanket / parents-children discovery primitives (roadmap Phase 1)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.mb import grow_shrink, iamb, inter_iamb, mmpc
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle

MBAlgo = Callable[..., frozenset[int]]
MB_ALGOS: list[MBAlgo] = [grow_shrink, iamb, inter_iamb]


def _true_pc(dag: DAG, i: int) -> frozenset[int]:
    return frozenset(dag.parents(i)) | frozenset(dag.children(i))


def _true_mb(dag: DAG, i: int) -> frozenset[int]:
    spouses: set[int] = set()
    for c in dag.children(i):
        spouses |= set(dag.parents(c))
    spouses.discard(i)
    return _true_pc(dag, i) | frozenset(spouses)


def _collider_dag() -> DAG:
    """X0 -> X2 <- X1, X2 -> X3. MB(0) = {2 (child), 1 (spouse)}."""
    return DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])


@pytest.mark.parametrize("algo", MB_ALGOS)
def test_mb_recovers_spouse_on_collider(algo: MBAlgo) -> None:
    dag = _collider_dag()
    oracle = DSeparationOracle(dag)
    assert algo(oracle, 0, alpha=0.5) == frozenset({1, 2})  # spouse 1 must be found
    for i in range(dag.n_vars):
        assert algo(oracle, i, alpha=0.5) == _true_mb(dag, i)


@pytest.mark.parametrize("algo", MB_ALGOS)
def test_mb_recovers_asia(algo: MBAlgo) -> None:
    dag, _ = ALL_FIXTURES["asia"]()
    oracle = DSeparationOracle(dag)
    for i in range(dag.n_vars):
        assert algo(oracle, i, alpha=0.5) == _true_mb(dag, i)


def test_mmpc_recovers_pc_on_collider() -> None:
    dag = _collider_dag()
    oracle = DSeparationOracle(dag)
    for i in range(dag.n_vars):
        assert mmpc(oracle, i, alpha=0.5) == _true_pc(dag, i)  # spouse excluded from PC


def test_mmpc_recovers_pc_asia() -> None:
    dag, _ = ALL_FIXTURES["asia"]()
    oracle = DSeparationOracle(dag)
    for i in range(dag.n_vars):
        assert mmpc(oracle, i, alpha=0.5) == _true_pc(dag, i)


def test_mmpc_symmetry_and_subset_of_raw() -> None:
    dag, _ = ALL_FIXTURES["asia"]()
    oracle = DSeparationOracle(dag)
    for i in range(dag.n_vars):
        raw = mmpc(oracle, i, alpha=0.5, symmetry=None)
        anded = mmpc(oracle, i, alpha=0.5, symmetry="and")
        assert anded <= raw


def test_recorder_captures_ci_calls() -> None:
    from cbcd import InMemoryRecorder

    oracle = DSeparationOracle(_collider_dag())
    rec = InMemoryRecorder()
    iamb(oracle, 0, alpha=0.5, recorder=rec)
    assert rec.metrics()["n_ci_total"] >= 1


def test_validation() -> None:
    oracle = DSeparationOracle(_collider_dag())
    with pytest.raises(CBCDInputError):
        iamb(oracle, 99, alpha=0.5)
    with pytest.raises(CBCDInputError):
        mmpc(oracle, 0, alpha=0.5, symmetry="xor")
