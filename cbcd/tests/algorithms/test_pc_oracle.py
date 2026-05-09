"""Structural regression: pc() with d-separation oracle recovers true CPDAG.

This is the gold-standard correctness bar. The oracle removes statistical
noise, so any non-zero structural Hamming distance reflects a bug in the
algorithm wiring or an incorrect Markov-equivalence class derivation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import pc
from cbcd.graph.marks import EndpointMark
from tests.fixtures import ALL_FIXTURES
from tests.oracle import DSeparationOracle


def _shd(a: np.ndarray, b: np.ndarray) -> int:
    """Number of endpoint mismatches across the matrix (counted once per edge)."""
    n = a.shape[0]
    diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] != b[i, j] or a[j, i] != b[j, i]:
                diff += 1
    return diff


@pytest.mark.parametrize("name", list(ALL_FIXTURES.keys()))
def test_pc_recovers_true_cpdag(name: str) -> None:
    factory = ALL_FIXTURES[name]
    dag, expected_cpdag = factory()
    oracle = DSeparationOracle(dag)
    n = dag.n_vars
    # Need a data array of correct shape so _normalize_data is happy; the
    # oracle ignores it.
    dummy_data = np.zeros((10, n), dtype=np.float64)
    recovered = pc(dummy_data, ci_test=oracle, alpha=0.5)
    assert recovered.n_vars == expected_cpdag.n_vars
    assert _shd(recovered.endpoints, expected_cpdag.endpoints) == 0, (
        f"fixture {name}: SHD != 0\n"
        f"recovered:\n{recovered.endpoints}\n"
        f"expected:\n{expected_cpdag.endpoints}"
    )


def test_recovered_cpdag_extends_to_dag() -> None:
    """Sanity: every recovered CPDAG must be DAG-extendable."""
    for name in ["y_structure", "fork", "chain", "m_structure", "diamond", "asia"]:
        dag, _ = ALL_FIXTURES[name]()
        oracle = DSeparationOracle(dag)
        dummy = np.zeros((10, dag.n_vars), dtype=np.float64)
        recovered = pc(dummy, ci_test=oracle, alpha=0.5)
        assert recovered.to_dag_extension() is not None, f"fixture {name} not extendable"


def test_recovered_endpoints_are_only_tail_or_arrow() -> None:
    dag, _ = ALL_FIXTURES["asia"]()
    oracle = DSeparationOracle(dag)
    dummy = np.zeros((10, dag.n_vars), dtype=np.float64)
    recovered = pc(dummy, ci_test=oracle, alpha=0.5)
    seen = set(int(m) for m in np.unique(recovered.endpoints))
    assert seen <= {EndpointMark.NO_EDGE, EndpointMark.TAIL, EndpointMark.ARROW}
