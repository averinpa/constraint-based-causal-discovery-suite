"""PC1Skeleton recovers true lagged parents on oracle fixtures."""

from __future__ import annotations

import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.timeseries import LaggedVar, PC1Skeleton
from tests.timeseries.fixtures import ALL_TS_FIXTURES
from tests.timeseries.oracle import DSeparationOracleLagged


def _true_parents(true_dag, target_var: int) -> set[LaggedVar]:
    """Read off the true lagged parent set for ``target_var`` at lag 0."""
    out: set[LaggedVar] = set()
    for tau in range(1, true_dag.max_lag + 1):
        for src in range(true_dag.n_vars):
            if true_dag.endpoints[tau, src, target_var] != 0:  # ARROW
                out.add(LaggedVar(src, -tau))
    return out


@pytest.mark.parametrize("name", list(ALL_TS_FIXTURES))
def test_pc1_recovers_true_parents(name: str) -> None:
    true_dag, _ = ALL_TS_FIXTURES[name]()
    oracle = DSeparationOracleLagged(true_dag)
    skel = PC1Skeleton()(oracle, alpha=0.5)
    for target_var in range(true_dag.n_vars):
        target = LaggedVar(target_var, 0)
        recovered = set(skel.parents[target])
        truth = _true_parents(true_dag, target_var)
        assert recovered == truth, (
            f"{name} target={target}: recovered {recovered}, expected {truth}"
        )


def test_pc1_records_sepsets_for_removed_pairs() -> None:
    true_dag, _ = ALL_TS_FIXTURES["sparse_var2"]()
    oracle = DSeparationOracleLagged(true_dag)
    skel = PC1Skeleton()(oracle, alpha=0.5)
    # X_{t-2} → Y_t is NOT a true edge (only X_{t-1} → Y_t exists). PC₁ must
    # have removed (X, -2) from parents(Y_t) — sepset is recorded.
    pair = frozenset({LaggedVar(0, -2), LaggedVar(1, 0)})
    assert pair in skel.sepsets


def test_pc1_rejects_n_jobs_gt_1() -> None:
    true_dag, _ = ALL_TS_FIXTURES["ar1_single"]()
    oracle = DSeparationOracleLagged(true_dag)
    with pytest.raises(CBCDInputError):
        PC1Skeleton()(oracle, alpha=0.5, n_jobs=2)
