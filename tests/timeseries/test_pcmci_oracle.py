"""pcmci() recovers expected TimeSeriesCPDAG with d-sep oracle (SHD = 0)."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import pcmci
from cbcd.graph import EndpointMark
from cbcd.timeseries import LaggedDataset
from tests.timeseries.fixtures import ALL_TS_FIXTURES
from tests.timeseries.oracle import DSeparationOracleLagged


def _shd_endpoints(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


@pytest.mark.parametrize("name", list(ALL_TS_FIXTURES))
def test_pcmci_recovers_expected_cpdag(name: str) -> None:
    true_dag, expected = ALL_TS_FIXTURES[name]()
    oracle = DSeparationOracleLagged(true_dag)
    # pcmci() expects a LaggedDataset; the data array is unused when ci_test
    # is provided directly (its n_vars and max_lag are checked against the
    # dataset's). Provide enough rows for the LaggedDataset to validate.
    data = np.zeros((50, true_dag.n_vars), dtype=np.float64)
    ds = LaggedDataset(data=data, max_lag=true_dag.max_lag)
    out = pcmci(ds, ci_test=oracle, alpha=0.5)
    assert out.n_vars == expected.n_vars
    assert out.max_lag == expected.max_lag
    assert _shd_endpoints(out.endpoints, expected.endpoints) == 0, (
        f"{name}: recovered\n{out.endpoints}\nexpected\n{expected.endpoints}"
    )


@pytest.mark.parametrize("name", list(ALL_TS_FIXTURES))
def test_pcmci_marks_are_arrow_or_no_edge(name: str) -> None:
    true_dag, _ = ALL_TS_FIXTURES[name]()
    oracle = DSeparationOracleLagged(true_dag)
    data = np.zeros((50, true_dag.n_vars), dtype=np.float64)
    ds = LaggedDataset(data=data, max_lag=true_dag.max_lag)
    out = pcmci(ds, ci_test=oracle, alpha=0.5)
    permitted = {int(EndpointMark.NO_EDGE), int(EndpointMark.ARROW)}
    seen = {int(m) for m in np.unique(out.endpoints)}
    assert seen <= permitted
