"""Input validation for pcmci()."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd import pcmci
from cbcd.exceptions import CBCDInputError
from cbcd.timeseries import LaggedDataset
from tests.timeseries.fixtures import ALL_TS_FIXTURES
from tests.timeseries.oracle import DSeparationOracleLagged


def _setup(name: str = "two_var_var1"):
    true_dag, _ = ALL_TS_FIXTURES[name]()
    oracle = DSeparationOracleLagged(true_dag)
    data = np.zeros((50, true_dag.n_vars), dtype=np.float64)
    ds = LaggedDataset(data=data, max_lag=true_dag.max_lag)
    return ds, oracle


def test_pcmci_alpha_out_of_bounds() -> None:
    ds, oracle = _setup()
    with pytest.raises(CBCDInputError):
        pcmci(ds, ci_test=oracle, alpha=0.0)
    with pytest.raises(CBCDInputError):
        pcmci(ds, ci_test=oracle, alpha=1.0)


def test_pcmci_pc_alpha_defaults_to_alpha() -> None:
    ds, oracle = _setup()
    a = pcmci(ds, ci_test=oracle, alpha=0.5, pc_alpha=None)
    b = pcmci(ds, ci_test=oracle, alpha=0.5, pc_alpha=0.5)
    assert np.array_equal(a.endpoints, b.endpoints)


def test_pcmci_pc_alpha_out_of_bounds() -> None:
    ds, oracle = _setup()
    with pytest.raises(CBCDInputError):
        pcmci(ds, ci_test=oracle, alpha=0.5, pc_alpha=1.0)


def test_pcmci_n_jobs_gt_1_rejected() -> None:
    ds, oracle = _setup()
    with pytest.raises(CBCDInputError):
        pcmci(ds, ci_test=oracle, alpha=0.5, n_jobs=2)


def test_pcmci_ci_test_n_vars_mismatch() -> None:
    true_dag, _ = ALL_TS_FIXTURES["two_var_var1"]()  # n_vars=2
    oracle = DSeparationOracleLagged(true_dag)
    data = np.zeros((50, 3), dtype=np.float64)  # n_vars=3, mismatched
    ds = LaggedDataset(data=data, max_lag=1)
    with pytest.raises(CBCDInputError):
        pcmci(ds, ci_test=oracle, alpha=0.5)


def test_pcmci_ci_test_max_lag_mismatch() -> None:
    true_dag, _ = ALL_TS_FIXTURES["two_var_var1"]()  # max_lag=1
    oracle = DSeparationOracleLagged(true_dag)
    data = np.zeros((50, true_dag.n_vars), dtype=np.float64)
    ds = LaggedDataset(data=data, max_lag=2)  # mismatched
    with pytest.raises(CBCDInputError):
        pcmci(ds, ci_test=oracle, alpha=0.5)


def test_pcmci_string_ci_test_resolves_to_parcorr() -> None:
    rng = np.random.default_rng(0)
    T = 500
    data = np.zeros((T, 2))
    for t in range(1, T):
        data[t, 0] = 0.6 * data[t - 1, 0] + rng.normal(scale=0.5)
        data[t, 1] = 0.5 * data[t - 1, 0] + rng.normal(scale=0.5)
    ds = LaggedDataset(data=data, max_lag=1)
    out = pcmci(ds, ci_test="parcorr", alpha=0.05)
    assert out.n_vars == 2
    assert out.max_lag == 1
