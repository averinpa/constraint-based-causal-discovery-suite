"""Input validation for fci() / rfci() / anytime_fci()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cbcd import anytime_fci, fci, rfci
from cbcd.exceptions import CBCDInputError
from tests.fixtures_pag import ALL_PAG_FIXTURES
from tests.oracle_pag import DSeparationOracleProjected


def _oracle_for(name: str):
    full_dag, n_observed, _ = ALL_PAG_FIXTURES[name]()
    oracle = DSeparationOracleProjected(full_dag, n_observed)
    return oracle, n_observed


def test_fci_alpha_out_of_bounds() -> None:
    oracle, n = _oracle_for("y_structure")
    data = np.zeros((10, n), dtype=np.float64)
    with pytest.raises(CBCDInputError):
        fci(data, ci_test=oracle, alpha=0.0)
    with pytest.raises(CBCDInputError):
        fci(data, ci_test=oracle, alpha=1.0)


def test_fci_n_jobs_gt_1_rejected() -> None:
    oracle, n = _oracle_for("y_structure")
    data = np.zeros((10, n), dtype=np.float64)
    with pytest.raises(CBCDInputError):
        fci(data, ci_test=oracle, alpha=0.5, n_jobs=2)


def test_fci_ci_test_dim_mismatch() -> None:
    oracle, n = _oracle_for("y_structure")
    # data has too many columns relative to ci_test.n_vars.
    data = np.zeros((10, n + 1), dtype=np.float64)
    with pytest.raises(CBCDInputError):
        fci(data, ci_test=oracle, alpha=0.5)


def test_fci_dataframe_equiv_ndarray() -> None:
    oracle, n = _oracle_for("confounded_chain_through_collider")
    data_array = np.zeros((10, n), dtype=np.float64)
    data_df = pd.DataFrame(data_array, columns=[f"v{i}" for i in range(n)])
    out_array = fci(data_array, ci_test=oracle, alpha=0.5)
    out_df = fci(data_df, ci_test=oracle, alpha=0.5)
    assert np.array_equal(out_array.endpoints, out_df.endpoints)


def test_rfci_n_jobs_gt_1_rejected() -> None:
    oracle, n = _oracle_for("y_structure")
    data = np.zeros((10, n), dtype=np.float64)
    with pytest.raises(CBCDInputError):
        rfci(data, ci_test=oracle, alpha=0.5, n_jobs=2)


def test_anytime_fci_max_cond_set_required_positional() -> None:
    oracle, n = _oracle_for("y_structure")
    data = np.zeros((10, n), dtype=np.float64)
    out = anytime_fci(data, 1, ci_test=oracle, alpha=0.5)
    assert out.n_vars == n
