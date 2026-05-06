"""LaggedVar / LaggedDataset / LaggedBackgroundKnowledge construction + validation."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.timeseries import LaggedBackgroundKnowledge, LaggedDataset, LaggedVar

# --- LaggedVar -------------------------------------------------------------


def test_lagged_var_zero_lag_is_contemporaneous() -> None:
    v = LaggedVar(0, 0)
    assert v.is_contemporaneous


def test_lagged_var_negative_lag() -> None:
    v = LaggedVar(0, -2)
    assert not v.is_contemporaneous
    assert v.lag == -2


def test_lagged_var_rejects_positive_lag() -> None:
    with pytest.raises(CBCDInputError):
        LaggedVar(0, 1)


def test_lagged_var_rejects_negative_var() -> None:
    with pytest.raises(CBCDInputError):
        LaggedVar(-1, 0)


def test_lagged_var_hashable() -> None:
    s = {LaggedVar(0, -1), LaggedVar(0, -1), LaggedVar(1, 0)}
    assert len(s) == 2


# --- LaggedDataset ---------------------------------------------------------


def test_lagged_dataset_basic() -> None:
    data = np.zeros((100, 3))
    ds = LaggedDataset(data=data, max_lag=2)
    assert ds.n_vars == 3
    assert ds.n_samples == 100
    assert ds.max_lag == 2


def test_lagged_dataset_rejects_max_lag_too_large() -> None:
    # max_lag must be < T - 1; T=10, max_lag=9 → invalid.
    with pytest.raises(CBCDInputError):
        LaggedDataset(data=np.zeros((10, 2)), max_lag=9)


def test_lagged_dataset_rejects_negative_max_lag() -> None:
    with pytest.raises(CBCDInputError):
        LaggedDataset(data=np.zeros((10, 2)), max_lag=-1)


def test_lagged_dataset_rejects_1d_data() -> None:
    with pytest.raises(CBCDInputError):
        LaggedDataset(data=np.zeros(10), max_lag=1)


def test_lagged_dataset_var_names_length_check() -> None:
    with pytest.raises(CBCDInputError):
        LaggedDataset(data=np.zeros((20, 3)), max_lag=1, var_names=("x", "y"))


# --- LaggedBackgroundKnowledge --------------------------------------------


def test_lbk_default_empty() -> None:
    bk = LaggedBackgroundKnowledge()
    assert bk.forbidden_lagged == frozenset()
    assert bk.required_lagged == frozenset()


def test_lbk_validates_time_direction() -> None:
    # src.lag > dst.lag is invalid (must be <=).
    bk_required = frozenset({(LaggedVar(0, 0), LaggedVar(1, -1))})
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(required_lagged=bk_required)


def test_lbk_rejects_self_loop() -> None:
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(required_lagged=frozenset({(LaggedVar(0, 0), LaggedVar(0, 0))}))


def test_lbk_rejects_overlap_required_forbidden() -> None:
    edge = (LaggedVar(0, -1), LaggedVar(1, 0))
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(
            required_lagged=frozenset({edge}),
            forbidden_lagged=frozenset({edge}),
        )


def test_lbk_rejects_required_autoregressive_when_forbidden() -> None:
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(
            required_lagged=frozenset({(LaggedVar(0, -1), LaggedVar(0, 0))}),
            no_autoregressive=frozenset({0}),
        )


def test_lbk_rejects_required_contemporaneous_when_forbidden_pair() -> None:
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(
            required_lagged=frozenset({(LaggedVar(0, 0), LaggedVar(1, 0))}),
            no_contemporaneous=frozenset({frozenset({0, 1})}),
        )


def test_lbk_rejects_required_violating_tier() -> None:
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(
            required_lagged=frozenset({(LaggedVar(1, 0), LaggedVar(0, 0))}),
            contemporaneous_tiers=(frozenset({0}), frozenset({1})),
        )


def test_lbk_rejects_var_in_multiple_tiers() -> None:
    with pytest.raises(CBCDInputError):
        LaggedBackgroundKnowledge(
            contemporaneous_tiers=(frozenset({0, 1}), frozenset({1, 2})),
        )


def test_lbk_is_forbidden_lagged_autoregressive() -> None:
    bk = LaggedBackgroundKnowledge(no_autoregressive=frozenset({0}))
    assert bk.is_forbidden_lagged(LaggedVar(0, -1), LaggedVar(0, 0))
    assert not bk.is_forbidden_lagged(LaggedVar(1, -1), LaggedVar(1, 0))


def test_lbk_is_forbidden_lagged_contemporaneous_tier() -> None:
    bk = LaggedBackgroundKnowledge(
        contemporaneous_tiers=(frozenset({0}), frozenset({1, 2})),
    )
    # tier 1 → tier 0 is forbidden.
    assert bk.is_forbidden_lagged(LaggedVar(1, 0), LaggedVar(0, 0))
    # tier 0 → tier 1 is OK.
    assert not bk.is_forbidden_lagged(LaggedVar(0, 0), LaggedVar(1, 0))
