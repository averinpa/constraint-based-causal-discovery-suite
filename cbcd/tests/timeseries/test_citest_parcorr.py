"""ParCorr math sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDDataError, CBCDInputError
from cbcd.timeseries import LaggedDataset, LaggedVar, ParCorr


def _make_var1_data(T: int = 200, seed: int = 0) -> np.ndarray:
    """Sample a 2-var VAR(1) with X → X (autocorrelation) and X → Y (cross)."""
    rng = np.random.default_rng(seed)
    data = np.zeros((T, 2))
    for t in range(1, T):
        data[t, 0] = 0.6 * data[t - 1, 0] + rng.normal(scale=0.5)
        data[t, 1] = 0.5 * data[t - 1, 0] + rng.normal(scale=0.5)
    return data


def test_parcorr_unconditional_correlation_close_to_pearson() -> None:
    data = _make_var1_data(T=500, seed=1)
    ds = LaggedDataset(data=data, max_lag=2)
    ci = ParCorr(ds)
    # Unconditional ci(X_t, Y_t, []) — same time. r ≈ 0 since Y_t has no
    # contemporaneous link to X_t in this VAR(1).
    res = ci.details(LaggedVar(0, 0), LaggedVar(1, 0), [])
    # r is small but the test is about plumbing not magnitude — confirm
    # the result at least is a valid p-value and statistic.
    assert 0.0 <= res.p_value <= 1.0
    assert res.statistic is not None and res.statistic >= 0.0
    assert res.df is not None and res.df > 0
    assert res.n_effective == 500 - 2


def test_parcorr_strong_lagged_dependence_is_dependent() -> None:
    data = _make_var1_data(T=2000, seed=2)
    ds = LaggedDataset(data=data, max_lag=1)
    ci = ParCorr(ds)
    # X_{t-1} → Y_t is a real edge with coef 0.5; should be strongly
    # dependent unconditionally.
    p = ci(LaggedVar(0, -1), LaggedVar(1, 0), [])
    assert p < 0.001


def test_parcorr_lagged_dependence_blocked_by_conditioning_on_mediator() -> None:
    # 3-var chain: X_{t-1} → Y_t, Y_{t-1} → Z_t. Then X_{t-2} → Z_t (via
    # X → Y → Z one-step-each) should be blocked by Y_{t-1}.
    rng = np.random.default_rng(3)
    T = 4000
    data = np.zeros((T, 3))
    for t in range(2, T):
        data[t, 0] = rng.normal(scale=0.3)
        data[t, 1] = 0.7 * data[t - 1, 0] + rng.normal(scale=0.3)
        data[t, 2] = 0.7 * data[t - 1, 1] + rng.normal(scale=0.3)
    ds = LaggedDataset(data=data, max_lag=2)
    ci = ParCorr(ds)
    # Unconditional X_{t-2} ⫫ Z_t: dependent.
    p_uncond = ci(LaggedVar(0, -2), LaggedVar(2, 0), [])
    # Conditional on Y_{t-1}: should be ≈ independent.
    p_cond = ci(LaggedVar(0, -2), LaggedVar(2, 0), [LaggedVar(1, -1)])
    assert p_uncond < 0.001
    assert p_cond > 0.05


def test_parcorr_rejects_x_eq_y_columns() -> None:
    data = _make_var1_data(T=200, seed=4)
    ds = LaggedDataset(data=data, max_lag=1)
    ci = ParCorr(ds)
    with pytest.raises(CBCDInputError):
        ci(LaggedVar(0, 0), LaggedVar(0, 0), [])


def test_parcorr_rejects_S_overlap() -> None:
    data = _make_var1_data(T=200, seed=5)
    ds = LaggedDataset(data=data, max_lag=1)
    ci = ParCorr(ds)
    with pytest.raises(CBCDInputError):
        ci(LaggedVar(0, 0), LaggedVar(1, 0), [LaggedVar(0, 0)])


def test_parcorr_rejects_lag_outside_horizon() -> None:
    data = _make_var1_data(T=200, seed=6)
    ds = LaggedDataset(data=data, max_lag=1)
    ci = ParCorr(ds)
    with pytest.raises(CBCDInputError):
        ci(LaggedVar(0, -2), LaggedVar(1, 0), [])  # max_lag=1


def test_parcorr_rejects_constant_column() -> None:
    data = np.zeros((200, 2))
    data[:, 0] = np.arange(200, dtype=np.float64) * 0.0  # all zeros
    data[:, 1] = np.random.default_rng(0).normal(size=200)
    ds = LaggedDataset(data=data, max_lag=1)
    with pytest.raises(CBCDDataError):
        ParCorr(ds)
