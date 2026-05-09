"""Numerical correctness of Fisher-Z."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cbcd.citest import FisherZ
from cbcd.exceptions import CBCDDataError, CBCDInputError


def _reference_fisher_z(
    data: np.ndarray, x: int, y: int, S: tuple[int, ...]
) -> tuple[float, float]:
    """Independent reference: partial correlation via correlation-matrix Schur."""
    n = data.shape[0]
    centered = data - data.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1, keepdims=True)
    normed = centered / std
    corr = (normed.T @ normed) / (n - 1)
    if not S:
        r = float(corr[x, y])
    else:
        idx = [x, y, *S]
        sub = corr[np.ix_(idx, idx)]
        inv = np.linalg.inv(sub)
        r = float(-inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1]))
    df = n - len(S) - 3
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    p = 2.0 * float(norm.sf(abs(z) * np.sqrt(df)))
    return r, p


def test_fisherz_unconditional_matches_correlation() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 5
    data = rng.standard_normal((n, p))
    ci = FisherZ(data)
    res = ci.details(0, 1, [])
    expected_r, expected_p = _reference_fisher_z(data, 0, 1, ())
    assert res.extra["r"] == pytest.approx(expected_r, abs=1e-12)
    assert res.p_value == pytest.approx(expected_p, abs=1e-12)
    assert res.df == n - 3


def test_fisherz_conditional_matches_reference() -> None:
    # Closed-form 3-variable case: X = e1, Z = X + e2, Y = Z + e3.
    # Then X and Y are dependent unconditionally, independent given Z.
    rng = np.random.default_rng(42)
    n = 5000
    e1 = rng.standard_normal(n)
    e2 = rng.standard_normal(n)
    e3 = rng.standard_normal(n)
    X = e1
    Z = X + e2
    Y = Z + e3
    data = np.column_stack([X, Y, Z])

    ci = FisherZ(data)
    # Unconditional: should reject H0 of independence.
    res_uncond = ci.details(0, 1, [])
    assert res_uncond.p_value < 1e-10
    # Conditional on Z (idx 2): should not reject.
    res_cond = ci.details(0, 1, [2])
    assert res_cond.p_value > 0.05

    # Check exact-match against reference.
    _, p_uncond_ref = _reference_fisher_z(data, 0, 1, ())
    _, p_cond_ref = _reference_fisher_z(data, 0, 1, (2,))
    assert res_uncond.p_value == pytest.approx(p_uncond_ref, abs=1e-12)
    assert res_cond.p_value == pytest.approx(p_cond_ref, abs=1e-12)


def test_fisherz_dataframe_input() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((100, 3)), columns=["a", "b", "c"])
    ci = FisherZ(df)
    assert ci.var_names == ("a", "b", "c")
    # DataFrame and ndarray inputs agree.
    ci_arr = FisherZ(df.to_numpy())
    assert ci.details(0, 1, []).p_value == pytest.approx(
        ci_arr.details(0, 1, []).p_value, abs=1e-15
    )


def test_fisherz_rejects_nan() -> None:
    data = np.array([[1.0, 2.0], [np.nan, 3.0], [1.5, 2.5], [2.0, 3.5]])
    with pytest.raises(CBCDDataError):
        FisherZ(data)


def test_fisherz_rejects_zero_variance() -> None:
    data = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 2.5], [1.0, 3.5]])
    with pytest.raises(CBCDDataError):
        FisherZ(data)


def test_fisherz_rejects_x_in_S() -> None:
    rng = np.random.default_rng(0)
    ci = FisherZ(rng.standard_normal((50, 3)))
    with pytest.raises(CBCDInputError):
        ci.details(0, 1, [0])
    with pytest.raises(CBCDInputError):
        ci.details(0, 0, [])
