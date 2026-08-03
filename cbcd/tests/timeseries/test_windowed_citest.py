"""WindowedCITest: the generic lifter of any i.i.d. cbcd.CITest onto the lagged design.

Self-contained (no citests dependency): the injected i.i.d. test is cbcd's own FisherZ, which
lets us assert the key equivalence WindowedCITest(FisherZ) == ParCorr numerically -- i.e. ParCorr
is exactly lagged Fisher-Z. The cross-package path (citests tests via the factory) is exercised in
the Paper 3 E4 harness, not in cbcd's own suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.citest import FisherZ
from cbcd.timeseries import LaggedCITest, LaggedDataset, LaggedVar, ParCorr, WindowedCITest


def _dataset(seed: int = 0, T: int = 400, n_vars: int = 4, max_lag: int = 2) -> LaggedDataset:
    rng = np.random.default_rng(seed)
    data = np.zeros((T, n_vars))
    data[0] = rng.normal(size=n_vars)
    for t in range(1, T):  # a simple stable AR(1)-ish series so partial correlations are non-trivial
        data[t] = 0.5 * data[t - 1] + rng.normal(scale=1.0, size=n_vars)
    return LaggedDataset(data, max_lag=max_lag)


def test_conforms_to_lagged_protocol() -> None:
    ds = _dataset()
    w = WindowedCITest(ds, FisherZ)
    assert isinstance(w, LaggedCITest)
    assert w.n_vars == ds.n_vars and w.max_lag == ds.max_lag


def test_windowed_fisherz_matches_parcorr() -> None:
    ds = _dataset()
    w = WindowedCITest(ds, FisherZ)
    p = ParCorr(ds)
    lv = LaggedVar
    queries = [
        (lv(0, 0), lv(1, 0), []),
        (lv(0, 0), lv(0, -1), []),
        (lv(0, 0), lv(1, -1), [lv(0, -1)]),
        (lv(2, 0), lv(3, 0), [lv(0, 0), lv(1, -2)]),
        (lv(1, 0), lv(2, -2), [lv(3, -1), lv(0, 0)]),
    ]
    for x, y, S in queries:
        assert w(x, y, S) == pytest.approx(p(x, y, S), abs=1e-9), f"mismatch at {(x, y, S)}"


def test_details_carries_pvalue_and_n_effective() -> None:
    ds = _dataset(T=300, max_lag=2)
    w = WindowedCITest(ds, FisherZ)
    r = w.details(LaggedVar(0, 0), LaggedVar(1, -1), [LaggedVar(0, -1)])
    assert 0.0 <= r.p_value <= 1.0
    assert r.n_effective == 300 - 2


def test_rejects_factory_with_wrong_width() -> None:
    ds = _dataset()
    # A factory that ignores the design and builds a too-small test must be caught.
    bad = lambda _design: FisherZ(np.zeros((50, 2)))  # noqa: E731
    with pytest.raises(Exception):
        WindowedCITest(ds, bad)
