"""VAR-process fixtures for time-series structural-correctness tests."""

from __future__ import annotations

import numpy as np

from cbcd.graph import EndpointMark
from cbcd.timeseries import TimeSeriesCPDAG, TimeSeriesDAG


def _ts_endpoints(
    n_vars: int,
    max_lag: int,
    lagged: list[tuple[int, int, int]],
) -> np.ndarray:
    ep = np.zeros((max_lag + 1, n_vars, n_vars), dtype=np.int8)
    for src, dst, tau in lagged:
        ep[tau, src, dst] = EndpointMark.ARROW
    return ep


def ar1_single_var() -> tuple[TimeSeriesDAG, TimeSeriesCPDAG]:
    """X_{t-1} → X_t. The simplest possible time-series DAG."""
    ep = _ts_endpoints(1, 1, [(0, 0, 1)])
    return TimeSeriesDAG(1, 1, ep), TimeSeriesCPDAG(1, 1, ep.copy())


def two_var_var1() -> tuple[TimeSeriesDAG, TimeSeriesCPDAG]:
    """2-variable VAR(1):
    X_{t-1} → X_t, X_{t-1} → Y_t, Y_{t-1} → X_t, Y_{t-1} → Y_t.
    """
    edges = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    ep = _ts_endpoints(2, 1, edges)
    return TimeSeriesDAG(2, 1, ep), TimeSeriesCPDAG(2, 1, ep.copy())


def sparse_var2() -> tuple[TimeSeriesDAG, TimeSeriesCPDAG]:
    """3-variable sparse VAR(2):
    X_{t-1} → Y_t, Y_{t-2} → Z_t, X_{t-2} → Z_t. No autocorrelation.
    """
    edges = [(0, 1, 1), (1, 2, 2), (0, 2, 2)]
    ep = _ts_endpoints(3, 2, edges)
    return TimeSeriesDAG(3, 2, ep), TimeSeriesCPDAG(3, 2, ep.copy())


ALL_TS_FIXTURES = {
    "ar1_single": ar1_single_var,
    "two_var_var1": two_var_var1,
    "sparse_var2": sparse_var2,
}
