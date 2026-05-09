"""TimeSeriesDAG / TimeSeriesCPDAG construction, accessors, projections."""

from __future__ import annotations

import numpy as np
import pytest

from cbcd.exceptions import CBCDInputError
from cbcd.graph import EndpointMark
from cbcd.timeseries import (
    LaggedVar,
    TimeSeriesCPDAG,
    TimeSeriesDAG,
)

ARR = EndpointMark.ARROW
TAIL = EndpointMark.TAIL
NO = EndpointMark.NO_EDGE


def _ts_endpoints_directed(
    n_vars: int,
    max_lag: int,
    lagged: list[tuple[int, int, int]],
    contemp: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build endpoints from a list of lagged edges (src_var, dst_var, tau) and
    optional directed contemporaneous edges (src_var, dst_var)."""
    ep = np.zeros((max_lag + 1, n_vars, n_vars), dtype=np.int8)
    for src, dst, tau in lagged:
        ep[tau, src, dst] = ARR
    for src, dst in contemp or []:
        ep[0, src, dst] = ARR
        ep[0, dst, src] = TAIL
    return ep


# --- TimeSeriesDAG ---------------------------------------------------------


def test_ts_dag_construction() -> None:
    # 2-var VAR(1): X_{t-1} → X_t, X_{t-1} → Y_t, Y_{t-1} → Y_t.
    ep = _ts_endpoints_directed(2, 1, [(0, 0, 1), (0, 1, 1), (1, 1, 1)])
    g = TimeSeriesDAG(2, 1, ep)
    assert g.has_edge(LaggedVar(0, -1), LaggedVar(0, 0))
    assert g.has_edge(LaggedVar(0, -1), LaggedVar(1, 0))
    assert not g.has_edge(LaggedVar(1, -1), LaggedVar(0, 0))


def test_ts_dag_rejects_lag0_self_loop() -> None:
    ep = np.zeros((2, 2, 2), dtype=np.int8)
    ep[0, 0, 0] = TAIL  # self-loop at lag 0
    with pytest.raises(CBCDInputError):
        TimeSeriesDAG(2, 1, ep)


def test_ts_dag_rejects_lag0_asymmetric_no_edge() -> None:
    ep = np.zeros((2, 2, 2), dtype=np.int8)
    ep[0, 0, 1] = ARR  # one side present, other side NO_EDGE
    with pytest.raises(CBCDInputError):
        TimeSeriesDAG(2, 1, ep)


def test_ts_dag_rejects_undirected_lag0_edge() -> None:
    ep = np.zeros((2, 2, 2), dtype=np.int8)
    ep[0, 0, 1] = TAIL
    ep[0, 1, 0] = TAIL  # undirected (TAIL-TAIL) — not allowed in DAG
    with pytest.raises(CBCDInputError):
        TimeSeriesDAG(2, 1, ep)


def test_ts_dag_lagged_edges() -> None:
    ep = _ts_endpoints_directed(2, 2, [(0, 1, 1), (1, 0, 2)])
    g = TimeSeriesDAG(2, 2, ep)
    edges = g.lagged_edges()
    keys = {(e.src.var, e.src.lag, e.dst.var, e.dst.lag) for e in edges}
    assert keys == {(0, -1, 1, 0), (1, -2, 0, 0)}


def test_ts_dag_parents_lag0() -> None:
    # X_{t-1} → X_t, Y_{t-1} → X_t, Y_{t-2} → X_t  → parents(X_t)
    ep = _ts_endpoints_directed(2, 2, [(0, 0, 1), (1, 0, 1), (1, 0, 2)])
    g = TimeSeriesDAG(2, 2, ep)
    parents_of_x = g.parents(LaggedVar(0, 0))
    assert set(parents_of_x) == {LaggedVar(0, -1), LaggedVar(1, -1), LaggedVar(1, -2)}


def test_ts_dag_parents_rejects_non_zero_lag() -> None:
    g = TimeSeriesDAG(2, 1, np.zeros((2, 2, 2), dtype=np.int8))
    with pytest.raises(CBCDInputError):
        g.parents(LaggedVar(0, -1))


def test_ts_dag_to_summary_graph() -> None:
    # Lagged edges X→Y at τ=1, Y→Z at τ=1; summary graph is X→Y→Z.
    ep = np.zeros((2, 3, 3), dtype=np.int8)
    ep[1, 0, 1] = ARR  # X_{t-1} → Y_t
    ep[1, 1, 2] = ARR  # Y_{t-1} → Z_t
    g = TimeSeriesDAG(3, 1, ep)
    dag = g.to_summary_graph()
    assert set(dag.directed_edges()) == {(0, 1), (1, 2)}


def test_ts_dag_to_contemporaneous_graph() -> None:
    ep = np.zeros((2, 3, 3), dtype=np.int8)
    ep[0, 0, 1] = ARR
    ep[0, 1, 0] = TAIL  # 0 → 1 contemporaneous
    ep[1, 0, 2] = ARR  # lagged edge — should NOT appear in contemp graph
    g = TimeSeriesDAG(3, 1, ep)
    contemp = g.to_contemporaneous_graph()
    assert set(contemp.directed_edges()) == {(0, 1)}


def test_ts_dag_lagged_cell_must_be_arrow() -> None:
    ep = np.zeros((2, 2, 2), dtype=np.int8)
    ep[1, 0, 1] = TAIL  # not allowed: lagged cell must be ARROW or NO_EDGE
    with pytest.raises(CBCDInputError):
        TimeSeriesDAG(2, 1, ep)


# --- TimeSeriesCPDAG -------------------------------------------------------


def test_ts_cpdag_allows_undirected_lag0() -> None:
    ep = np.zeros((2, 2, 2), dtype=np.int8)
    ep[0, 0, 1] = TAIL
    ep[0, 1, 0] = TAIL
    g = TimeSeriesCPDAG(2, 1, ep)
    assert g.endpoints[0, 0, 1] == TAIL


def test_ts_cpdag_lagged_arrows_only() -> None:
    ep = _ts_endpoints_directed(2, 1, [(0, 1, 1), (1, 0, 1)])
    g = TimeSeriesCPDAG(2, 1, ep)
    edges = g.lagged_edges()
    assert len(edges) == 2


def test_ts_cpdag_equality() -> None:
    ep1 = _ts_endpoints_directed(2, 1, [(0, 1, 1)])
    ep2 = _ts_endpoints_directed(2, 1, [(0, 1, 1)])
    ep3 = _ts_endpoints_directed(2, 1, [(1, 0, 1)])
    assert TimeSeriesCPDAG(2, 1, ep1) == TimeSeriesCPDAG(2, 1, ep2)
    assert TimeSeriesCPDAG(2, 1, ep1) != TimeSeriesCPDAG(2, 1, ep3)
