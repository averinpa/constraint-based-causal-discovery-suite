"""Lagged d-separation oracle for time-series structural-correctness tests.

Given a ``TimeSeriesDAG`` (the true causal structure under stationarity), the
oracle unrolls the graph into a static ``nx.DiGraph`` over ``(var, t)`` pairs
for ``t ∈ [0, T_max]`` and answers ``is_d_separator`` queries using
``networkx``. Stationarity of the process means we can pick any sufficiently-
interior reference time ``T`` and translate ``LaggedVar(v, -τ)`` to
``(v, T - τ)`` in the unrolled graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from cbcd.graph.marks import EndpointMark
from cbcd.timeseries.citest import LaggedCITestResult
from cbcd.timeseries.graph import TimeSeriesDAG
from cbcd.timeseries.lagged import LaggedVar


class DSeparationOracleLagged:
    """LaggedCITest that answers d-separation on an unrolled time-series DAG."""

    n_vars: int
    max_lag: int

    def __init__(self, true_dag: TimeSeriesDAG, *, t_horizon: int | None = None) -> None:
        self.n_vars = true_dag.n_vars
        self.max_lag = true_dag.max_lag
        if t_horizon is None:
            t_horizon = max(3 * true_dag.max_lag, 6)
        self._t_horizon = t_horizon
        # Reference time: deep enough into the unrolled graph that every lag
        # maps to a non-negative absolute time.
        self._t_ref = max(2 * true_dag.max_lag, 3)

        g = nx.DiGraph()
        for v in range(true_dag.n_vars):
            for t in range(t_horizon + 1):
                g.add_node((v, t))
        # Lag-0 (contemporaneous) directed edges: i → j at every t.
        for i in range(true_dag.n_vars):
            for j in range(true_dag.n_vars):
                if i == j:
                    continue
                if (
                    true_dag.endpoints[0, i, j] == EndpointMark.ARROW
                    and true_dag.endpoints[0, j, i] == EndpointMark.TAIL
                ):
                    for t in range(t_horizon + 1):
                        g.add_edge((i, t), (j, t))
        # Lagged edges (τ ≥ 1): (i, t-τ) → (j, t) for valid t.
        for tau in range(1, true_dag.max_lag + 1):
            for i in range(true_dag.n_vars):
                for j in range(true_dag.n_vars):
                    if true_dag.endpoints[tau, i, j] == EndpointMark.ARROW:
                        for t in range(tau, t_horizon + 1):
                            g.add_edge((i, t - tau), (j, t))
        self._g = g

    def _node(self, lv: LaggedVar) -> tuple[int, int]:
        # lv.lag is ≤ 0. Map to (var, t_ref + lv.lag).
        t = self._t_ref + lv.lag
        if t < 0 or t > self._t_horizon:
            raise ValueError(f"LaggedVar {lv} maps to t={t} outside [0, {self._t_horizon}]")
        return (lv.var, t)

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> LaggedCITestResult:
        x_node = self._node(x)
        y_node = self._node(y)
        cond = {self._node(s) for s in S}
        if nx.is_d_separator(self._g, {x_node}, {y_node}, cond):
            return LaggedCITestResult(p_value=1.0)
        return LaggedCITestResult(p_value=0.0)
