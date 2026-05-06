"""D-separation oracle CI test for structural-correctness testing.

Returns p_value = 1.0 when X and Y are d-separated by S, p_value = 0.0 otherwise.
This removes statistical noise — any discrepancy in the recovered structure is
a bug in the algorithm, not the statistics.

Uses networkx's d_separated implementation as the ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from cbcd.citest.protocol import CITestResult
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark


class DSeparationOracle:
    """CITest implementation that answers d-separation queries on a known DAG."""

    n_vars: int

    def __init__(self, true_dag: DAG) -> None:
        self.n_vars = true_dag.n_vars
        self._g = nx.DiGraph()
        self._g.add_nodes_from(range(true_dag.n_vars))
        for i in range(true_dag.n_vars):
            for j in range(true_dag.n_vars):
                if (
                    true_dag.endpoints[i, j] == EndpointMark.ARROW
                    and true_dag.endpoints[j, i] == EndpointMark.TAIL
                ):
                    self._g.add_edge(i, j)

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        cond = {int(s) for s in S}
        if nx.is_d_separator(self._g, {int(x)}, {int(y)}, cond):
            return CITestResult(p_value=1.0)
        return CITestResult(p_value=0.0)
