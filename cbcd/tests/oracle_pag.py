"""D-separation oracle that projects to observed variables.

Wraps a DAG over observed + latent variables (latents at indices ≥ n_observed)
and answers d-separation queries from the full DAG. The CI test surface is
over observed indices only (``n_vars = n_observed``); latents stay hidden but
participate in d-separation as expected.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from cbcd.citest.protocol import CITestResult
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark


class DSeparationOracleProjected:
    n_vars: int

    def __init__(self, full_dag: DAG, n_observed: int) -> None:
        self.n_vars = n_observed
        self._g = nx.DiGraph()
        self._g.add_nodes_from(range(full_dag.n_vars))
        for i in range(full_dag.n_vars):
            for j in range(full_dag.n_vars):
                if (
                    full_dag.endpoints[i, j] == EndpointMark.ARROW
                    and full_dag.endpoints[j, i] == EndpointMark.TAIL
                ):
                    self._g.add_edge(i, j)

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        cond = {int(s) for s in S}
        if nx.is_d_separator(self._g, {int(x)}, {int(y)}, cond):
            return CITestResult(p_value=1.0)
        return CITestResult(p_value=0.0)
