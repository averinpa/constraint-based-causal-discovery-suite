"""D-separation CI oracle wrapping a generated DAG.

Exposes :class:`DSeparationOracle`, a duck-typed implementation of the
``cbcd.CITest`` Protocol. The oracle does not depend on or import
``cbcd`` — conformance is purely structural:

* ``n_vars: int`` attribute,
* ``__call__(x, y, S) -> float`` returning the p-value,
* ``details(x, y, S)`` returning an object with a ``.p_value`` attribute.

The p-value is 1.0 when ``x ⫫ y | S`` holds under d-separation in the
underlying DAG and 0.0 otherwise, so any constraint-based algorithm
that decides independence on ``p > alpha`` for ``alpha < 1`` recovers
the oracle answer exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
from networkx.algorithms.d_separation import is_d_separator


@dataclass(frozen=True, slots=True)
class _CITestResult:
    """Minimal result type — only ``p_value`` is read by callers."""

    p_value: float


class DSeparationOracle:
    """D-separation CI oracle for a DAG over named variables.

    Variables are addressed by integer indices in ``[0, n_vars)``;
    index ``i`` corresponds to ``var_names[i]`` in the underlying graph.
    Returns ``p_value=1.0`` for d-separated pairs and ``0.0`` otherwise.
    """

    def __init__(self, dag: nx.DiGraph, var_names: Sequence[str]):
        names = tuple(var_names)
        if len(set(names)) != len(names):
            raise ValueError("var_names must be unique")
        graph_nodes = set(dag.nodes())
        missing = [n for n in names if n not in graph_nodes]
        if missing:
            raise ValueError(f"var_names contain nodes not in the DAG: {missing}")
        self._dag = dag
        self.var_names: tuple[str, ...] = names
        self.n_vars: int = len(names)

    def _check_idx(self, idx: int, label: str) -> None:
        if not 0 <= idx < self.n_vars:
            raise IndexError(f"{label} index {idx} out of range [0, {self.n_vars})")

    def _is_d_separated(self, x: int, y: int, S: Sequence[int]) -> bool:
        self._check_idx(x, "x")
        self._check_idx(y, "y")
        if x == y:
            raise ValueError("x and y must differ")
        s_set: set[str] = set()
        for s in S:
            s_int = int(s)
            self._check_idx(s_int, "S")
            s_set.add(self.var_names[s_int])
        return bool(
            is_d_separator(
                self._dag,
                {self.var_names[x]},
                {self.var_names[y]},
                s_set,
            )
        )

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return 1.0 if self._is_d_separated(x, y, S) else 0.0

    def details(self, x: int, y: int, S: Sequence[int]) -> _CITestResult:
        return _CITestResult(p_value=self.__call__(x, y, S))
