"""Directed acyclic graph."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError
from cbcd.graph.base import _GraphBase
from cbcd.graph.marks import EndpointMark


class DAG(_GraphBase):
    """Directed acyclic graph. Endpoints are TAIL/ARROW only; no cycles."""

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(n_vars, endpoints, var_names)

        permitted = {EndpointMark.NO_EDGE, EndpointMark.TAIL, EndpointMark.ARROW}
        marks_seen = set(int(m) for m in np.unique(self.endpoints))
        unsupported = marks_seen - {int(p) for p in permitted}
        if unsupported:
            raise CBCDInputError(
                f"DAG only supports NO_EDGE/TAIL/ARROW marks; got {sorted(unsupported)}"
            )

        # Every present edge must be directed: one end ARROW, other end TAIL.
        for i, j in zip(*np.where(self.endpoints != EndpointMark.NO_EDGE), strict=True):
            i_, j_ = int(i), int(j)
            if i_ >= j_:
                continue
            mij = int(self.endpoints[i_, j_])
            mji = int(self.endpoints[j_, i_])
            if not (
                (mij == EndpointMark.ARROW and mji == EndpointMark.TAIL)
                or (mij == EndpointMark.TAIL and mji == EndpointMark.ARROW)
            ):
                raise CBCDInputError(
                    f"DAG edge ({i_}, {j_}) is not directed: marks=({mji}, {mij})"
                )

        if _has_directed_cycle(self.endpoints):
            raise CBCDInputError("DAG must be acyclic")

    @classmethod
    def from_directed_edges(
        cls,
        n_vars: int,
        edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        var_names: tuple[str, ...] | None = None,
    ) -> DAG:
        """Construct a DAG from a list of (parent, child) directed edges."""
        ep = np.zeros((n_vars, n_vars), dtype=np.int8)
        for parent, child in edges:
            ep[parent, child] = EndpointMark.ARROW
            ep[child, parent] = EndpointMark.TAIL
        return cls(n_vars, ep, var_names)

    def parents(self, i: int) -> tuple[int, ...]:
        # j is parent of i when edge j → i: endpoints[j, i] = ARROW.
        return tuple(int(j) for j in np.where(self.endpoints[:, i] == EndpointMark.ARROW)[0])

    def children(self, i: int) -> tuple[int, ...]:
        # j is child of i when edge i → j: endpoints[i, j] = ARROW.
        return tuple(int(j) for j in np.where(self.endpoints[i, :] == EndpointMark.ARROW)[0])

    def directed_edges(self) -> tuple[tuple[int, int], ...]:
        rows, cols = np.where(self.endpoints == EndpointMark.ARROW)
        return tuple((int(r), int(c)) for r, c in zip(rows, cols, strict=True))


def _has_directed_cycle(endpoints: NDArray[np.int8]) -> bool:
    n = endpoints.shape[0]
    # Build child adjacency.
    children = [
        list(int(j) for j in np.where(endpoints[i, :] == EndpointMark.ARROW)[0])
        for i in range(n)
    ]
    indegree = [0] * n
    for u in range(n):
        for v in children[u]:
            indegree[v] += 1
    queue = [i for i in range(n) if indegree[i] == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in children[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return visited != n
