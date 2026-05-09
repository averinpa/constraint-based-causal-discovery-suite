"""Completed Partially Directed Acyclic Graph (CPDAG) and intermediate."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError
from cbcd.graph.base import _GraphBase
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark


def _validate_cpdag_marks(endpoints: NDArray[np.int8]) -> None:
    permitted = {EndpointMark.NO_EDGE, EndpointMark.TAIL, EndpointMark.ARROW}
    marks_seen = {int(m) for m in np.unique(endpoints)}
    unsupported = marks_seen - {int(p) for p in permitted}
    if unsupported:
        raise CBCDInputError(
            f"CPDAG only supports NO_EDGE/TAIL/ARROW marks; got {sorted(unsupported)}"
        )
    n = endpoints.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            mij = int(endpoints[i, j])
            mji = int(endpoints[j, i])
            if mij == EndpointMark.NO_EDGE and mji == EndpointMark.NO_EDGE:
                continue
            if mij == EndpointMark.ARROW and mji == EndpointMark.TAIL:
                continue
            if mij == EndpointMark.TAIL and mji == EndpointMark.ARROW:
                continue
            if mij == EndpointMark.TAIL and mji == EndpointMark.TAIL:
                continue
            raise CBCDInputError(
                f"CPDAG edge ({i}, {j}) is not directed or undirected: marks=({mji}, {mij})"
            )


class PartialCPDAG(_GraphBase):
    """Intermediate state between collider orientation and edge-rule closure.

    Carries the same endpoint matrix shape as ``CPDAG`` and ``ambiguous_triples``
    discovered during collider orientation. Not necessarily Meek-closed.
    """

    ambiguous_triples: frozenset[tuple[int, int, int]]

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
        ambiguous_triples: frozenset[tuple[int, int, int]] = frozenset(),
    ) -> None:
        super().__init__(n_vars, endpoints, var_names)
        _validate_cpdag_marks(self.endpoints)
        self.ambiguous_triples = frozenset(ambiguous_triples)


class CPDAG(_GraphBase):
    """Completed Partially Directed Acyclic Graph.

    Directed edges represent oriented arcs that are common to all DAGs in the
    Markov equivalence class; undirected edges are reversible.
    """

    ambiguous_triples: frozenset[tuple[int, int, int]]
    definite_non_colliders: frozenset[tuple[int, int, int]]

    def __init__(
        self,
        n_vars: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
        ambiguous_triples: frozenset[tuple[int, int, int]] = frozenset(),
        definite_non_colliders: frozenset[tuple[int, int, int]] = frozenset(),
    ) -> None:
        super().__init__(n_vars, endpoints, var_names)
        _validate_cpdag_marks(self.endpoints)
        self.ambiguous_triples = frozenset(ambiguous_triples)
        self.definite_non_colliders = frozenset(definite_non_colliders)

    def directed_edges(self) -> tuple[tuple[int, int], ...]:
        rows, cols = np.where(self.endpoints == EndpointMark.ARROW)
        return tuple((int(r), int(c)) for r, c in zip(rows, cols, strict=True))

    def undirected_edges(self) -> tuple[frozenset[int], ...]:
        seen: set[frozenset[int]] = set()
        out: list[frozenset[int]] = []
        n = self.n_vars
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    self.endpoints[i, j] == EndpointMark.TAIL
                    and self.endpoints[j, i] == EndpointMark.TAIL
                ):
                    fs = frozenset({i, j})
                    if fs not in seen:
                        seen.add(fs)
                        out.append(fs)
        return tuple(out)

    def parents(self, i: int) -> tuple[int, ...]:
        out: list[int] = []
        for j in range(self.n_vars):
            if (
                self.endpoints[j, i] == EndpointMark.ARROW
                and self.endpoints[i, j] == EndpointMark.TAIL
            ):
                out.append(j)
        return tuple(out)

    def neighbors(self, i: int) -> tuple[int, ...]:
        """Undirected neighbours of i (i.e., j with i — j)."""
        out: list[int] = []
        for j in range(self.n_vars):
            if i == j:
                continue
            if (
                self.endpoints[i, j] == EndpointMark.TAIL
                and self.endpoints[j, i] == EndpointMark.TAIL
            ):
                out.append(j)
        return tuple(out)

    def adjacent(self, i: int) -> tuple[int, ...]:
        return tuple(int(j) for j in np.where(self.endpoints[i, :] != EndpointMark.NO_EDGE)[0])

    def to_dag_extension(self) -> DAG | None:
        """Return a DAG in the Markov equivalence class, or None if none exists.

        Dor & Tarsi (1992): repeatedly find a vertex X with (a) no outgoing
        directed edge among non-oriented vertices, (b) every undirected
        neighbour adjacent to every parent of X. Orient all undirected edges
        incident to X as directed into X.
        """
        endpoints = self.endpoints.copy()
        n = self.n_vars
        oriented = np.zeros(n, dtype=bool)

        for _ in range(n):
            chosen = -1
            for x in range(n):
                if oriented[x]:
                    continue
                has_outgoing = False
                for y in range(n):
                    if oriented[y] or y == x:
                        continue
                    if (
                        endpoints[x, y] == EndpointMark.ARROW
                        and endpoints[y, x] == EndpointMark.TAIL
                    ):
                        has_outgoing = True
                        break
                if has_outgoing:
                    continue
                nbrs = [
                    y
                    for y in range(n)
                    if not oriented[y]
                    and y != x
                    and endpoints[x, y] == EndpointMark.TAIL
                    and endpoints[y, x] == EndpointMark.TAIL
                ]
                parents_x = [
                    y
                    for y in range(n)
                    if y != x
                    and endpoints[y, x] == EndpointMark.ARROW
                    and endpoints[x, y] == EndpointMark.TAIL
                ]
                bad = False
                for nb in nbrs:
                    for p in parents_x:
                        if nb == p:
                            continue
                        if endpoints[nb, p] == EndpointMark.NO_EDGE:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue
                chosen = x
                break

            if chosen < 0:
                return None
            for y in range(n):
                if y == chosen or oriented[y]:
                    continue
                if (
                    endpoints[chosen, y] == EndpointMark.TAIL
                    and endpoints[y, chosen] == EndpointMark.TAIL
                ):
                    endpoints[y, chosen] = EndpointMark.ARROW
                    endpoints[chosen, y] = EndpointMark.TAIL
            oriented[chosen] = True

        return DAG(n, endpoints, self.var_names)
