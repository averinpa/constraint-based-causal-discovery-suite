"""Markov-blanket subgraph extraction.

The Markov blanket of variable ``v`` in a graph ``g`` is::

    MB(v) = {v} ∪ parents(v) ∪ children(v) ∪ co_parents(children(v))
            ∪ undirected_neighbours(v)

Same definition as bnmetrics 0.1.x (`utils.py:71-126`) on the int8 endpoint
matrix. Bidirected/CIRCLE neighbours are NOT included in the blanket
in this v0.2 release (matches 0.1.x DAG/CPDAG semantics); PAG-aware
extension is open work.
"""

from __future__ import annotations

import numpy as np

from bnmetrics._graph import _Graph
from bnmetrics.adapter import _resolve_var, _to_endpoints
from bnmetrics.descriptive import _directed_children, _directed_parents
from bnmetrics.marks import EndpointMark

TAIL = int(EndpointMark.TAIL)


def markov_blanket_indices(g: object, var: int | str) -> tuple[int, ...]:
    """Indices of `var`'s Markov blanket in `g`'s ORIGINAL index space.

    Useful when callers want to slice their own arrays without
    re-numbering. The returned tuple is sorted ascending.
    """
    n_vars, endpoints, var_names = _to_endpoints(g)
    v = _resolve_var(var, var_names, n_vars)

    parents = set(int(u) for u in _directed_parents(endpoints, v))
    children = set(int(u) for u in _directed_children(endpoints, v))
    co_parents: set[int] = set()
    for c in children:
        for u in _directed_parents(endpoints, c):
            iu = int(u)
            if iu != v:
                co_parents.add(iu)

    # Undirected neighbours: endpoints[u, v] == TAIL == endpoints[v, u].
    col = endpoints[:, v]
    row = endpoints[v, :]
    undirected = set(int(u) for u in np.where((col == TAIL) & (row == TAIL))[0])

    blanket = {v} | parents | children | co_parents | undirected
    return tuple(sorted(blanket))


def markov_blanket(g: object, var: int | str) -> _Graph:
    """Return the Markov-blanket subgraph of `var` in `g`.

    The returned `_Graph` satisfies `GraphLike`. Variable indices are
    re-numbered to ``0..k-1`` over the blanket. ``var_names`` of the
    sub-graph carries the original names for the included variables
    (preserving caller-side identity).
    """
    n_vars, endpoints, var_names = _to_endpoints(g)
    v = _resolve_var(var, var_names, n_vars)

    indices = markov_blanket_indices(g, var)
    sub_n = len(indices)
    sub_endpoints = endpoints[np.ix_(indices, indices)].copy()

    sub_names: tuple[str, ...] | None = (
        tuple(var_names[i] for i in indices) if var_names is not None else None
    )

    # `v` is resolved up-front to validate the `var` argument; reuse-elision
    # is fine since markov_blanket_indices does the same lookup.
    del v

    return _Graph(n_vars=sub_n, endpoints=sub_endpoints, var_names=sub_names)
