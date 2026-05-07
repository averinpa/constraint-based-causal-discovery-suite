"""Descriptive metrics over a single graph.

All functions accept any GraphLikeInput (GraphLike, ndarray, list, or
networkx.DiGraph) and return scalar counts. Per-node functions take a
``var: int | str`` handle resolved against ``var_names``.

Edge-mark conventions (matching :class:`bnm.EndpointMark`):

    directed   i→j : endpoints[i,j]=ARROW, endpoints[j,i]=TAIL
    undirected i—j : endpoints[i,j]=TAIL,  endpoints[j,i]=TAIL
    bidirected i↔j : endpoints[i,j]=ARROW, endpoints[j,i]=ARROW
    circle (PAG)   : at least one endpoint is CIRCLE

For collider/reversible/root/leaf semantics on DAGs and CPDAGs, this
module matches bnm 0.1.x exactly — verified by the legacy snapshot in
``tests/fixtures_legacy.json``. Generalisation to PAG inputs (bidirected
parents, CIRCLE-aware shielding) is conservative (PAG-specific marks
are not yet treated as shield-equivalent to directed edges) and will
be revisited when the first PAG-input use case appears.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bnm.adapter import _resolve_var, _to_endpoints
from bnm.marks import EndpointMark

NO_EDGE = int(EndpointMark.NO_EDGE)
TAIL = int(EndpointMark.TAIL)
ARROW = int(EndpointMark.ARROW)
CIRCLE = int(EndpointMark.CIRCLE)


# ---- helpers ------------------------------------------------------------


def _upper_pairs(endpoints: NDArray[np.int8]) -> tuple[NDArray, NDArray]:
    """Return (mark_at_j_array, mark_at_i_array) for the upper-triangle
    edges of `endpoints`. Used to count edges by type without
    double-counting."""
    n = endpoints.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return endpoints[iu, ju], endpoints[ju, iu]


def _directed_parents(endpoints: NDArray[np.int8], v: int) -> NDArray[np.int_]:
    """Indices `u` such that `u → v` is a directed edge."""
    col = endpoints[:, v]
    row = endpoints[v, :]
    return np.where((col == ARROW) & (row == TAIL))[0]


def _directed_children(endpoints: NDArray[np.int8], v: int) -> NDArray[np.int_]:
    """Indices `u` such that `v → u` is a directed edge."""
    col = endpoints[:, v]
    row = endpoints[v, :]
    return np.where((row == ARROW) & (col == TAIL))[0]


def _directed_adjacent(endpoints: NDArray[np.int8], a: int, b: int) -> bool:
    """True iff there is a directed edge between `a` and `b` in either
    direction. 0.1.x's shielding test for collider counting."""
    a_to_b = endpoints[a, b] == ARROW and endpoints[b, a] == TAIL
    b_to_a = endpoints[b, a] == ARROW and endpoints[a, b] == TAIL
    return a_to_b or b_to_a


# ---- whole-graph counts ------------------------------------------------


def count_edges(g: object) -> int:
    """Total edges (directed + undirected + bidirected + circle-ended)."""
    _, endpoints, _ = _to_endpoints(g)
    a, b = _upper_pairs(endpoints)
    return int(np.sum((a != NO_EDGE) | (b != NO_EDGE)))


def count_nodes(g: object) -> int:
    """Number of variables — equivalent to `g.n_vars` after normalisation."""
    n_vars, _, _ = _to_endpoints(g)
    return n_vars


def count_directed_arcs(g: object) -> int:
    """Edges with one ARROW and one TAIL endpoint."""
    _, endpoints, _ = _to_endpoints(g)
    a, b = _upper_pairs(endpoints)
    return int(np.sum(((a == ARROW) & (b == TAIL)) | ((a == TAIL) & (b == ARROW))))


def count_undirected_arcs(g: object) -> int:
    """Edges with two TAIL endpoints."""
    _, endpoints, _ = _to_endpoints(g)
    a, b = _upper_pairs(endpoints)
    return int(np.sum((a == TAIL) & (b == TAIL)))


def count_bidirected_arcs(g: object) -> int:
    """Edges with two ARROW endpoints. (Not present in bnm 0.1.x output;
    requires int8-matrix or cbcd PAG input to be non-zero.)"""
    _, endpoints, _ = _to_endpoints(g)
    a, b = _upper_pairs(endpoints)
    return int(np.sum((a == ARROW) & (b == ARROW)))


def count_circle_edges(g: object) -> int:
    """Edges where at least one endpoint is CIRCLE."""
    _, endpoints, _ = _to_endpoints(g)
    a, b = _upper_pairs(endpoints)
    return int(np.sum((a == CIRCLE) | (b == CIRCLE)))


def count_colliders(g: object) -> int:
    """Unshielded colliders ``u → v ← w`` with `u, w` not directed-adjacent.

    Definition (0.1.x parity on DAG/CPDAG inputs):
      - parents of v = {u : u → v is a directed edge}.
      - shield(u, w) = there is a directed edge between u and w in either
        direction.
      - For each pair of parents (u, w) of v, increment if not shielded.
    """
    _, endpoints, _ = _to_endpoints(g)
    n_vars = endpoints.shape[0]
    count = 0
    for v in range(n_vars):
        parents = _directed_parents(endpoints, v)
        if len(parents) < 2:
            continue
        for k in range(len(parents)):
            for m in range(k + 1, len(parents)):
                u, w = int(parents[k]), int(parents[m])
                if not _directed_adjacent(endpoints, u, w):
                    count += 1
    return count


def count_root_nodes(g: object) -> int:
    """Nodes with no directed in-edges and no undirected/bidirected/circle
    incidence (strict)."""
    _, endpoints, _ = _to_endpoints(g)
    n_vars = endpoints.shape[0]
    count = 0
    for v in range(n_vars):
        col = endpoints[:, v]
        row = endpoints[v, :]
        has_directed_parent = bool(np.any((col == ARROW) & (row == TAIL)))
        # Any undirected/bidirected/circle incidence at v?
        has_undirected = bool(
            np.any((col == TAIL) & (row == TAIL))
            | np.any((col == ARROW) & (row == ARROW))
            | np.any((col == CIRCLE) | (row == CIRCLE))
        )
        if not has_directed_parent and not has_undirected:
            count += 1
    return count


def count_leaf_nodes(g: object) -> int:
    """Nodes with no directed out-edges and no undirected/bidirected/circle
    incidence (strict)."""
    _, endpoints, _ = _to_endpoints(g)
    n_vars = endpoints.shape[0]
    count = 0
    for v in range(n_vars):
        col = endpoints[:, v]
        row = endpoints[v, :]
        has_directed_child = bool(np.any((row == ARROW) & (col == TAIL)))
        has_undirected = bool(
            np.any((col == TAIL) & (row == TAIL))
            | np.any((col == ARROW) & (row == ARROW))
            | np.any((col == CIRCLE) | (row == CIRCLE))
        )
        if not has_directed_child and not has_undirected:
            count += 1
    return count


def count_isolated_nodes(g: object) -> int:
    """Nodes with no edges of any kind."""
    _, endpoints, _ = _to_endpoints(g)
    n_vars = endpoints.shape[0]
    count = 0
    for v in range(n_vars):
        if not (np.any(endpoints[:, v] != NO_EDGE) or np.any(endpoints[v, :] != NO_EDGE)):
            count += 1
    return count


def count_reversible_arcs(g: object) -> int:
    """Directed arcs not part of any unshielded collider.

    A directed arc ``u → v`` is reversible iff `v` is not the apex of an
    unshielded collider (`v` doesn't have two non-adjacent parents).
    Matches 0.1.x's `count_reversible_arcs`.
    """
    _, endpoints, _ = _to_endpoints(g)
    n_vars = endpoints.shape[0]
    collider_apexes: set[int] = set()
    for v in range(n_vars):
        parents = _directed_parents(endpoints, v)
        if len(parents) < 2:
            continue
        for k in range(len(parents)):
            for m in range(k + 1, len(parents)):
                u, w = int(parents[k]), int(parents[m])
                if not _directed_adjacent(endpoints, u, w):
                    collider_apexes.add(v)
                    break
            else:
                continue
            break

    count = 0
    for v in range(n_vars):
        if v in collider_apexes:
            continue
        # Number of directed arcs ending at v.
        col = endpoints[:, v]
        row = endpoints[v, :]
        count += int(np.sum((col == ARROW) & (row == TAIL)))
    return count


# ---- per-node ----------------------------------------------------------


def in_degree(g: object, var: int | str) -> int:
    """Number of directed in-edges to `var`."""
    n_vars, endpoints, var_names = _to_endpoints(g)
    v = _resolve_var(var, var_names, n_vars)
    return int(len(_directed_parents(endpoints, v)))


def out_degree(g: object, var: int | str) -> int:
    """Number of directed out-edges from `var`."""
    n_vars, endpoints, var_names = _to_endpoints(g)
    v = _resolve_var(var, var_names, n_vars)
    return int(len(_directed_children(endpoints, v)))
