"""Structural Intervention Distance (Peters & Bühlmann 2015).

Port of bnm 0.1.x's `sid_metric` (`scripts/legacy_0_1_x/sid.py`) onto
the int8 endpoint matrix natively. The algorithmic content is faithful
to the R `SID` package by Jonas Peters; this module fixes two 0.1.x
bugs (audit §1, §6) and replaces the nx.DiGraph round-trips with
direct integer-index operations on the canonical endpoint matrix.

References:
    Peters & Bühlmann, "Structural Intervention Distance for Evaluating
    Causal Graphs," 2015. https://doi.org/10.48550/arXiv.1306.1043

Semantic requirements:

- ``g1`` (true graph) MUST be a pure DAG: every edge directed
  (TAIL→ARROW), no CIRCLEs, no undirected, no bidirected. Acyclicity
  is the caller's responsibility — same contract as 0.1.x; checking is
  O(V+E) and adds noise to fixture-heavy test loops.
- ``g2`` (estimated graph) may be a DAG or CPDAG (TAIL/TAIL allowed
  for the undirected components). CIRCLE marks and bidirected (ARROW/
  ARROW) edges are rejected.

Bug fixes from 0.1.x:

- **§1 (empty possible parents).** When the estimated graph has a node
  with zero possible-parent candidates (no incoming TAIL/TAIL
  undirected edges) AND `gp_is_essential_graph` is False, 0.1.x crashes
  on ``np.meshgrid(*[[0,1]] * 0).T.reshape(-1, 0)``. Fixed by
  special-casing the zero-candidate path to a single trivial assignment.
- **§6 (hash-seed non-determinism).** 0.1.x's
  `get_undirected_components_with_isolates` walks `set(G.nodes())`,
  whose iteration order depends on PYTHONHASHSEED. The component
  ordering then propagates into `gp_is_essential_graph` checks and
  `incorrect_sum` accumulation. Replaced with a deterministic
  index-sorted union-find component finder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bnm.adapter import _to_endpoints
from bnm.exceptions import BNMDataError, BNMInputError
from bnm.marks import EndpointMark

NO_EDGE = int(EndpointMark.NO_EDGE)
TAIL = int(EndpointMark.TAIL)
ARROW = int(EndpointMark.ARROW)
CIRCLE = int(EndpointMark.CIRCLE)


# ---- result dataclass --------------------------------------------------


@dataclass(frozen=True, slots=True)
class SIDResult:
    """Structural Intervention Distance result.

    Mirrors the dict shape of bnm 0.1.x's `sid()` return for backwards
    interpretation, but as a typed frozen dataclass.
    """

    sid: int
    """Total mis-classified intervention pairs."""

    sid_lower_bound: int
    """Lower bound on SID across CPDAG ambiguity (= ``sid`` when g2 is a DAG)."""

    sid_upper_bound: int
    """Upper bound on SID across CPDAG ambiguity (= ``sid`` when g2 is a DAG)."""

    incorrect_mat: NDArray[np.int8]
    """``(n, n)`` binary matrix marking ``(i, j)`` pairs whose
    intervention prediction differs between g1 and g2."""

    @property
    def is_tight(self) -> bool:
        """``True`` when lower == upper (no Markov-equivalence ambiguity)."""
        return self.sid_lower_bound == self.sid_upper_bound


# ---- input validation --------------------------------------------------


def _require_dag(endpoints: NDArray[np.int8], *, source: str) -> None:
    """Validate that every edge is a pure directed arc (TAIL→ARROW only)."""
    n = endpoints.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            mij, mji = int(endpoints[i, j]), int(endpoints[j, i])
            if mij == NO_EDGE:
                continue
            forward = mij == ARROW and mji == TAIL
            backward = mij == TAIL and mji == ARROW
            if not (forward or backward):
                raise BNMInputError(
                    f"sid: {source} must be a pure DAG (TAIL→ARROW edges only); "
                    f"edge ({i}, {j}) has marks ({mji}, {mij})"
                )


def _require_dag_or_cpdag(endpoints: NDArray[np.int8], *, source: str) -> None:
    """Validate g2: directed (TAIL/ARROW) or undirected (TAIL/TAIL).
    No CIRCLE, no bidirected (ARROW/ARROW)."""
    n = endpoints.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            mij, mji = int(endpoints[i, j]), int(endpoints[j, i])
            if mij == NO_EDGE:
                continue
            ok = (
                (mij == ARROW and mji == TAIL)
                or (mij == TAIL and mji == ARROW)
                or (mij == TAIL and mji == TAIL)
            )
            if not ok:
                raise BNMInputError(
                    f"sid: {source} must be a DAG or CPDAG (no CIRCLE marks, no "
                    f"bidirected edges); edge ({i}, {j}) has marks ({mji}, {mij})"
                )


# ---- adjacency-matrix conversion --------------------------------------


def _to_sid_adj(endpoints: NDArray[np.int8]) -> NDArray[np.int_]:
    """Convert an int8 endpoint matrix to the SID-internal adjacency form
    where ``adj[i, j] == 1`` iff i may be a parent of j (i.e. directed
    i→j or undirected i—j).

    For DAG/CPDAG inputs only (callers must validate first).
    """
    n = endpoints.shape[0]
    adj = np.zeros((n, n), dtype=np.int_)
    # Directed i→j: endpoints[i, j] == ARROW (mark at j is arrowhead).
    adj[endpoints == ARROW] = 1
    # Undirected i—j: both endpoints TAIL → both adj entries 1.
    undirected = (endpoints == TAIL) & (endpoints.T == TAIL)
    adj[undirected] = 1
    np.fill_diagonal(adj, 0)
    return adj


# ---- deterministic undirected-component finder (audit §6 fix) ----------


def _undirected_components(endpoints: NDArray[np.int8]) -> list[tuple[int, ...]]:
    """Return the connected components induced by undirected (TAIL/TAIL)
    edges. Each component is a sorted tuple of node indices; components
    are returned in ascending order of their smallest element. Singletons
    (nodes with no undirected incidence) are returned as length-1
    components.

    Replaces 0.1.x's `get_undirected_components_with_isolates` which
    used `set(G.nodes())` and was hash-seed-dependent.
    """
    n = endpoints.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(n):
        for j in range(i + 1, n):
            if endpoints[i, j] == TAIL and endpoints[j, i] == TAIL:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for v in range(n):
        groups.setdefault(find(v), []).append(v)

    return [tuple(sorted(members)) for _, members in sorted(groups.items())]


# ---- reachability primitives (mechanical port of 0.1.x sid.py) --------


def _compute_path_matrix(adj_mat: NDArray[np.int_]) -> NDArray[np.int_]:
    """Repeated-squaring reachability matrix. ``out[i, j] == 1`` iff
    ``j`` is reachable from ``i`` via a directed path of any length
    (or ``i == j``). O(n³ log n) but n is small in practice."""
    p = adj_mat.shape[1]
    if p == 0:
        return adj_mat.copy()
    path_matrix = np.eye(p, dtype=np.int_) + adj_mat
    k = max(int(np.ceil(np.log(max(p, 2)) / np.log(2))), 1)
    for _ in range(k):
        path_matrix = np.dot(path_matrix, path_matrix)
    return (path_matrix > 0).astype(np.int_)


def _compute_path_matrix2(
    adj_mat: NDArray[np.int_],
    cond_set: NDArray[np.int_],
    path_matrix1: NDArray[np.int_] | None,
) -> NDArray[np.int_]:
    """Reachability after removing all outgoing edges from `cond_set`."""
    if len(cond_set) == 0:
        assert path_matrix1 is not None
        return path_matrix1
    g_mod = adj_mat.copy()
    g_mod[cond_set, :] = 0
    return _compute_path_matrix(g_mod)


def _dsepadj(
    a_mat: NDArray[np.int_],
    i: int,
    cond_set: NDArray[np.int_],
) -> dict[str, NDArray[np.bool_]]:
    """Return reachability flags from node `i` under conditioning on `cond_set`.

    Mechanical port of 0.1.x's `dsepadj`. Two boolean arrays:
      - reachable_nodes[k]: True iff k is reachable from i under
        d-separation w.r.t. cond_set.
      - reachable_on_non_causal_path[k]: True iff k is reachable from i
        via a non-causal path (i.e. through a path that backs into i's
        ancestors).
    """
    adj_mat = a_mat.copy()
    path_matrix = _compute_path_matrix(adj_mat)
    path_matrix2 = _compute_path_matrix2(adj_mat, cond_set, path_matrix)

    if len(cond_set) == 0:
        anc_of_cond_set = np.array([], dtype=int)
    elif len(cond_set) == 1:
        anc_of_cond_set = np.where(path_matrix[:, cond_set[0]] > 0)[0]
    else:
        anc_of_cond_set = np.where(np.sum(path_matrix[:, cond_set], axis=1) > 0)[0]

    p = adj_mat.shape[1]
    reachability_matrix = np.zeros((2 * p, 2 * p), dtype=np.int_)
    reachable_on_non_causal_path_later = np.zeros((2, 2), dtype=np.int_)
    reachable_nodes = np.zeros(2 * p, dtype=np.int_)
    reachable_on_non_causal_path = np.zeros(2 * p, dtype=np.int_)
    already_checked = np.zeros(p, dtype=np.int_)
    to_check: list[int] = []

    reachable_ch = np.where(adj_mat[i, :] == 1)[0]
    if len(reachable_ch) > 0:
        to_check.extend(reachable_ch.tolist())
        reachable_nodes[reachable_ch] = 1
        adj_mat[i, reachable_ch] = 0

    reachable_pa = np.where(adj_mat[:, i] == 1)[0]
    if len(reachable_pa) > 0:
        to_check.extend(reachable_pa.tolist())
        reachable_nodes[reachable_pa + p] = 1
        reachable_on_non_causal_path[reachable_pa + p] = 1
        adj_mat[reachable_pa, i] = 0

    k = -1
    while k < len(to_check) - 1:
        k += 1
        a1 = to_check[k]

        if not already_checked[a1]:
            current_node = a1
            already_checked[a1] = a1

            parents = np.where(adj_mat[:, current_node] == 1)[0]
            pa1 = np.setdiff1d(parents, cond_set)
            reachability_matrix[pa1, current_node] = 1
            reachability_matrix[pa1 + p, current_node] = 1

            if current_node in anc_of_cond_set:
                reachability_matrix[current_node, parents + p] = 1
                if path_matrix2[i, current_node] > 0:
                    new_rows = np.column_stack((np.full(len(parents), current_node), parents))
                    reachable_on_non_causal_path_later = np.vstack(
                        (reachable_on_non_causal_path_later, new_rows)
                    )
                new_to_check = np.setdiff1d(parents, np.where(already_checked)[0])
                to_check.extend(new_to_check.tolist())

            if current_node not in cond_set:
                reachability_matrix[current_node + p, parents + p] = 1
                new_to_check = np.setdiff1d(parents, np.where(already_checked)[0])
                to_check.extend(new_to_check.tolist())

            children = np.where(adj_mat[current_node, :] == 1)[0]
            ch1 = np.setdiff1d(children, cond_set)
            reachability_matrix[ch1 + p, current_node + p] = 1

            ch2 = np.intersect1d(children, anc_of_cond_set)
            reachability_matrix[ch2, current_node + p] = 1
            ch2b = np.intersect1d(ch2, np.where(path_matrix2[i, :] > 0)[0])
            if len(ch2b) > 0:
                new_rows = np.column_stack((ch2b, np.full(len(ch2b), current_node)))
                reachable_on_non_causal_path_later = np.vstack(
                    (reachable_on_non_causal_path_later, new_rows)
                )
            if current_node not in cond_set:
                reachability_matrix[current_node, children] = 1
                reachability_matrix[current_node + p, children] = 1
                new_to_check = np.setdiff1d(children, np.where(already_checked)[0])
                to_check.extend(new_to_check.tolist())

    reachability_matrix = _compute_path_matrix(reachability_matrix)
    ttt2 = np.where(reachable_nodes == 1)[0]
    if len(ttt2) == 1:
        tt2 = np.where(reachability_matrix[ttt2[0], :] > 0)[0]
    else:
        tt2 = np.where(np.sum(reachability_matrix[ttt2, :], axis=0) > 0)[0]
    reachable_nodes[tt2] = 1

    ttt = np.where(reachable_on_non_causal_path == 1)[0]
    if len(ttt) == 1:
        tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
    else:
        tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]
    reachable_on_non_causal_path[tt] = 1

    if reachable_on_non_causal_path_later.shape[0] > 2:
        for kk in range(2, reachable_on_non_causal_path_later.shape[0]):
            reachable_through = reachable_on_non_causal_path_later[kk, 0]
            new_reachable = reachable_on_non_causal_path_later[kk, 1]
            reachable_on_non_causal_path[new_reachable + p] = 1
            reachability_matrix[new_reachable, reachable_through] = 0
            reachability_matrix[new_reachable, reachable_through + p] = 0
            reachability_matrix[new_reachable + p, reachable_through] = 0
            reachability_matrix[new_reachable + p, reachable_through + p] = 0

        ttt = np.where(reachable_on_non_causal_path == 1)[0]
        if len(ttt) == 1:
            tt = np.where(reachability_matrix[ttt[0], :] > 0)[0]
        else:
            tt = np.where(np.sum(reachability_matrix[ttt, :], axis=0) > 0)[0]
        reachable_on_non_causal_path[tt] = 1

    return {
        "reachable_nodes": np.sum(
            np.column_stack((reachable_nodes[:p], reachable_nodes[p:])), axis=1
        )
        > 0,
        "reachable_on_non_causal_path": np.sum(
            np.column_stack((reachable_on_non_causal_path[:p], reachable_on_non_causal_path[p:])),
            axis=1,
        )
        > 0,
    }


# ---- DAG-extension enumeration ----------------------------------------


def _all_dags_intern(
    adj_full: NDArray[np.int_],
    sub_adj: NDArray[np.int_],
    node_indices: NDArray[np.int_],
    tmp: list[NDArray[np.int_]] | None = None,
) -> list[NDArray[np.int_]]:
    """Recursively extend an undirected component to all v-structure-free DAGs.

    Mechanical port of 0.1.x's `all_dags_intern`. The recursion picks a
    candidate sink whose neighbours form a clique, orients all of its
    incident undirected edges toward it, removes it from the component,
    and recurses. Returns flattened DAG adjacency arrays.
    """
    if tmp is None:
        tmp = []
    if np.any((sub_adj + sub_adj.T) == 1):
        raise ValueError("Submatrix is not fully undirected")

    if np.sum(sub_adj) == 0:
        flat = adj_full.copy().flatten()
        if not any(np.array_equal(flat, t) for t in tmp):
            tmp.append(flat)
        return tmp

    candidate_sinks = np.where(sub_adj.sum(axis=0) > 0)[0]

    for sink_local_idx in candidate_sinks:
        neighbours_local_idx = np.where(sub_adj[sink_local_idx] == 1)[0]
        if neighbours_local_idx.size > 0:
            subgraph_neighbours = sub_adj[np.ix_(neighbours_local_idx, neighbours_local_idx)]
            if not np.all(subgraph_neighbours + np.eye(len(neighbours_local_idx))):
                continue

        adj_new = adj_full.copy()
        sink_global_idx = node_indices[sink_local_idx]
        for nbh_local in neighbours_local_idx:
            nbh_global = node_indices[nbh_local]
            adj_new[nbh_global, sink_global_idx] = 1
            adj_new[sink_global_idx, nbh_global] = 0

        mask = np.ones(len(sub_adj), dtype=bool)
        mask[sink_local_idx] = False
        new_sub_adj = sub_adj[np.ix_(mask, mask)]
        new_node_indices = node_indices[mask]

        tmp = _all_dags_intern(adj_new, new_sub_adj, new_node_indices, tmp)

    return tmp


def _all_dags(
    adj_matrix: NDArray[np.int_],
    node_indices: list[int],
) -> NDArray[np.int_] | int:
    """Wrapper. Returns -1 if the component isn't fully undirected,
    otherwise a 2D array of flattened DAG adjacencies."""
    sub_adj = adj_matrix[np.ix_(node_indices, node_indices)]
    if np.any((sub_adj + sub_adj.T) == 1):
        return -1
    dags = _all_dags_intern(adj_matrix.copy(), sub_adj, np.array(node_indices), tmp=None)
    return np.array(dags) if dags else np.array([])


# ---- chordal-component test (replaces nx.is_chordal w/ deterministic
#      MCS algorithm operating on the subgraph index space) -------------


def _is_chordal_subgraph(adj: NDArray[np.int_], indices: tuple[int, ...]) -> bool:
    """Maximum-cardinality search chordality test on the undirected
    subgraph induced by `indices`. Edges in the subgraph are
    `adj[i, j] == 1 AND adj[j, i] == 1` for i, j in indices.

    Replaces nx.is_chordal so SID has no networkx dependency and runs
    deterministically. Algorithm: Tarjan & Yannakakis (1984)
    perfect-elimination ordering via MCS, then chord test.
    """
    n = len(indices)
    if n <= 2:
        return True
    nbh: list[list[int]] = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            i, j = indices[a], indices[b]
            if adj[i, j] == 1 and adj[j, i] == 1:
                nbh[a].append(b)
                nbh[b].append(a)

    # MCS: pick the node with the most labelled neighbours; tie-break
    # by smallest local index for determinism.
    label = [0] * n
    visited = [False] * n
    order: list[int] = []
    for _ in range(n):
        candidate = -1
        best_label = -1
        for v in range(n):
            if not visited[v] and label[v] > best_label:
                best_label = label[v]
                candidate = v
        visited[candidate] = True
        order.append(candidate)
        for u in nbh[candidate]:
            if not visited[u]:
                label[u] += 1

    # Perfect-elimination ordering check (Tarjan-Yannakakis 1984): MCS
    # order's reverse is the elimination order. Equivalently: for each
    # `v`, its EARLIER-in-MCS neighbours (= LATER-in-elimination) must
    # form a clique.
    pos = {v: k for k, v in enumerate(order)}
    nbh_set = [set(adj) for adj in nbh]
    for v in order:
        earlier = [u for u in nbh[v] if pos[u] < pos[v]]
        for a in range(len(earlier)):
            for b in range(a + 1, len(earlier)):
                if earlier[b] not in nbh_set[earlier[a]]:
                    return False
    return True


# ---- main entry point -------------------------------------------------


def sid(g1: object, g2: object) -> SIDResult:
    """Compute the Structural Intervention Distance between true DAG
    `g1` and estimated DAG/CPDAG `g2`.

    Args:
        g1: Reference graph. MUST be a pure DAG (TAIL→ARROW edges only).
        g2: Estimated graph. May be a DAG or CPDAG (TAIL/TAIL allowed).

    Returns:
        :class:`SIDResult` with `sid`, `sid_lower_bound`,
        `sid_upper_bound`, `incorrect_mat`.

    Raises:
        BNMInputError: g1 has non-directed edges, g2 has CIRCLE or
        bidirected marks, or n_vars mismatch.
        BNMDataError: var_names disagree.
    """
    n1, ep1, names1 = _to_endpoints(g1)
    n2, ep2, names2 = _to_endpoints(g2)
    if n1 != n2:
        raise BNMInputError(f"sid: g1 has {n1} variables, g2 has {n2}; must match")
    if names1 is not None and names2 is not None and names1 != names2:
        raise BNMDataError(
            "sid: g1.var_names and g2.var_names differ; alignment by positional "
            "index is unsafe when names disagree"
        )
    _require_dag(ep1, source="g1")
    _require_dag_or_cpdag(ep2, source="g2")

    adj1 = _to_sid_adj(ep1)
    adj2 = _to_sid_adj(ep2)
    p = n1
    if p == 0:
        return SIDResult(0, 0, 0, np.zeros((0, 0), dtype=np.int8))

    incorrect_int = np.zeros(adj1.shape, dtype=np.int_)
    correct_int = np.zeros(adj1.shape, dtype=np.int_)
    min_total = 0
    max_total = 0
    path_matrix = _compute_path_matrix(adj1)
    conn_comp = _undirected_components(ep2)

    gp_is_essential_graph = True
    for comp in conn_comp:
        if len(comp) > 1 and not _is_chordal_subgraph(adj2, comp):
            gp_is_essential_graph = False
            break

    mmm: NDArray[np.int_] = np.array([], dtype=np.int_)
    incorrect_sum: NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    for comp in conn_comp:
        node_indices = list(comp)
        if len(node_indices) == 0:
            continue

        if gp_is_essential_graph:
            if len(node_indices) > 1:
                result = _all_dags(adj2, node_indices)
                if (
                    isinstance(result, int)
                    and result == -1
                    or isinstance(result, np.ndarray)
                    and result.size == 0
                ):
                    gp_is_essential_graph = False
                    mmm = np.array([adj2.copy().flatten()])
                else:
                    mmm = result  # type: ignore[assignment]
            else:
                mmm = np.array([adj2.copy().flatten()])
            incorrect_sum = np.zeros(len(mmm), dtype=np.float64)

        for i in node_indices:
            certain_pa_gp = np.where((adj2[:, i] * (1 - adj2[i, :])) == 1)[0]
            possible_pa_gp = np.where((adj2[:, i] * adj2[i, :]) == 1)[0]
            pa_g = np.where(adj1[:, i] == 1)[0]

            if not gp_is_essential_graph:
                # AUDIT §1 FIX: special-case zero-candidate path so
                # np.meshgrid(*[]) doesn't blow up.
                if len(possible_pa_gp) == 0:
                    mmm = np.array([adj2.T.flatten()])
                    unique_rows = np.array([0])
                    maxcount = 1
                    incorrect_sum = np.zeros(1, dtype=np.float64)
                    all_parents_of_i: list[int] = []
                else:
                    maxcount = 2 ** len(possible_pa_gp)
                    unique_rows = np.arange(maxcount)
                    base_flat = adj2.T.flatten()
                    mmm = np.tile(base_flat, (maxcount, 1))
                    grid = np.array(np.meshgrid(*[[0, 1]] * len(possible_pa_gp))).T.reshape(
                        -1, len(possible_pa_gp)
                    )
                    for row_idx, combo in enumerate(grid):
                        for j_local, val in enumerate(combo):
                            mmm[row_idx, i + possible_pa_gp[j_local] * p] = val
                    incorrect_sum = np.zeros(maxcount, dtype=np.float64)
                    all_parents_of_i = [i + k * p for k in range(p)]
            else:
                if len(mmm) > 1:
                    all_parents_of_i = [i + k * p for k in range(p)]
                    parent_sets = mmm[:, all_parents_of_i]
                    _, unique_indices = np.unique(parent_sets, axis=0, return_index=True)
                    unique_rows = np.sort(unique_indices)
                    maxcount = len(unique_rows)
                else:
                    maxcount = 1
                    unique_rows = np.array([0])
                    all_parents_of_i = [i + k * p for k in range(p)]

            count = 0
            while count < maxcount:
                if maxcount == 1:
                    pa_gp = certain_pa_gp
                else:
                    gp_new = mmm[unique_rows[count], :].reshape((p, p))
                    pa_gp = np.where(gp_new[:, i] == 1)[0]

                check_d_sep = _dsepadj(adj1, i, pa_gp)
                reachable_w_out_causal_path = check_d_sep["reachable_on_non_causal_path"]

                for j in range(p):
                    if i == j:
                        continue
                    finished = False
                    ij_g_null = path_matrix[i, j] == 0
                    ij_gp_null = j in pa_gp

                    if ij_g_null and ij_gp_null:
                        finished = True
                        correct_int[i, j] = 1

                    if ij_gp_null and not ij_g_null:
                        incorrect_int[i, j] = 1
                        incorrect_sum[unique_rows[count]] += 1
                        finished = True

                    if not finished and set(pa_gp) == set(pa_g):
                        finished = True
                        correct_int[i, j] = 1

                    if not finished:
                        if path_matrix[i, j] > 0:
                            chi_caus_path = np.where(adj1[i, :] & path_matrix[:, j])[0]
                            if len(chi_caus_path) > 0 and len(pa_gp) > 0:
                                submatrix = path_matrix[np.ix_(chi_caus_path, pa_gp)]
                                if np.sum(submatrix) > 0:
                                    incorrect_int[i, j] = 1
                                    incorrect_sum[unique_rows[count]] += 1
                                    finished = True

                        if not finished:
                            if reachable_w_out_causal_path[j] == 1:
                                incorrect_int[i, j] = 1
                                incorrect_sum[unique_rows[count]] += 1
                            else:
                                correct_int[i, j] = 1
                count += 1

            if not gp_is_essential_graph:
                min_total += int(np.min(incorrect_sum))
                max_total += int(np.max(incorrect_sum))
                incorrect_sum = np.zeros_like(incorrect_sum)

        if gp_is_essential_graph:
            if len(incorrect_sum) > 0:
                min_total += int(np.min(incorrect_sum))
                max_total += int(np.max(incorrect_sum))
            incorrect_sum = (
                np.zeros_like(incorrect_sum) if len(incorrect_sum) > 0 else incorrect_sum
            )

    return SIDResult(
        sid=int(np.sum(incorrect_int)),
        sid_lower_bound=int(min_total),
        sid_upper_bound=int(max_total),
        incorrect_mat=incorrect_int.astype(np.int8),
    )
