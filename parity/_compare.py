"""Shared utilities for parity comparisons against causal-learn / tigramite.

These scripts live OUTSIDE the cbcd package and are not part of the wheel.
They depend on causal-learn and tigramite — install separately:

    uv pip install causal-learn tigramite

The cbcd package itself does NOT depend on either.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- causal-learn endpoint mapping -----------------------------------------
#
# causal-learn / Tetrad: graph[i, j] = mark at vertex i of edge {i, j}:
#     0 = no edge,  1 = ARROW at i,  -1 = TAIL at i,  2 = CIRCLE at i
# cbcd: endpoints[i, j] = mark at vertex j of edge {i, j}:
#     0 = NO_EDGE,  1 = TAIL at j,   2 = ARROW at j,    3 = CIRCLE at j
#
# So cl_graph[i, j] corresponds to cbcd.endpoints[j, i].

_CL_TO_CBCD = {0: 0, -1: 1, 1: 2, 2: 3}


def cl_graph_to_cbcd_endpoints(cl_graph: NDArray) -> NDArray:
    """Convert a causal-learn graph matrix to a cbcd endpoint matrix."""
    n = cl_graph.shape[0]
    ep = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            mark = int(cl_graph[i, j])
            if mark == 0:
                continue
            ep[j, i] = _CL_TO_CBCD[mark]
    return ep


# --- tigramite mapping -----------------------------------------------------


def tigramite_graph_to_cbcd_endpoints(tg_graph: NDArray) -> NDArray:
    """Convert a tigramite ``results['graph']`` array to a cbcd time-series
    endpoint array.

    tigramite shape: ``(n_vars, n_vars, max_lag + 1)``, dtype string.
        ``tg_graph[i, j, tau]`` describes the edge from ``i`` (at lag -tau)
        to ``j`` (at lag 0). Symbols:
            ''     -> no edge
            '-->'  -> directed, src to dst
            '<--'  -> directed, dst to src (only for lag-0)
            'o-o'  -> circle-circle
            'o->'  -> circle at src, arrow at dst
            '<-o'  -> arrow at src, circle at dst (lag-0)
            'x-x', 'x->', etc. — bidirected variants
    cbcd shape: ``(max_lag + 1, n_vars, n_vars)``, dtype int8.
        ``endpoints[tau, i, j]`` = mark at vertex ``j`` (present-time end)
        of the lagged edge ``i_{t-tau} → j_t``.
    """
    n_vars, _, max_lag_plus_one = tg_graph.shape
    ep = np.zeros((max_lag_plus_one, n_vars, n_vars), dtype=np.int8)
    for tau in range(1, max_lag_plus_one):
        for i in range(n_vars):
            for j in range(n_vars):
                s = str(tg_graph[i, j, tau])
                if s == "":
                    continue
                if s == "-->":
                    ep[tau, i, j] = 2  # ARROW at j
    # Lag-0 (contemporaneous) not handled — vanilla PCMCI does not produce
    # contemporaneous edges; PCMCI+ extension required.
    return ep


# --- comparison helpers ----------------------------------------------------


def shd_endpoints(a: NDArray, b: NDArray) -> int:
    return int(np.sum(a != b))


def adjacency_iid(endpoints: NDArray) -> NDArray:
    return (endpoints != 0).astype(np.int8)


def adjacency_lagged(endpoints: NDArray) -> NDArray:
    return (endpoints != 0).astype(np.int8)


# --- Linear-Gaussian SCM simulator ----------------------------------------


def simulate_linear_gaussian(
    cbcd_dag,
    n: int,
    *,
    noise_scale: float = 0.5,
    edge_coef: float = 0.7,
    seed: int = 0,
) -> NDArray:
    """Simulate ``n`` rows from a linear-Gaussian SCM with the given DAG.

    Each non-root variable is the sum of (edge_coef × parent) plus
    independent Gaussian noise.
    """
    from cbcd.graph import EndpointMark

    rng = np.random.default_rng(seed)
    p = cbcd_dag.n_vars
    # Build child adjacency in topological order.
    parents_of: dict[int, list[int]] = {i: [] for i in range(p)}
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if (
                cbcd_dag.endpoints[j, i] == EndpointMark.ARROW
                and cbcd_dag.endpoints[i, j] == EndpointMark.TAIL
            ):
                parents_of[i].append(j)
    topo = _topo_order(p, parents_of)
    data = np.empty((n, p), dtype=np.float64)
    for v in topo:
        noise = rng.normal(scale=noise_scale, size=n)
        if not parents_of[v]:
            data[:, v] = noise
        else:
            contrib = sum(edge_coef * data[:, par] for par in parents_of[v])
            data[:, v] = contrib + noise
    return data


def _topo_order(n: int, parents_of: dict[int, list[int]]) -> list[int]:
    indeg = {i: len(parents_of[i]) for i in range(n)}
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for v, ps in parents_of.items():
        for p in ps:
            children[p].append(v)
    order: list[int] = []
    stack = [i for i in range(n) if indeg[i] == 0]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in children[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    return order


def simulate_var(
    edges: list[tuple[int, int, int, float]],
    n_vars: int,
    T: int,
    *,
    noise_scale: float = 0.5,
    seed: int = 0,
) -> NDArray:
    """Simulate from a VAR process. ``edges`` is a list of
    ``(src_var, dst_var, tau, coef)`` quadruples, meaning
    ``X_{src,t-tau} → X_{dst,t}`` with the given coefficient.
    """
    rng = np.random.default_rng(seed)
    max_lag = max((tau for _, _, tau, _ in edges), default=1)
    burn = max_lag + 50
    data = np.zeros((T + burn, n_vars), dtype=np.float64)
    # Group incoming edges per dst.
    incoming: dict[int, list[tuple[int, int, float]]] = {i: [] for i in range(n_vars)}
    for src, dst, tau, coef in edges:
        incoming[dst].append((src, tau, coef))
    for t in range(max_lag, T + burn):
        for v in range(n_vars):
            val = rng.normal(scale=noise_scale)
            for src, tau, coef in incoming[v]:
                val += coef * data[t - tau, src]
            data[t, v] = val
    return data[burn:]
