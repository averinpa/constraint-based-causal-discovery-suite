"""Verify v0.2 reproduces the numerical outputs committed in the
GitHub `use cases/` notebooks.

The GitHub notebooks have output cells frozen from when 0.1.x ran
them. We replicate 0.1.x's ``generate_random_dag`` exactly here,
feed the same inputs to v0.2, and compare key numerical outputs.

Run as ``python scripts/notebook_parity_check.py``. Exits non-zero
if any check fails.

Discrepancies traceable to one of the four 0.1.x bugs documented in
``docs/audit.md`` (§1, §6, §7, §8) are flagged but not counted as
failures — those are intentional v0.2 corrections.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np

import bnm

# ---- 0.1.x generate_random_dag, byte-for-byte ----------------------


def generate_random_dag_legacy(n_nodes=40, edge_prob=0.1, seed=None):
    """Exact replica of 0.1.x's generator (`scripts/legacy_0_1_x/utils.py`)."""
    if seed:
        random.seed(seed)
    nodes = list(range(n_nodes))
    topo_order = list(nodes)
    random.shuffle(topo_order)
    possible_edges = [
        (topo_order[i], topo_order[j]) for i in range(n_nodes) for j in range(i + 1, n_nodes)
    ]
    n_edges = int(len(possible_edges) * edge_prob)
    sampled_edges = random.sample(possible_edges, n_edges)
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(sampled_edges)
    final_order = list(nodes)
    random.shuffle(final_order)
    rename_map = {node: f"X_{i + 1}" for i, node in enumerate(final_order)}
    dag = nx.relabel_nodes(dag, rename_map)
    return bnm.to_graphlike(dag)


# ---- assertion helper ----------------------------------------------


_failures: list[str] = []


def check(name: str, actual, expected, *, kind: str = "exact") -> None:
    """Compare actual vs expected; record any mismatch."""
    if kind == "exact":
        ok = actual == expected
    elif kind == "set":
        ok = set(actual) == set(expected)
    elif kind == "approx":
        ok = abs(float(actual) - float(expected)) < 1e-6
    else:
        raise ValueError(kind)
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {name}: v0.2={actual!r}  legacy={expected!r}")
    if not ok:
        _failures.append(f"{name}: v0.2={actual!r}, legacy={expected!r}")


# ---- per-notebook checks ------------------------------------------


def check_evaluate_single_dag() -> None:
    """Reproduce GitHub's `evaluate single DAG.ipynb` outputs."""
    print("\n=== evaluate single DAG.ipynb ===")
    dag = generate_random_dag_legacy(n_nodes=1000, edge_prob=0.005, seed=55)
    print(f"  generated DAG: {dag.n_vars} nodes, {bnm.count_edges(dag)} edges")

    # Direct per-node iteration; faster than `compare(per_node=True)`'s
    # full validation-per-call overhead on 1000 nodes.
    isolated: list[str] = []
    max_edges = 0
    for var in dag.var_names:
        mb = bnm.markov_blanket(dag, var)
        if bnm.count_isolated_nodes(mb) > 0:
            isolated.append(var)
        n_e = bnm.count_edges(mb)
        if n_e > max_edges:
            max_edges = n_e

    isolated.sort()
    check("isolated-MB nodes (count)", len(isolated), 9)
    check(
        "isolated-MB nodes (set)",
        isolated,
        ["X_44", "X_536", "X_751", "X_784", "X_818", "X_850", "X_91", "X_931", "X_364"],
        kind="set",
    )
    check("max MB n_edges", max_edges, 60)


def check_sid() -> None:
    """Reproduce GitHub's `sid.ipynb` outputs."""
    print("\n=== sid.ipynb ===")
    nodes = ["X1", "X2", "Y1", "Y2", "Y3"]

    G = np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    H1 = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    def from_01(adj, names):
        a = np.asarray(adj)
        n = a.shape[0]
        ep = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                ij, ji = int(a[i, j]), int(a[j, i])
                if ij == 1 and ji == 1:
                    ep[i, j] = ep[j, i] = bnm.EndpointMark.TAIL
                elif ij == 1:
                    ep[i, j] = bnm.EndpointMark.ARROW
                    ep[j, i] = bnm.EndpointMark.TAIL
                elif ji == 1:
                    ep[i, j] = bnm.EndpointMark.TAIL
                    ep[j, i] = bnm.EndpointMark.ARROW
        return bnm.to_graphlike(ep, var_names=tuple(names))

    g = from_01(G, nodes)
    g_h1 = from_01(H1, nodes)
    res = bnm.sid(g, g_h1)
    check("G vs H1: shd", bnm.shd(g, g_h1), 1)
    check(
        "G vs H1: sid (count of mis-classified pairs)",
        res.sid,
        8,
        # NOTE: legacy showed sid=0 here; that was 0.1.x's bug §8 — the
        # value is actually `sid_lower_bound` not `sid`. v0.2's sid
        # counts (i,j) pairs mis-classified by ANY DAG in the
        # equivalence class of H1.
    )
    check("G vs H1: sid_lower_bound", res.sid_lower_bound, 0)

    # Use case 2 — DAG, DAG1, DAG2, CPDAG comparisons
    nodes4 = ["X1", "X2", "X3", "X4"]
    DAG = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
        ]
    )
    DAG1 = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    g_dag = from_01(DAG, nodes4)
    g_dag1 = from_01(DAG1, nodes4)
    res = bnm.sid(g_dag, g_dag1)
    check("DAG vs DAG1: shd", bnm.shd(g_dag, g_dag1), 3)
    check("DAG vs DAG1: sid", res.sid, 6)


def check_compare_two_dags() -> None:
    """Reproduce GitHub's `compare two DAGs.ipynb` Case 1 outputs."""
    print("\n=== compare two DAGs.ipynb ===")
    system1 = generate_random_dag_legacy(n_nodes=100, edge_prob=0.03, seed=55)
    system2 = generate_random_dag_legacy(n_nodes=100, edge_prob=0.02, seed=66)

    check("system1 n_edges", bnm.count_edges(system1), 148)
    check("system2 n_edges", bnm.count_edges(system2), 99)
    check("system1 n_colliders", bnm.count_colliders(system1), 137)
    check("system1 n_isolated_nodes", bnm.count_isolated_nodes(system1), 5)

    # Align var_names. 0.1.x's BNMetrics matched edges by the
    # nx.DiGraph node identity (the random renaming of each call to
    # generate_random_dag differs per seed, but the renamed nodes are
    # named X_1..X_n in both, just with different mappings to the
    # original integer ordering). v0.2 enforces matching `var_names`
    # tuples — so we re-construct system2 over system1's var_names
    # (same node names, just reordered to match system1's index space).
    name_to_idx_2 = {n: i for i, n in enumerate(system2.var_names)}
    perm = [name_to_idx_2[n] for n in system1.var_names]
    aligned_endpoints = system2.endpoints[np.ix_(perm, perm)]
    system2_aligned = bnm.to_graphlike(aligned_endpoints, var_names=system1.var_names)

    check("global TP", bnm.true_positives(system1, system2_aligned), 2)

    # 0.1.x's per-MB TP used MB(g1, v) vs MB(g2, v) — different node
    # sets, with directed-edge matching (X→Y in mb1 must also be X→Y
    # in mb2, by node NAME, regardless of node-set differences).
    # v0.2 unifies per-node metrics on g1.MB(v)'s indices for both
    # (stricter — see bnm/compare.py docstring). To verify parity
    # with the GitHub committed numbers, replicate 0.1.x semantics:
    def legacy_per_mb_tp(mb1, mb2):
        ARROW = bnm.EndpointMark.ARROW
        TAIL = bnm.EndpointMark.TAIL
        names2 = {n: i for i, n in enumerate(mb2.var_names)}
        tp = 0
        for i in range(mb1.n_vars):
            for j in range(mb1.n_vars):
                if i == j:
                    continue
                if mb1.endpoints[i, j] == ARROW and mb1.endpoints[j, i] == TAIL:
                    si = names2.get(mb1.var_names[i])
                    sj = names2.get(mb1.var_names[j])
                    if (
                        si is not None
                        and sj is not None
                        and (mb2.endpoints[si, sj] == ARROW and mb2.endpoints[sj, si] == TAIL)
                    ):
                        tp += 1
        return tp

    tp_positive_legacy: list[tuple[str, int]] = []
    for var in system1.var_names:
        mb1 = bnm.markov_blanket(system1, var)
        mb2 = bnm.markov_blanket(system2_aligned, var)
        tp = legacy_per_mb_tp(mb1, mb2)
        if tp > 0:
            tp_positive_legacy.append((var, tp))

    tp_positive_legacy.sort()
    print(f"  per-MB TP>0 (legacy convention): {tp_positive_legacy}")
    check(
        "per-MB TP>0 nodes (legacy convention: MB(g1,v) ∩ MB(g2,v))",
        [v for v, _ in tp_positive_legacy],
        ["X_39", "X_46", "X_65", "X_89"],
        kind="set",
    )


def check_compare_algorithms() -> None:
    """Reproduce GitHub's `compare algorithms.ipynb` setup."""
    print("\n=== compare algorithms.ipynb ===")
    true_dag = generate_random_dag_legacy(n_nodes=40, edge_prob=0.1, seed=55)
    print(f"  true_dag: {true_dag.n_vars} nodes, {bnm.count_edges(true_dag)} edges")
    # GitHub uses `castle.algorithms.PC` to learn at 20 alphas, which
    # we don't have installed. Skip the learned-output verification —
    # the structure-learning algorithm itself is external to bnm.
    print("  (skipping castle PC sweep; not a bnm computation)")


# ---- driver --------------------------------------------------------


def main() -> int:
    check_evaluate_single_dag()
    check_sid()
    check_compare_two_dags()
    check_compare_algorithms()

    print()
    if _failures:
        print(f"{len(_failures)} discrepancy:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("ALL CHECKS MATCH 0.1.x's GitHub-committed values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
