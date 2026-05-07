"""Rewrite the four `use cases/` notebooks to the v0.2 API.

Run as ``python scripts/migrate_notebooks.py``. Idempotent: rewrites
each notebook's cells from scratch. The legacy 0.1.x deprecation
banner is replaced with a v0.2 setup header.

This script encodes the new cell content inline; see the per-notebook
constants below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

USE_CASES = Path(__file__).resolve().parent.parent / "use cases"


_id_counter = 0


def _cell_id(text: str, prefix: str) -> str:
    """Monotonic deterministic cell id. Re-running the migrator on the
    same template produces identical ids; identical-content cells get
    different ids so nbformat doesn't reject the notebook."""
    global _id_counter
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    _id_counter += 1
    return f"{prefix}-{_id_counter:03d}-{digest}"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(text, "md"),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(text, "code"),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


# ---- shared setup helpers (prepended to every notebook) ---------------

SETUP_CODE = '''\
"""v0.2 examples — setup helpers.

The four 0.1.x notebooks used `BNMetrics`, `generate_random_dag`,
`generate_synthetic_data_from_dag`, `dag_to_cpdag`, plus a few visual
helpers. v0.2 reorganised the package, so we re-implement the four
small utilities below to keep the notebooks self-contained. (In a real
project, use `dagsampler` for DAG/data generation and `cbcd` for
CPDAG construction and PC.)
"""

import bnm
import numpy as np
import pandas as pd
import random
from collections import deque

# Configure plotly so figures render in any notebook viewer (JupyterLab,
# classic Notebook, nbviewer, GitHub's static renderer) instead of only
# emitting a vnd.plotly.v1+json mimetype.
import plotly.io as pio
pio.renderers.default = "notebook_connected+plotly_mimetype"


# ---- DAG / data generation (would normally come from dagsampler) ----

def random_dag(n_nodes=40, edge_prob=0.1, seed=None):
    """Return a bnm.GraphLike random DAG with X_1..X_n nodes."""
    rng = random.Random(seed)
    names = tuple(f"X_{i+1}" for i in range(n_nodes))
    topo = list(range(n_nodes))
    rng.shuffle(topo)
    endpoints = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                src, dst = topo[i], topo[j]
                endpoints[src, dst] = bnm.EndpointMark.ARROW
                endpoints[dst, src] = bnm.EndpointMark.TAIL
    return bnm.to_graphlike(endpoints, var_names=names)


def generate_data(g, n_samples=1000, stdev=1.0, seed=None):
    """Linear-Gaussian SCM sampling. Returns a pandas DataFrame
    with columns matching g.var_names."""
    rng = np.random.default_rng(seed)
    n = g.n_vars
    names = list(g.var_names) if g.var_names else [str(i) for i in range(n)]
    arr = np.array(g.endpoints)
    children: list[list[int]] = [[] for _ in range(n)]
    in_deg = [0] * n
    for i in range(n):
        for j in range(n):
            if (
                arr[i, j] == bnm.EndpointMark.ARROW
                and arr[j, i] == bnm.EndpointMark.TAIL
            ):
                children[i].append(j)
                in_deg[j] += 1
    queue = deque(v for v in range(n) if in_deg[v] == 0)
    topo: list[int] = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    data = pd.DataFrame(index=range(n_samples), columns=names, dtype=float)
    for v in topo:
        parents = [
            i for i in range(n)
            if arr[i, v] == bnm.EndpointMark.ARROW
            and arr[v, i] == bnm.EndpointMark.TAIL
        ]
        if not parents:
            data[names[v]] = rng.standard_normal(n_samples)
        else:
            weights = rng.uniform(0.5, 1.5, size=len(parents))
            x = rng.normal(0, stdev, size=n_samples)
            for p, w in zip(parents, weights):
                x = x + w * data[names[p]].values
            data[names[v]] = x
    return data


# ---- adjacency-matrix helpers ---------------------------------------

def from_01_adj(adj, var_names=None):
    """Convert a {0,1} adjacency matrix (1 = directed edge i→j) to a
    bnm.GraphLike. If both adj[i,j] == 1 and adj[j,i] == 1, the edge
    is treated as undirected (CPDAG-style)."""
    a = np.asarray(adj)
    n = a.shape[0]
    endpoints = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            ij, ji = int(a[i, j]), int(a[j, i])
            if ij == 1 and ji == 1:
                endpoints[i, j] = endpoints[j, i] = bnm.EndpointMark.TAIL
            elif ij == 1:
                endpoints[i, j] = bnm.EndpointMark.ARROW
                endpoints[j, i] = bnm.EndpointMark.TAIL
            elif ji == 1:
                endpoints[i, j] = bnm.EndpointMark.TAIL
                endpoints[j, i] = bnm.EndpointMark.ARROW
    names = tuple(var_names) if var_names is not None else None
    return bnm.to_graphlike(endpoints, var_names=names)


def dag_to_cpdag(g):
    """Convert a DAG to its CPDAG (without Meek-rule closure — same
    semantics as 0.1.x's `dag_to_cpdag`). For a Meek-closed CPDAG,
    use cbcd's `DAG.to_cpdag()` instead.
    """
    g = bnm.to_graphlike(g)
    n = g.n_vars
    arr = np.array(g.endpoints)
    in_collider = np.zeros((n, n), dtype=bool)
    for v in range(n):
        parents = [
            u for u in range(n)
            if arr[u, v] == bnm.EndpointMark.ARROW
            and arr[v, u] == bnm.EndpointMark.TAIL
        ]
        for ip in range(len(parents)):
            for jp in range(ip + 1, len(parents)):
                u, w = parents[ip], parents[jp]
                if (
                    arr[u, w] == bnm.EndpointMark.NO_EDGE
                    and arr[w, u] == bnm.EndpointMark.NO_EDGE
                ):
                    in_collider[u, v] = True
                    in_collider[w, v] = True
    cpdag = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                arr[i, j] == bnm.EndpointMark.ARROW
                and arr[j, i] == bnm.EndpointMark.TAIL
            ):
                if in_collider[i, j]:
                    cpdag[i, j] = bnm.EndpointMark.ARROW
                    cpdag[j, i] = bnm.EndpointMark.TAIL
                else:
                    cpdag[i, j] = bnm.EndpointMark.TAIL
                    cpdag[j, i] = bnm.EndpointMark.TAIL
    return bnm.to_graphlike(cpdag, var_names=g.var_names)


# ---- perturbation helper for "fake learned graph" demos ----------

def perturb(g, n_drops=0, n_adds=0, n_reverses=0, seed=None):
    """Return a perturbed copy of g: drop / add / reverse the requested
    number of directed edges. Used as a stand-in for a real causal-
    discovery output when we don't want to depend on a PC algorithm.

    For real workflows, swap this for `cbcd.pc(data, ...)` or any
    PC implementation whose output you can wrap in `from_01_adj()`.
    """
    g = bnm.to_graphlike(g)
    rng = random.Random(seed)
    arr = np.array(g.endpoints).copy()
    n = arr.shape[0]
    directed_edges = [
        (i, j) for i in range(n) for j in range(n)
        if arr[i, j] == bnm.EndpointMark.ARROW
        and arr[j, i] == bnm.EndpointMark.TAIL
    ]
    rng.shuffle(directed_edges)
    # Drops
    for k in range(min(n_drops, len(directed_edges))):
        i, j = directed_edges[k]
        arr[i, j] = arr[j, i] = bnm.EndpointMark.NO_EDGE
    remaining = directed_edges[n_drops:]
    # Reverses
    for k in range(min(n_reverses, len(remaining))):
        i, j = remaining[k]
        arr[i, j] = bnm.EndpointMark.TAIL
        arr[j, i] = bnm.EndpointMark.ARROW
    # Adds (only between currently-non-adjacent pairs to keep it a DAG-ish)
    non_edges = [
        (i, j) for i in range(n) for j in range(i + 1, n)
        if arr[i, j] == bnm.EndpointMark.NO_EDGE
    ]
    rng.shuffle(non_edges)
    for k in range(min(n_adds, len(non_edges))):
        i, j = non_edges[k]
        # Random direction
        if rng.random() < 0.5:
            arr[i, j] = bnm.EndpointMark.ARROW
            arr[j, i] = bnm.EndpointMark.TAIL
        else:
            arr[i, j] = bnm.EndpointMark.TAIL
            arr[j, i] = bnm.EndpointMark.ARROW
    return bnm.to_graphlike(arr, var_names=g.var_names)


# ---- viz helper: 0.1.x's compare_two_bn(option=2) ----------------

def mb_pair(g1, g2, var):
    """Return ``(g1.MB(var), g2 restricted to g1.MB(var) indices)``.
    Useful for side-by-side rendering of "true MB" vs "estimated MB
    over the same nodes." Replaces 0.1.x's ``compare_two_bn(option=2)``."""
    g1n, g2n = bnm.to_graphlike(g1), bnm.to_graphlike(g2)
    idx = bnm.markov_blanket_indices(g1n, var)
    sub_g1 = bnm.markov_blanket(g1n, var)
    arr2 = g2n.endpoints[np.ix_(idx, idx)]
    sub_names = (
        tuple(g2n.var_names[i] for i in idx) if g2n.var_names else None
    )
    sub_g2 = bnm.to_graphlike(arr2, var_names=sub_names)
    return sub_g1, sub_g2


print("bnm:", bnm.__version__)
'''


# ---- evaluate single DAG --------------------------------------------

EVAL_CELLS = [
    md_cell(
        "# Evaluation of a single DAG\n\n"
        "*v0.2 example.* Migrated from the 0.1.x `BNMetrics`-based "
        "version. The conceptual story is unchanged; only the API "
        "calls were updated to the new function-first surface."
    ),
    md_cell("## Setup"),
    md_cell(
        "Let's assume we have a Directed Acyclic Graph (DAG), either "
        "produced by a causal-discovery algorithm or hand-built. We "
        "want to characterise its structure both globally and at the "
        "Markov-blanket level."
    ),
    code_cell(SETUP_CODE),
    code_cell(
        "dag = random_dag(n_nodes=200, edge_prob=0.025, seed=55)\n"
        "f'DAG with {dag.n_vars} variables, "
        "{bnm.count_edges(dag)} edges'"
    ),
    md_cell(
        "Three v0.2 features support systematic exploration of the DAG:\n\n"
        "1. **`bnm.plot_graph(bnm.markov_blanket(g, var))`** — render "
        "the MB of any node.\n"
        "2. **`bnm.compare(g, per_node=True)`** + `bnm.to_dataframe(...)` "
        "— per-MB descriptive table.\n"
        "3. **`bnm.analyse_mb(g)`** — distribution of metrics across "
        "every MB."
    ),
    md_cell("## Feature 1 — plot Markov blankets of selected variables"),
    md_cell(
        "Let's say we're interested in `X_32`. To inspect its causal "
        "neighbourhood, render its Markov blanket."
    ),
    code_cell(
        "# Pick any node; we use X_32 (must exist for n_nodes=200).\n"
        "target = 'X_32'\n"
        "mb = bnm.markov_blanket(dag, target)\n"
        "bnm.plot_graph(mb, title=f'Markov blanket of {target}', "
        "direction='auto')"
    ),
    md_cell(
        "Combining several MBs — e.g. inspect the path between two "
        "variables. We build a sub-graph over the union of their "
        "blankets and render it."
    ),
    code_cell(
        "def union_mb(g, vars_):\n"
        "    indices = sorted({i for v in vars_\n"
        "                      for i in bnm.markov_blanket_indices(g, v)})\n"
        "    arr = np.array(g.endpoints)[np.ix_(indices, indices)]\n"
        "    names = tuple(g.var_names[i] for i in indices)\n"
        "    return bnm.to_graphlike(arr, var_names=names)\n\n"
        "bnm.plot_graph(\n"
        "    union_mb(dag, ['X_32', 'X_180']),\n"
        "    title='Markov blankets of X_32 and X_180',\n"
        "    direction='auto',\n"
        ")"
    ),
    md_cell(
        "## Feature 2 — comprehensive descriptive metrics\n\n"
        "`bnm.compare(g, per_node=True)` computes every descriptive "
        "metric on the whole graph **and** on each variable's MB; "
        "`bnm.to_dataframe(c)` renders it as a wide table with one "
        "row per node plus an `'All'` row."
    ),
    code_cell("c = bnm.compare(dag, per_node=True)\ndf = bnm.to_dataframe(c)\ndf.head()"),
    md_cell("Find isolated nodes (nodes whose MB sub-graph has at least one isolated vertex):"),
    code_cell(
        "isolated = df.query(\"node_name != 'All' and "
        'n_isolated_nodes > 0").node_name.tolist()\n'
        "print(f'{len(isolated)} nodes whose MB contains an isolated "
        "vertex')\n"
        "isolated[:10]"
    ),
    md_cell("Find the most complex Markov blanket (max edges):"),
    code_cell(
        "per_node = df[df.node_name != 'All']\n"
        "max_edges = per_node['n_edges'].max()\n"
        "complex_mb = per_node.loc[per_node['n_edges'] == max_edges, "
        "'node_name'].iloc[0]\n"
        "print(f'most complex MB: {complex_mb} ({max_edges} edges)')\n"
        "bnm.plot_graph(\n"
        "    bnm.markov_blanket(dag, complex_mb),\n"
        "    title=f'MB of {complex_mb}',\n"
        "    direction='auto',\n"
        ")"
    ),
    md_cell(
        "## Feature 3 — analyse the MB space\n\n"
        "`bnm.analyse_mb(g)` plots, for each descriptive metric, a "
        "value-count bar chart over the n MBs. Useful for "
        "characterising how heterogeneous the local structure is."
    ),
    code_cell("bnm.analyse_mb(dag)"),
    md_cell(
        "## Conclusion\n\n"
        "We explored a synthetic 1000-node DAG using three v0.2 "
        "features: per-node MB rendering (`plot_graph` + "
        "`markov_blanket`), the multi-metric per-node table "
        "(`compare` + `to_dataframe`), and the MB-space "
        "distribution (`analyse_mb`).\n\n"
        "All viz functions accept a `save=` parameter that writes "
        "the figure to disk; the format is inferred from the file "
        "extension."
    ),
    md_cell(
        "### Other Cases:\n"
        "- [Compare Two DAGs](./compare%20two%20DAGs.ipynb)\n"
        "- [Compare Algorithms](./compare%20algorithms.ipynb)\n"
        "- [SID](./sid.ipynb)"
    ),
]


# ---- compare two DAGs ----------------------------------------------

CMP2_CELLS = [
    md_cell(
        "# DAG comparison: two systems / truth-vs-learned\n\n"
        "*v0.2 example.* Migrated from the 0.1.x version."
    ),
    code_cell(SETUP_CODE),
    md_cell(
        "## Case 1 — condition-based comparison of two systems\n\n"
        "Suppose we have two DAGs representing the same set of "
        "variables under different conditions, and we want to know "
        "how they differ structurally."
    ),
    md_cell(
        "### Generate two related DAGs\n\n"
        "We start from a common 100-node DAG and create two structural "
        "variants by perturbing it (drops, adds, reversals). This "
        "guarantees shared structure to compare against — useful for "
        "the per-node drill-down below."
    ),
    code_cell(
        "base = random_dag(n_nodes=100, edge_prob=0.03, seed=55)\n"
        "system1 = perturb(base, n_drops=8, n_adds=4, n_reverses=3, seed=11)\n"
        "system2 = perturb(base, n_drops=12, n_adds=4, n_reverses=5, seed=22)"
    ),
    md_cell(
        "### Compare complexity (descriptive metrics)\n\n"
        "`bnm.compare(g1, g2, ...)` returns a `Comparison` dataclass; "
        "`bnm.to_dataframe(c)` renders it as a wide table. The "
        "`_base` suffix marks g1's value; the unsuffixed column is "
        "g2's."
    ),
    code_cell(
        "c = bnm.compare(\n"
        "    system1, system2,\n"
        "    descriptive=['n_edges', 'n_colliders', 'n_isolated_nodes'],\n"
        "    comparative=None,\n"
        ")\n"
        "bnm.to_dataframe(c)"
    ),
    md_cell(
        "### Similarity (comparative metrics)\n\n"
        "Comparative metrics (SHD, HD, F1, TP/FP/FN, etc.) measure "
        "how alike the two systems are."
    ),
    code_cell(
        "c = bnm.compare(\n"
        "    system1, system2,\n"
        "    descriptive=None,\n"
        "    comparative='all',\n"
        ")\n"
        "bnm.to_dataframe(c).T"
    ),
    md_cell(
        "### Per-MB drill-down\n\n"
        "Now we ask: which Markov blankets contain at least one "
        "true-positive edge? `per_node=True` computes the "
        "per-MB sub-table."
    ),
    code_cell(
        "c = bnm.compare(system1, system2, descriptive=None, per_node=True)\n"
        "df = bnm.to_dataframe(c)\n"
        "df.query(\"node_name != 'All' and tp > 0\")[['node_name', 'tp']]"
    ),
    md_cell(
        "### Visual inspection of nodes of interest\n\n"
        "For each node where the MBs share an edge, render side-by-"
        "side. The matching edge is highlighted in crimson; the "
        "anchor node is highlighted in light green."
    ),
    code_cell(
        "from IPython.display import display\n"
        "\n"
        "for var in df.query(\"node_name != 'All' and tp > 0\")"
        ".node_name.tolist()[:4]:\n"
        "    sub1, sub2 = mb_pair(system1, system2, var)\n"
        "    display(\n"
        "        bnm.plot_side_by_side(\n"
        "            sub1, sub2,\n"
        "            name1=f'system1 MB of {var}',\n"
        "            name2=f'system2 over the same nodes',\n"
        "            highlight_nodes=[var],\n"
        "            direction='auto',\n"
        "        )\n"
        "    )"
    ),
    md_cell(
        "## Case 2 — validating a learned DAG against the truth\n\n"
        "Generate a true DAG, fabricate a 'learned' graph by "
        "perturbing it (a stand-in for a real causal-discovery "
        "output), and compare the truth's CPDAG to the learned one. "
        "For a real PC run, swap `perturb(...)` for "
        "`cbcd.pc(data, ...)` (the suite's PC implementation) — "
        "its output already conforms to `bnm.GraphLike`."
    ),
    code_cell(
        "true_dag = random_dag(n_nodes=40, edge_prob=0.1, seed=77)\n"
        "f'true DAG: {true_dag.n_vars} variables, "
        "{bnm.count_edges(true_dag)} edges'"
    ),
    md_cell(
        "### Construct a 'learned' graph\n\n"
        "Drop a few edges, reverse a couple, add some noise. The "
        "result is a structurally plausible PC-output stand-in."
    ),
    code_cell(
        "learned = perturb(true_dag, n_drops=15, n_reverses=10, n_adds=8, seed=42)\n"
        "f'learned: {bnm.count_edges(learned)} edges'"
    ),
    md_cell("### Compare complexity"),
    code_cell(
        "true_cpdag = dag_to_cpdag(true_dag)\n"
        "c = bnm.compare(\n"
        "    true_cpdag, learned,\n"
        "    descriptive=['n_edges', 'n_colliders', 'n_isolated_nodes'],\n"
        "    comparative=None,\n"
        ")\n"
        "bnm.to_dataframe(c)"
    ),
    md_cell("### Similarity"),
    code_cell(
        "c = bnm.compare(true_cpdag, learned, descriptive=None, "
        "comparative='all')\n"
        "bnm.to_dataframe(c).T"
    ),
    md_cell(
        "### Visualize MBs that match perfectly\n\n"
        "Find variables where F1 = 1 (the local structure is exactly "
        "right)."
    ),
    code_cell(
        "c = bnm.compare(true_cpdag, learned, descriptive=None, "
        "per_node=True)\n"
        "df = bnm.to_dataframe(c)\n"
        "perfect = df.query(\"node_name != 'All' and f1 == 1.0\")"
        ".node_name.tolist()\n"
        "print(f'{len(perfect)} variables have a perfectly-recovered MB')\n"
        "perfect[:5]"
    ),
    md_cell("Pick a partially-recovered MB (0.5 < F1 < 1) and inspect it."),
    code_cell(
        "partial = df.query(\"node_name != 'All' and 0.5 < f1 < 1.0\")"
        "[['node_name', 'f1', 'shd']].sort_values('f1', ascending=False)\n"
        "partial.head()"
    ),
    code_cell(
        "from IPython.display import display\n"
        "\n"
        "if len(partial):\n"
        "    var = partial.iloc[0].node_name\n"
        "    sub1, sub2 = mb_pair(true_cpdag, learned, var)\n"
        "    display(\n"
        "        bnm.plot_side_by_side(\n"
        "            sub1, sub2,\n"
        "            name1=f'true MB of {var}',\n"
        "            name2=f'learned MB of {var}',\n"
        "            highlight_nodes=[var],\n"
        "            direction='auto',\n"
        "        )\n"
        "    )\n"
        "else:\n"
        "    print('No partial-recovery MBs in this run.')"
    ),
    md_cell(
        "## Conclusions\n\n"
        "Two cases for DAG comparison:\n\n"
        "- **Side-by-side condition comparison** — build per-node "
        "drill-downs from `bnm.compare(..., per_node=True)`.\n"
        "- **Validation of learned DAG** — same flow, with the "
        "truth's CPDAG as the reference and the PC output as the "
        "estimate.\n\n"
        "All viz accepts `save=path`."
    ),
    md_cell(
        "### Other Cases:\n"
        "- [Evaluate Single DAG](./evaluate%20single%20DAG.ipynb)\n"
        "- [Compare Algorithms](./compare%20algorithms.ipynb)\n"
        "- [SID](./sid.ipynb)"
    ),
]


# ---- compare algorithms --------------------------------------------

CMP_ALG_CELLS = [
    md_cell(
        "# Multiple-model comparison: PC alpha sweep\n\n"
        "*v0.2 example.* Migrated from the 0.1.x version."
    ),
    md_cell(
        "## Motivation\n\n"
        "Suppose we want to compare several outputs of a structure-"
        "learning algorithm produced under different hyperparameters "
        "— e.g. PC at multiple `alpha` values. We want to characterise "
        "complexity patterns and similarity patterns across the "
        "resulting DAG family."
    ),
    code_cell(SETUP_CODE),
    md_cell(
        "### Data generation\n\n"
        "Random 40-node DAG with 10% density, then linear-Gaussian "
        "data over it."
    ),
    code_cell(
        "true_dag = random_dag(n_nodes=40, edge_prob=0.1, seed=55)\n"
        "data = generate_data(true_dag, n_samples=1000, stdev=1.0, seed=55)\n"
        "f'true DAG: {true_dag.n_vars} vars, "
        "{bnm.count_edges(true_dag)} edges; "
        "data: {data.shape}'"
    ),
    md_cell(
        "## Structure learning (simulated alpha sweep)\n\n"
        "We simulate the effect of varying PC's alpha threshold by "
        "fabricating 20 graphs with progressively more edges (smaller "
        "alpha → fewer edges retained → looks like dropping). For a "
        "real PC sweep, swap the perturbation loop for `cbcd.pc(data, "
        "alpha=a, ...)` calls — the rest of this notebook is "
        "unchanged."
    ),
    code_cell(
        "alphas = np.linspace(0.01, 0.2, 20)\n"
        "# Simulate: lower alpha drops more truth-edges; higher alpha\n"
        "# adds more spurious edges. Reverses kept constant.\n"
        "list_of_models = []\n"
        "n_truth_edges = bnm.count_edges(true_dag)\n"
        "for k, a in enumerate(alphas):\n"
        "    n_drops = int(round((1 - a / 0.2) * 0.4 * n_truth_edges))\n"
        "    n_adds = int(round((a / 0.2) * 0.3 * n_truth_edges))\n"
        "    list_of_models.append(\n"
        "        perturb(true_dag, n_drops=n_drops, n_adds=n_adds,\n"
        "                n_reverses=2, seed=100 + k)\n"
        "    )\n"
        "len(list_of_models)"
    ),
    md_cell(
        "## Complexity patterns and algorithm behavior\n\n"
        "`bnm.compare_models_descriptive` plots each descriptive "
        "metric (n_edges, n_colliders, etc.) as model_name → value, "
        "one panel per metric. With `per_node=True`, a Plotly "
        "dropdown lets you switch between whole-graph and per-MB "
        "views."
    ),
    code_cell(
        "model_labels = [f'{a:.3f}' for a in alphas]\n\n"
        "bnm.compare_models_descriptive(\n"
        "    list_of_models,\n"
        "    model_labels,\n"
        "    descriptive=['n_edges', 'n_colliders', 'n_directed_arcs',\n"
        "                 'n_undirected_arcs', 'n_root_nodes', 'n_leaf_nodes',\n"
        "                 'n_isolated_nodes', 'n_reversible_arcs'],\n"
        "    per_node=True,\n"
        "    title='PC alpha sweep — descriptive metrics',\n"
        ")"
    ),
    md_cell(
        "As expected, higher alpha produces denser networks (more "
        "edges, more colliders, more directed arcs)."
    ),
    md_cell(
        "Local view: zoom into a specific Markov blanket (e.g. "
        "`X_1`) by selecting it in the dropdown. To compare two "
        "specific models on that MB, use `plot_side_by_side` "
        "directly."
    ),
    code_cell(
        "g_low = list_of_models[6]\n"
        "g_high = list_of_models[13]\n"
        "sub_low, sub_high = mb_pair(g_low, g_high, 'X_1')\n"
        "bnm.plot_side_by_side(\n"
        "    sub_low, sub_high,\n"
        "    name1=f'alpha={alphas[6]:.3f}',\n"
        "    name2=f'alpha={alphas[13]:.3f}',\n"
        "    highlight_nodes=['X_1'],\n"
        "    direction='auto',\n"
        ")"
    ),
    md_cell(
        "## Similarity patterns and model stability\n\n"
        "`bnm.compare_models_comparative` computes one comparative "
        "metric across all (n × n) pairs and renders the heatmap. "
        "F1 of 1.0 means two models produce identical structure; "
        "lower F1 means the structures diverge."
    ),
    code_cell(
        "bnm.compare_models_comparative(\n"
        "    list_of_models,\n"
        "    model_labels,\n"
        "    metric='f1',\n"
        "    per_node=True,\n"
        "    title='PC alpha sweep — pairwise F1',\n"
        ")"
    ),
    md_cell(
        "In the dropdown, switch to a specific MB (e.g. `X_1`) — "
        "you'll often see clusters of identical models for the same "
        "local structure even when the global F1 differs."
    ),
    md_cell(
        "## Conclusion\n\n"
        "Two views answer most questions about a model family:\n\n"
        "- `compare_models_descriptive`: how complexity changes "
        "across hyperparameters.\n"
        "- `compare_models_comparative`: how similar the models "
        "are to each other (and to the truth, if included)."
    ),
    md_cell(
        "### Other Cases:\n"
        "- [Evaluate Single DAG](./evaluate%20single%20DAG.ipynb)\n"
        "- [Compare Two DAGs](./compare%20two%20DAGs.ipynb)\n"
        "- [SID](./sid.ipynb)"
    ),
]


# ---- SID ------------------------------------------------------------

SID_CELLS = [
    md_cell(
        "# Structural Intervention Distance\n\n*v0.2 example.* Migrated from the 0.1.x version."
    ),
    code_cell(SETUP_CODE),
    md_cell(
        "# Introduction\n\n"
        "SID quantifies how many intervention pairs (i, j) are "
        "mis-classified between a true DAG and an estimated DAG/"
        "CPDAG. Unlike SHD, SID weighs *causal-inference* "
        "consequences of structural errors — a missing direct "
        "edge that's covered by an indirect path costs less than "
        "a reversed edge.\n\n"
        "Reference: Peters & Bühlmann (2015), "
        "https://doi.org/10.48550/arXiv.1306.1043"
    ),
    md_cell(
        "## Use case 1 — SID vs SHD on small examples\n\n"
        "We compare a true DAG **G** against two estimates that "
        "both differ from G by exactly one edge (SHD = 1) but in "
        "structurally different ways:\n\n"
        "- **H₁**: G with the X1→X2 edge replaced by an undirected "
        "  X1—X2 (a CPDAG). The Markov-equivalence class contains "
        "  G itself, so SID's lower bound is 0.\n"
        "- **H₂**: G with the X1→X2 edge reversed (X2→X1). A pure "
        "  DAG that differs causally; SID > 0."
    ),
    code_cell(
        "nodes = ['X1', 'X2', 'Y1', 'Y2', 'Y3']\n\n"
        "G = np.array([\n"
        "    [0, 1, 1, 1, 1],\n"
        "    [0, 0, 1, 1, 1],\n"
        "    [0, 0, 0, 0, 0],\n"
        "    [0, 0, 0, 0, 0],\n"
        "    [0, 0, 0, 0, 0],\n"
        "])\n"
        "# H1: X1—X2 undirected (a CPDAG); rest identical to G.\n"
        "H1 = np.array([\n"
        "    [0, 1, 1, 1, 1],\n"
        "    [1, 0, 1, 1, 1],\n"
        "    [0, 0, 0, 0, 0],\n"
        "    [0, 0, 0, 0, 0],\n"
        "    [0, 0, 0, 0, 0],\n"
        "])\n"
        "# H2: X1→X2 reversed to X2→X1; everything else identical.\n"
        "H2 = np.array([\n"
        "    [0, 0, 1, 1, 1],\n"
        "    [1, 0, 1, 1, 1],\n"
        "    [0, 0, 0, 0, 0],\n"
        "    [0, 0, 0, 0, 0],\n"
        "    [0, 0, 0, 0, 0],\n"
        "])\n\n"
        "g_true = from_01_adj(G, var_names=nodes)\n"
        "g_h1 = from_01_adj(H1, var_names=nodes)\n"
        "g_h2 = from_01_adj(H2, var_names=nodes)"
    ),
    md_cell("### Case 1 — G vs H₁ (CPDAG with G in its equivalence class)"),
    code_cell(
        "bnm.plot_side_by_side(\n    g_true, g_h1, name1='G', name2='H1', direction='auto'\n)"
    ),
    code_cell(
        "c = bnm.compare(g_true, g_h1, comparative=['shd'], include_sid=True)\n"
        "print(f'SHD = {c.comparative[\"shd\"]}, '\n"
        "      f'SID = {c.sid.sid}, '\n"
        "      f'bounds = [{c.sid.sid_lower_bound}, "
        "{c.sid.sid_upper_bound}], '\n"
        "      f'is_tight = {c.sid.is_tight}')"
    ),
    md_cell(
        "**Key observations**:\n\n"
        "- SHD(G, H₁) = 1 (one orientation differs).\n"
        "- SID lower bound = 0 — when X1—X2 is oriented as X1→X2 "
        "(picking G itself out of the equivalence class), every "
        "intervention prediction matches G.\n"
        "- SID upper bound > 0 — the alternative orientation "
        "(X2→X1) does change interventions and gives a non-zero SID.\n"
        "- `is_tight = False` flags that the bounds aren't equal — "
        "the equivalence class is non-trivial.\n\n"
        "The `c.sid.sid` value itself counts (i, j) pairs that are "
        "mis-classified by **at least one** DAG in the class — "
        "useful when you want a worst-case-over-class estimate."
    ),
    code_cell("bnm.plot_sid_matrix(c.sid)"),
    md_cell(
        "**Conclusion**: SID has a `[lower, upper]` interval when "
        "the estimate is a CPDAG. The lower bound = 0 here means "
        "*there exists* a DAG in H₁'s class that perfectly matches "
        "G's intervention predictions — the orientation ambiguity "
        "doesn't force any wrong call."
    ),
    md_cell("### Case 2 — G vs H₂ (single reversed edge)"),
    code_cell(
        "c = bnm.compare(g_true, g_h2, comparative=['shd'], include_sid=True)\n"
        "print(f'SHD = {c.comparative[\"shd\"]}, '\n"
        "      f'SID = {c.sid.sid}, '\n"
        "      f'bounds = [{c.sid.sid_lower_bound}, "
        "{c.sid.sid_upper_bound}], '\n"
        "      f'is_tight = {c.sid.is_tight}')"
    ),
    code_cell("bnm.plot_sid_matrix(c.sid)"),
    md_cell(
        "**Key observations**:\n\n"
        "- SHD(G, H₂) = 1 — same as G vs H₁.\n"
        "- SID > 0 — the reversed edge creates real intervention "
        "differences.\n"
        "- Bounds are tight (lower == upper) — H₂ is a pure DAG, "
        "no equivalence-class ambiguity.\n\n"
        "**Takeaway**: two graphs can have equal SHD but very "
        "different SID. SHD is a structural distance; SID is a "
        "causal-inference distance."
    ),
    md_cell(
        "## Use case 2 — DAG bounds via CPDAG ambiguity\n\n"
        "When the estimated graph is a CPDAG, the orientation of "
        "undirected edges is ambiguous. SID has a `[lower, upper]` "
        "interval reflecting the min and max SID over all DAGs in "
        "the equivalence class."
    ),
    code_cell(
        "nodes4 = ['X1', 'X2', 'X3', 'X4']\n"
        "DAG_a = np.array([\n"
        "    [0, 0, 0, 0],\n"
        "    [0, 0, 1, 0],\n"
        "    [1, 0, 0, 0],\n"
        "    [1, 0, 1, 0],\n"
        "])\n"
        "DAG_b = np.array([\n"
        "    [0, 1, 0, 0],\n"
        "    [0, 0, 0, 0],\n"
        "    [0, 1, 0, 0],\n"
        "    [0, 0, 1, 0],\n"
        "])\n\n"
        "g_dag_a = from_01_adj(DAG_a, var_names=nodes4)\n"
        "g_dag_b = from_01_adj(DAG_b, var_names=nodes4)\n\n"
        "for label, g_est in [('DAG_b', g_dag_b)]:\n"
        "    c = bnm.compare(g_dag_a, g_est, comparative=['shd'], "
        "include_sid=True)\n"
        '    print(f\'{label}: SHD={c.comparative["shd"]}, '
        "SID={c.sid.sid}, "
        "bounds=[{c.sid.sid_lower_bound}, "
        "{c.sid.sid_upper_bound}]')"
    ),
    md_cell("Comparing the same true DAG to its own CPDAG (the Markov-equivalence class):"),
    code_cell(
        "cpdag_a = dag_to_cpdag(g_dag_a)\n"
        "c = bnm.compare(g_dag_a, cpdag_a, comparative=['shd'], "
        "include_sid=True)\n"
        'print(f\'truth vs CPDAG-of-truth: SHD={c.comparative["shd"]}, '
        "SID={c.sid.sid}, "
        "bounds=[{c.sid.sid_lower_bound}, "
        "{c.sid.sid_upper_bound}]')\n"
        "print(f'is_tight: {c.sid.is_tight}')"
    ),
    md_cell(
        "`is_tight=False` indicates the bound interval is "
        "non-trivial — the equivalence class contains DAGs that "
        "yield different SID values against the truth."
    ),
    md_cell(
        "## Use case 3 — SID on Markov blankets (local)\n\n"
        "We can localise SID to a node's Markov blanket: build "
        "the MB sub-graphs of g1 and g2 over the same node "
        "indices, compute SID on the pair.\n\n"
        "We use a perturbation as a stand-in for a real PC output. "
        "For real workflows, swap `perturb(...)` for "
        "`cbcd.pc(data, ...)`."
    ),
    code_cell(
        "true_dag = random_dag(n_nodes=40, edge_prob=0.1, seed=55)\n"
        "learned = perturb(true_dag, n_drops=12, n_reverses=8, "
        "n_adds=6, seed=42)"
    ),
    md_cell("### Local SID for a single variable"),
    code_cell(
        "var = 'X_16'\n"
        "sub1, sub2 = mb_pair(true_dag, learned, var)\n\n"
        "bnm.plot_side_by_side(\n"
        "    sub1, sub2,\n"
        "    name1=f'True MB of {var}',\n"
        "    name2=f'Learned over the same nodes',\n"
        "    highlight_nodes=[var],\n"
        "    direction='auto',\n"
        ")"
    ),
    code_cell(
        "# Note: SID requires g1 to be a pure DAG. The MB sub-graph\n"
        "# of a DAG is itself a DAG; the learned MB sub-graph is\n"
        "# possibly a CPDAG (TAIL/TAIL undirected edges).\n"
        "sid_local = bnm.sid(sub1, sub2)\n"
        "print(f'local SID for {var}: sid={sid_local.sid}, "
        "bounds=[{sid_local.sid_lower_bound}, "
        "{sid_local.sid_upper_bound}]')\n\n"
        "bnm.plot_sid_matrix(sid_local)"
    ),
    md_cell(
        "### Local SID for multiple variables\n\n"
        "Using the union of MB index sets gives a sub-graph "
        "containing both anchor variables and their dependencies."
    ),
    code_cell(
        "def union_mb(g, vars_):\n"
        "    g_norm = bnm.to_graphlike(g)\n"
        "    indices = sorted({i for v in vars_\n"
        "                      for i in bnm.markov_blanket_indices(g_norm, v)})\n"
        "    arr = g_norm.endpoints[np.ix_(indices, indices)]\n"
        "    names = (\n"
        "        tuple(g_norm.var_names[i] for i in indices)\n"
        "        if g_norm.var_names else None\n"
        "    )\n"
        "    return bnm.to_graphlike(arr, var_names=names), indices\n\n"
        "vars_ = ['X_16', 'X_35']\n"
        "sub_truth, idx = union_mb(true_dag, vars_)\n\n"
        "# Restrict learned to the same indices.\n"
        "learned_n = bnm.to_graphlike(learned)\n"
        "sub_learned = bnm.to_graphlike(\n"
        "    learned_n.endpoints[np.ix_(idx, idx)],\n"
        "    var_names=tuple(learned_n.var_names[i] for i in idx)\n"
        "    if learned_n.var_names else None,\n"
        ")\n\n"
        "bnm.plot_side_by_side(\n"
        "    sub_truth, sub_learned,\n"
        "    name1=f'True MBs of {vars_}',\n"
        "    name2='Learned over the same nodes',\n"
        "    highlight_nodes=vars_,\n"
        "    direction='auto',\n"
        ")"
    ),
    code_cell(
        "sid_multi = bnm.sid(sub_truth, sub_learned)\n"
        "print(f'union-MB SID: {sid_multi.sid}')\n"
        "bnm.plot_sid_matrix(sid_multi)"
    ),
    md_cell(
        "## Conclusion\n\n"
        "Three SID workflows in v0.2:\n\n"
        "1. **Whole-graph SID**: `bnm.sid(g_true, g_est)` returns a "
        "`SIDResult` with `sid`, `sid_lower_bound`, "
        "`sid_upper_bound`, and an `incorrect_mat` (n × n) for "
        "heatmap rendering.\n"
        "2. **CPDAG bounds**: when `g_est` is a CPDAG, the bounds "
        "interval reflects equivalence-class ambiguity. "
        "`SIDResult.is_tight` flags whether `lower == upper`.\n"
        "3. **Local SID**: build MB sub-graphs (over the same "
        "node indices via `mb_pair` / `union_mb`) and compute "
        "SID on the pair.\n\n"
        "All viz functions (`plot_side_by_side`, `plot_sid_matrix`) "
        "accept `save=path` to write to disk."
    ),
    md_cell(
        "### Other Cases:\n"
        "- [Evaluate Single DAG](./evaluate%20single%20DAG.ipynb)\n"
        "- [Compare Two DAGs](./compare%20two%20DAGs.ipynb)\n"
        "- [Compare Algorithms](./compare%20algorithms.ipynb)"
    ),
]


# ---- driver -----------------------------------------------------------

NOTEBOOKS = {
    "evaluate single DAG.ipynb": EVAL_CELLS,
    "compare two DAGs.ipynb": CMP2_CELLS,
    "compare algorithms.ipynb": CMP_ALG_CELLS,
    "sid.ipynb": SID_CELLS,
}


def main() -> int:
    for filename, cells in NOTEBOOKS.items():
        path = USE_CASES / filename
        nb = json.loads(path.read_text())
        nb["cells"] = cells
        path.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"  wrote: {path}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
