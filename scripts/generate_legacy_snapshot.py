"""Generate tests/fixtures_legacy.json from bnm 0.1.x.

This script is run ONCE on the pre-rewrite bnm 0.1.x source to freeze the
expected outputs of every metric that survives into v0.2. The v0.2 test
suite asserts equality against the produced JSON; any deliberate semantic
deviation requires a journal entry and a regeneration.

Usage:
    PYTHONHASHSEED=0 python scripts/generate_legacy_snapshot.py [--out PATH]

PYTHONHASHSEED=0 IS REQUIRED for reproducibility. bnm 0.1.x's SID
internally calls `set(G.nodes())` and walks `nx.connected_components`
output without sorting, so without a pinned hash seed the SID upper
bounds (and occasionally the lower bounds) drift across runs.
Regenerating without PYTHONHASHSEED=0 produces a non-comparable
snapshot — see docs/audit.md bug §6.

Output schema is documented in docs/audit.md (Slice 0 deliverable).

Notes on importing bnm 0.1.x:
    bnm/__init__.py at HEAD imports `from .viz import ...` which pulls
    plotly/graphviz/IPython at module load. None of those are needed by
    the metric functions we exercise. We install lightweight stubs in
    sys.modules BEFORE the bnm import so the script runs in any
    environment with networkx + numpy + pandas (which are already in the
    cbcd venv). The 0.1.x source itself was relocated to
    `scripts/legacy_0_1_x/` once v0.2 took over `bnm/`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import types
from datetime import date
from pathlib import Path

# --- enforce reproducibility ---------------------------------------------

if os.environ.get("PYTHONHASHSEED") != "0":
    # 0.1.x SID has hash-seed-dependent non-determinism (see audit §6).
    # Re-exec ourselves with PYTHONHASHSEED=0 if the user forgot.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

# --- stub viz/notebook deps before importing bnm --------------------------


class _Stub(types.ModuleType):
    def __getattr__(self, name: str) -> _Stub:
        attr = _Stub(f"{self.__name__}.{name}")
        setattr(self, name, attr)
        return attr

    def __call__(self, *_args: object, **_kwargs: object) -> _Stub:
        return _Stub(f"{self.__name__}()")


def _install_stubs() -> None:
    for mod in (
        "graphviz",
        "plotly",
        "plotly.graph_objects",
        "plotly.subplots",
        "IPython",
        "IPython.display",
    ):
        sys.modules.setdefault(mod, _Stub(mod))


_install_stubs()

BNM_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
# The legacy 0.1.x source was relocated to `scripts/legacy_0_1_x/` once
# v0.2 took over `bnm/`. We add `scripts/` to sys.path and import the
# legacy package under its new name to keep this generator runnable
# without git checkouts.
sys.path.insert(0, str(SCRIPTS_DIR))

import legacy_0_1_x as bnm  # noqa: E402
import networkx as nx  # noqa: E402
from legacy_0_1_x import metrics as m  # noqa: E402
from legacy_0_1_x.utils import (  # noqa: E402
    dag_to_cpdag,
    mark_and_collapse_bidirected_edges,
)

# --- fixture builders -----------------------------------------------------

GENERATOR_SEED = 42


def _digraph_from_edges(
    n: int,
    names: list[str],
    edges: list[tuple[int, int]],
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for src, dst in edges:
        g.add_edge(names[src], names[dst])
    return g


def canonical_fixtures() -> dict[str, nx.DiGraph]:
    """Hand-built canonical structures with stable node names.

    Names are chosen so list(g.nodes()) yields a deterministic order
    matching the documented v0.2 ordering rule.
    """
    out: dict[str, nx.DiGraph] = {}

    out["empty_3"] = _digraph_from_edges(3, ["A", "B", "C"], [])

    out["chain_3"] = _digraph_from_edges(3, ["A", "B", "C"], [(0, 1), (1, 2)])

    out["fork_3"] = _digraph_from_edges(3, ["A", "B", "C"], [(0, 1), (0, 2)])

    out["collider_3"] = _digraph_from_edges(3, ["A", "B", "C"], [(0, 2), (1, 2)])

    out["Y_4"] = _digraph_from_edges(4, ["A", "B", "C", "D"], [(0, 2), (1, 2), (2, 3)])

    out["M_4"] = _digraph_from_edges(4, ["A", "B", "C", "D"], [(0, 2), (1, 2), (1, 3)])

    out["diamond_4"] = _digraph_from_edges(
        4, ["A", "B", "C", "D"], [(0, 1), (0, 2), (1, 3), (2, 3)]
    )

    asia_names = [
        "asia",
        "tub",
        "smoke",
        "lung",
        "bronc",
        "either",
        "xray",
        "dysp",
    ]
    asia_edges = [
        (0, 1),  # asia -> tub
        (2, 3),  # smoke -> lung
        (2, 4),  # smoke -> bronc
        (1, 5),  # tub -> either
        (3, 5),  # lung -> either
        (5, 6),  # either -> xray
        (5, 7),  # either -> dysp
        (4, 7),  # bronc -> dysp
    ]
    out["asia_8"] = _digraph_from_edges(8, asia_names, asia_edges)

    return out


def random_dag(n: int, edge_prob: float, rng: random.Random, fid: str) -> nx.DiGraph:
    """Random DAG with deterministic-named nodes X_0..X_{n-1}.

    Node-name ordering is by index, so list(g.nodes()) is stable across
    Python dict-order changes.
    """
    names = [f"X_{i}" for i in range(n)]
    topo = list(range(n))
    rng.shuffle(topo)
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                g.add_edge(names[topo[i]], names[topo[j]])
    return g


def random_fixtures() -> dict[str, nx.DiGraph]:
    rng = random.Random(GENERATOR_SEED)
    out: dict[str, nx.DiGraph] = {}
    schedule = [
        (5, 0.30, 5),
        (10, 0.20, 5),
        (15, 0.15, 5),
        (20, 0.10, 5),
    ]
    for n, p, count in schedule:
        for k in range(count):
            fid = f"random_n{n}_p{int(p * 100):02d}_{k}"
            out[fid] = random_dag(n, p, rng, fid)
    return out


def perturb_dag(g: nx.DiGraph, rng: random.Random) -> nx.DiGraph:
    """Make a copy of g with a small edge perturbation: one of
    add a random non-edge, delete a random edge, reverse a random edge.
    """
    h = g.copy()
    nodes = list(h.nodes())
    edges = list(h.edges())
    actions = []
    if edges:
        actions.extend(["delete", "reverse"])
    if len(nodes) >= 2:
        actions.append("add")
    if not actions:
        return h
    action = rng.choice(actions)
    if action == "delete":
        u, v = rng.choice(edges)
        h.remove_edge(u, v)
    elif action == "reverse":
        u, v = rng.choice(edges)
        h.remove_edge(u, v)
        h.add_edge(v, u)
    elif action == "add":
        for _ in range(20):
            u, v = rng.sample(nodes, 2)
            if not h.has_edge(u, v) and not h.has_edge(v, u):
                h.add_edge(u, v)
                break
    return h


# --- snapshot serialization -----------------------------------------------


def _edges_with_type(g: nx.DiGraph) -> list[list[str]]:
    """Serialize g's edges as [src, dst, type] triples after the
    mark_and_collapse pass, so the snapshot reproduces the *post-mutation*
    graph that 0.1.x metrics actually consume.
    """
    out = []
    for u, v, data in g.edges(data=True):
        out.append([str(u), str(v), data.get("type", "directed")])
    return out


def _descriptive(g: nx.DiGraph) -> dict[str, float]:
    """All descriptive metrics that survive into v0.2."""
    return {
        "n_edges": m.count_edges(g),
        "n_nodes": m.count_nodes(g),
        "n_colliders": m.count_colliders(g),
        "n_root_nodes": m.count_root_nodes(g),
        "n_leaf_nodes": m.count_leaf_nodes(g),
        "n_isolated_nodes": m.count_isolated_nodes(g),
        "n_directed_arcs": m.count_directed_arcs(g),
        "n_undirected_arcs": m.count_undirected_arcs(g),
        "n_reversible_arcs": m.count_reversible_arcs(g),
    }


def _per_node_degree(g: nx.DiGraph) -> dict[str, dict[str, int]]:
    return {
        str(node): {
            "in": int(m.count_in_degree(g, node)),
            "out": int(m.count_out_degree(g, node)),
        }
        for node in g.nodes()
    }


def _comparative(g1: nx.DiGraph, g2: nx.DiGraph) -> dict[str, float]:
    return {
        "additions": m.count_additions(g1, g2),
        "deletions": m.count_deletions(g1, g2),
        "reversals": m.count_reversals(g1, g2),
        "shd": m.shd(g1, g2),
        "hd": m.hd(g1, g2),
        "tp": m.count_true_positives(g1, g2),
        "fp": m.count_false_positives(g1, g2),
        "fn": m.count_false_negatives(g1, g2),
        "precision": m.precision(g1, g2),
        "recall": m.recall(g1, g2),
        "f1_score": m.f1_score(g1, g2),
    }


def _g1_is_pure_dag(g: nx.DiGraph) -> bool:
    """SID requires g1 to have no undirected edges (post-mark/collapse)."""
    return all(data.get("type", "directed") == "directed" for _, _, data in g.edges(data=True))


def _maybe_sid(g1: nx.DiGraph, g2: nx.DiGraph) -> dict[str, object] | None:
    """Return SID dict, or None when SID is undefined (g1 not a DAG), or
    a stub `{"skipped": "<reason>"}` when 0.1.x's SID raises.

    0.1.x's SID has known crashes (e.g. empty `possible_pa_gp` when
    `gp_is_essential_graph` flips to False mid-iteration). We capture
    these as "skipped" so the snapshot doesn't lock v0.2 to legacy bugs;
    Slice 3 hand-computes the right answer for those cases.
    """
    if not _g1_is_pure_dag(g1):
        return None
    try:
        sid_dict = m.sid(g1, g2)
    except (ValueError, IndexError, KeyError) as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}"}
    return {
        "sid": float(sid_dict["sid"]),
        "sid_lower_bound": float(sid_dict["sid_lower_bound"]),
        "sid_upper_bound": float(sid_dict["sid_upper_bound"]),
    }


def build_snapshot() -> dict:
    canon = canonical_fixtures()
    rand = random_fixtures()
    fixtures = {**canon, **rand}

    # All metrics in 0.1.x assume mark_and_collapse has run. We snapshot
    # the *post-mutation* graphs so that v0.2's _to_endpoints adapter
    # has a deterministic input form and we can prove parity by
    # reconstructing the same graph from the JSON edge list.
    fixtures_marked = {fid: mark_and_collapse_bidirected_edges(g) for fid, g in fixtures.items()}

    fixtures_out: dict[str, dict] = {}
    for fid, g in fixtures_marked.items():
        kind = "canonical" if fid in canon else "random"
        fixtures_out[fid] = {
            "kind": kind,
            "n_nodes": g.number_of_nodes(),
            "node_names": list(map(str, g.nodes())),
            "edges": _edges_with_type(g),
            "descriptive": _descriptive(g),
            "per_node_degree": _per_node_degree(g),
        }

    pairs_out: list[dict] = []

    def _add_pair(pid: str, g1_id: str, g1: nx.DiGraph, g2_id: str, g2: nx.DiGraph) -> None:
        entry: dict = {
            "id": pid,
            "g1": g1_id,
            "g2": g2_id,
            "comparative": _comparative(g1, g2),
        }
        sid = _maybe_sid(g1, g2)
        if sid is not None:
            entry["sid"] = sid
        pairs_out.append(entry)

    # 1. self pairs (SHD=0 sanity)
    for fid, g in fixtures_marked.items():
        _add_pair(f"{fid}__vs__self", fid, g, fid, g)

    # 2. dag vs its CPDAG
    cpdag_cache: dict[str, nx.DiGraph] = {}
    for fid, g in fixtures_marked.items():
        cpdag_raw = dag_to_cpdag(g)
        cpdag = mark_and_collapse_bidirected_edges(cpdag_raw)
        cpdag_id = f"{fid}__cpdag"
        cpdag_cache[fid] = cpdag
        # Add the cpdag as its own fixture entry so v0.2 tests can rebuild it.
        fixtures_out[cpdag_id] = {
            "kind": "derived_cpdag",
            "of": fid,
            "n_nodes": cpdag.number_of_nodes(),
            "node_names": list(map(str, cpdag.nodes())),
            "edges": _edges_with_type(cpdag),
            "descriptive": _descriptive(cpdag),
            "per_node_degree": _per_node_degree(cpdag),
        }
        _add_pair(f"{fid}__vs__cpdag", fid, g, cpdag_id, cpdag)

    # 3. dag vs perturbation
    perturb_rng = random.Random(GENERATOR_SEED + 1)
    for fid, g in fixtures_marked.items():
        if g.number_of_edges() == 0:
            continue
        h_raw = perturb_dag(g, perturb_rng)
        h = mark_and_collapse_bidirected_edges(h_raw)
        h_id = f"{fid}__perturbed"
        fixtures_out[h_id] = {
            "kind": "derived_perturbation",
            "of": fid,
            "n_nodes": h.number_of_nodes(),
            "node_names": list(map(str, h.nodes())),
            "edges": _edges_with_type(h),
            "descriptive": _descriptive(h),
            "per_node_degree": _per_node_degree(h),
        }
        _add_pair(f"{fid}__vs__perturbed", fid, g, h_id, h)

    # 4. canonical cross-pairs (well-known SHDs)
    canonical_cross = [
        ("chain_3", "fork_3"),
        ("chain_3", "collider_3"),
        ("fork_3", "collider_3"),
        ("Y_4", "M_4"),
        ("diamond_4", "Y_4"),
        ("asia_8", "asia_8"),  # duplicate self for explicit cross-section listing
    ]
    for a, b in canonical_cross:
        if a == b:
            continue  # already in self pairs
        _add_pair(f"{a}__vs__{b}", a, fixtures_marked[a], b, fixtures_marked[b])

    git_hash = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BNM_ROOT),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )

    return {
        "meta": {
            "bnm_version": bnm.__version__ if hasattr(bnm, "__version__") else "0.1.0",
            "bnm_git_hash": git_hash,
            "generator_seed": GENERATOR_SEED,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
            "generated_at": date.today().isoformat(),
            "node_ordering_rule": (
                "list(g.nodes()) insertion order; canonical fixtures use "
                "explicit node_names; random fixtures use X_0..X_{n-1}"
            ),
            "notes": (
                "All graphs are post-mark_and_collapse_bidirected_edges. "
                "Bidirected fixtures intentionally excluded (0.1.x collapses "
                "them to undirected, which is a v0.2 semantic break)."
            ),
        },
        "fixtures": fixtures_out,
        "pairs": pairs_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=BNM_ROOT / "tests" / "fixtures_legacy.json",
    )
    args = parser.parse_args()

    snapshot = build_snapshot()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n")

    n_fixtures = len(snapshot["fixtures"])
    n_pairs = len(snapshot["pairs"])
    print(f"wrote {args.out} ({n_fixtures} fixtures, {n_pairs} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
