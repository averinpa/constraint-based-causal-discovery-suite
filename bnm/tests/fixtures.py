"""Test fixtures for bnm v0.2.

Plain functions (not pytest fixtures), imported by test modules. Style
mirrors cbcd's `tests/fixtures.py`.

Two layers:
  - ``make_*`` builders for hand-built canonical graphs.
  - ``load_legacy_snapshot`` + ``from_legacy_fixture`` for the frozen
    bnm 0.1.x snapshot at ``tests/fixtures_legacy.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bnm._graph import _Graph
from bnm.marks import EndpointMark

ARROW = int(EndpointMark.ARROW)
TAIL = int(EndpointMark.TAIL)


# ---- builders -----------------------------------------------------------


def make_dag(
    n_vars: int,
    directed_edges: list[tuple[int, int]],
    var_names: tuple[str, ...] | None = None,
) -> _Graph:
    """Construct a `_Graph` containing only directed edges.

    `directed_edges` is a list of ``(src, dst)`` index pairs encoding
    ``src → dst``.
    """
    endpoints = np.zeros((n_vars, n_vars), dtype=np.int8)
    for src, dst in directed_edges:
        endpoints[src, dst] = ARROW
        endpoints[dst, src] = TAIL
    return _Graph(n_vars=n_vars, endpoints=endpoints, var_names=var_names)


def make_cpdag(
    n_vars: int,
    directed_edges: list[tuple[int, int]],
    undirected_edges: list[tuple[int, int]],
    var_names: tuple[str, ...] | None = None,
) -> _Graph:
    """Construct a `_Graph` with directed and undirected edges.

    Directed: ``src → dst``. Undirected: ``i — j`` (TAIL on both ends).
    """
    endpoints = np.zeros((n_vars, n_vars), dtype=np.int8)
    for src, dst in directed_edges:
        endpoints[src, dst] = ARROW
        endpoints[dst, src] = TAIL
    for i, j in undirected_edges:
        endpoints[i, j] = TAIL
        endpoints[j, i] = TAIL
    return _Graph(n_vars=n_vars, endpoints=endpoints, var_names=var_names)


# ---- canonical graphs (subset of the legacy snapshot) -------------------


def empty_3() -> _Graph:
    return make_dag(3, [], var_names=("A", "B", "C"))


def chain_3() -> _Graph:
    """A → B → C."""
    return make_dag(3, [(0, 1), (1, 2)], var_names=("A", "B", "C"))


def fork_3() -> _Graph:
    """A → B, A → C."""
    return make_dag(3, [(0, 1), (0, 2)], var_names=("A", "B", "C"))


def collider_3() -> _Graph:
    """A → C, B → C."""
    return make_dag(3, [(0, 2), (1, 2)], var_names=("A", "B", "C"))


def y_4() -> _Graph:
    """A → C, B → C, C → D."""
    return make_dag(4, [(0, 2), (1, 2), (2, 3)], var_names=("A", "B", "C", "D"))


def m_4() -> _Graph:
    """A → C, B → C, B → D."""
    return make_dag(4, [(0, 2), (1, 2), (1, 3)], var_names=("A", "B", "C", "D"))


def diamond_4() -> _Graph:
    """A → B, A → C, B → D, C → D."""
    return make_dag(4, [(0, 1), (0, 2), (1, 3), (2, 3)], var_names=("A", "B", "C", "D"))


def asia_8() -> _Graph:
    """The bnlearn ASIA network."""
    names = ("asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp")
    edges = [
        (0, 1),  # asia → tub
        (2, 3),  # smoke → lung
        (2, 4),  # smoke → bronc
        (1, 5),  # tub → either
        (3, 5),  # lung → either
        (5, 6),  # either → xray
        (5, 7),  # either → dysp
        (4, 7),  # bronc → dysp
    ]
    return make_dag(8, edges, var_names=names)


# ---- legacy snapshot loader --------------------------------------------


_LEGACY_SNAPSHOT_PATH = Path(__file__).parent / "fixtures_legacy.json"


def load_legacy_snapshot() -> dict[str, Any]:
    """Load the frozen bnm 0.1.x snapshot."""
    return json.loads(_LEGACY_SNAPSHOT_PATH.read_text())


def from_legacy_fixture(entry: dict[str, Any]) -> _Graph:
    """Reconstruct a `_Graph` from a legacy-snapshot fixture entry.

    The entry has ``node_names: list[str]`` and
    ``edges: list[[src, dst, type]]`` where type is 'directed' or
    'undirected'.
    """
    names = tuple(entry["node_names"])
    n = entry["n_nodes"]
    name_to_idx = {n: i for i, n in enumerate(names)}
    endpoints = np.zeros((n, n), dtype=np.int8)
    for src, dst, edge_type in entry["edges"]:
        i = name_to_idx[src]
        j = name_to_idx[dst]
        if edge_type == "directed":
            endpoints[i, j] = ARROW
            endpoints[j, i] = TAIL
        elif edge_type == "undirected":
            endpoints[i, j] = TAIL
            endpoints[j, i] = TAIL
        else:
            raise ValueError(
                f"unsupported edge type {edge_type!r} in legacy fixture {entry.get('kind')}"
            )
    return _Graph(n_vars=n, endpoints=endpoints, var_names=names)
