"""MMB-by-MMB (Xie et al., 2024) — the paper's guarantees, checked at a d-separation oracle.

Theorem 1 says an edge at the target is decidable inside MMB+(target). Theorem 2 says the
V-structures involving the target found there are true. Theorem 3 says that when a stopping rule
fires, the structure for the target equals what global learning gives. Those are asserted here
against ``fci`` on the same oracle.

One claim is NOT asserted, because it is false as stated. Section 3.1's MMB definition omits the
vertices joined to the target by a bidirected edge, and the test below measures that rather than
assuming it.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from cbcd import InMemoryRecorder, fci, mmb_by_mmb, total_conditioning
from cbcd.exceptions import CBCDInputError
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark as M
from tests.oracle_pag import DSeparationOracleProjected
from tests.test_locals import _marks_at, _rand_latent_case

ARROW, TAIL, CIRCLE, NO_EDGE = int(M.ARROW), int(M.TAIL), int(M.CIRCLE), int(M.NO_EDGE)


def _true_mag(dag: DAG, n_obs: int) -> dict[tuple[int, int], tuple[str, str]]:
    """The MAG over the observed variables, by the standard construction."""
    g = nx.DiGraph()
    g.add_nodes_from(range(dag.n_vars))
    for i in range(dag.n_vars):
        for j in range(dag.n_vars):
            if dag.endpoints[i, j] == M.ARROW and dag.endpoints[j, i] == M.TAIL:
                g.add_edge(i, j)
    obs = list(range(n_obs))
    edges: dict[tuple[int, int], tuple[str, str]] = {}
    for x, y in itertools.combinations(obs, 2):
        rest = [v for v in obs if v not in (x, y)]
        if any(
            nx.is_d_separator(g, {x}, {y}, set(S))
            for k in range(len(rest) + 1)
            for S in itertools.combinations(rest, k)
        ):
            continue
        xa, ya = x in nx.ancestors(g, y), y in nx.ancestors(g, x)
        edges[(x, y)] = ("t", "a") if xa else ("a", "t") if ya else ("a", "a")
    return edges


def _mmb_as_written(edges, t: int, n: int, *, with_spouses: bool) -> set[int]:
    """Section 3.1 — parents, children, and non-adjacent vertices with a collider path to t."""

    def mark(u, v):
        if (u, v) in edges:
            return edges[(u, v)][1]
        return edges[(v, u)][0] if (v, u) in edges else None

    def adj(u):
        return [w for w in range(n) if w != u and mark(u, w) is not None]

    out = {w for w in adj(t) if mark(t, w) == "a" and mark(w, t) == "t"}  # children
    out |= {w for w in adj(t) if mark(w, t) == "a" and mark(t, w) == "t"}  # parents
    if with_spouses:
        out |= {w for w in adj(t) if mark(t, w) == "a" and mark(w, t) == "a"}
    for v in range(n):
        if v == t or mark(t, v) is not None:
            continue
        stack = [[v, w] for w in adj(v)]
        while stack:
            path = stack.pop()
            if path[-1] == t:
                out.add(v)
                break
            if len(path) > n:
                continue
            for w in adj(path[-1]):
                if w not in path and mark(path[-2], path[-1]) == "a" and mark(w, path[-1]) == "a":
                    stack.append([*path, w])
    return out


def test_paper_mmb_definition_omits_bidirected_neighbours() -> None:
    """Section 3.1's MMB is not a Markov blanket. A vertex joined to the target by a bidirected
    edge is adjacent, so the third clause excludes it, and it is neither parent nor child. Yet
    nothing can m-separate an adjacent vertex from the target, so it must be in any blanket.
    Adding those vertices reconciles the definition with the correct set exactly."""
    rng = np.random.default_rng(42)
    as_written_wrong = with_spouses_wrong = total = 0
    for _ in range(25):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        edges = _true_mag(dag, n_obs)
        for t in range(n_obs):
            truth = set(total_conditioning(oracle, t, alpha=0.5))
            total += 1
            as_written_wrong += int(_mmb_as_written(edges, t, n_obs, with_spouses=False) != truth)
            with_spouses_wrong += int(
                _mmb_as_written(edges, t, n_obs, with_spouses=True) != truth
            )
    assert as_written_wrong > 0, "the omission did not show up; widen the graph sample"
    assert with_spouses_wrong == 0, (
        f"adding bidirected neighbours still leaves {with_spouses_wrong} of {total} wrong"
    )


def test_adjacencies_at_target_match_global_fci() -> None:
    """Theorem 1 — the edges at the target are decidable inside MMB+(target)."""
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            loc = mmb_by_mmb(data, t, ci_test=oracle, alpha=0.5).endpoints
            got = {w for w in range(n_obs) if w != t and loc[t, w] != NO_EDGE}
            want = {w for w in range(n_obs) if w != t and glob[t, w] != NO_EDGE}
            assert got == want, f"target {t}: {sorted(got)} vs fci {sorted(want)}"
            checked += 1
    assert checked > 50


def test_committed_marks_at_target_match_global_fci() -> None:
    """Theorem 3 — on the edges at the target, no committed mark disagrees with global FCI."""
    rng = np.random.default_rng(11)
    wrong, committed = 0, 0
    for _ in range(12):
        dag, n_obs = _rand_latent_case(rng)
        oracle = DSeparationOracleProjected(dag, n_obs)
        data = np.zeros((5, n_obs))
        glob = fci(data, ci_test=oracle, alpha=0.5).endpoints
        for t in range(n_obs):
            loc = mmb_by_mmb(data, t, ci_test=oracle, alpha=0.5).endpoints
            for (i, j), mark in _marks_at(loc, t).items():
                if mark == CIRCLE:
                    continue
                committed += 1
                wrong += int(mark != int(glob[i, j]))
    assert committed > 0, "nothing was committed, the test proves nothing"
    assert wrong == 0, f"{wrong} of {committed} committed marks at the target disagree with fci"


def test_records_and_validates() -> None:
    dag = DAG.from_directed_edges(4, [(0, 2), (1, 2), (2, 3)])
    oracle = DSeparationOracleProjected(dag, 4)
    data = np.zeros((5, 4))
    rec = InMemoryRecorder()
    mmb_by_mmb(data, 2, ci_test=oracle, alpha=0.5, recorder=rec, run_id="M")
    frames = rec.to_frames()
    run = frames["runs"].iloc[0]
    assert run["algorithm"] == "mmb_by_mmb" and run["run_id"] == "M"
    assert len(frames["ci_tests"]) > 0
    assert len(frames["regions"]) > 0
    assert set(frames["ci_tests"]["phase"]) <= {"blanket", "skeleton", "pdsep_refinement"}
    with pytest.raises(CBCDInputError):
        mmb_by_mmb(data, 9, ci_test=oracle)
