"""MARVEL background-knowledge parity: ``marvel(bg) == pc(bg)`` on oracle DAGs.

Background knowledge is applied by ``marvel`` the same way ``pc`` applies it (forbidden adjacencies
pruned from the skeleton with their separating sets dropped; required/forbidden orientation and tier
ordering enforced by the shared ``SepsetOrienter`` + ``MeekRules``), so the two must return the
identical CPDAG. Every ``BackgroundKnowledge`` instance here is *derived from the true DAG* and thus
consistent with it (required = true edges, forbidden-directed = reversed true edges, forbidden-
adjacent = a true non-edge, tiers = a topological-depth partition), so it never contradicts the
ground truth. The reference-parity bar is ``pc`` with the same background -- not the no-background
CPDAG (background may legitimately change the essential graph).
"""

from __future__ import annotations

import collections

import numpy as np

from cbcd import InMemoryRecorder, marvel, pc
from cbcd.background import BackgroundKnowledge
from cbcd.graph.dag import DAG
from tests.oracle import DSeparationOracle
from tests.test_marvel import _rand_dag, _shd


def _tiers_from_dag(dag: DAG) -> tuple[frozenset[int], ...]:
    """Topological-depth tiers: tier(v) = longest path length from a root to ``v``. Every edge
    ``u->v`` then satisfies tier(u) < tier(v), so the partition is consistent with the DAG (it
    forbids only backward, i.e. non-existent, directed edges)."""
    n = dag.n_vars
    depth = [0] * n
    children = {i: list(dag.children(i)) for i in range(n)}
    indeg = [0] * n
    for i in range(n):
        for c in children[i]:
            indeg[c] += 1
    q = collections.deque(i for i in range(n) if indeg[i] == 0)
    while q:
        u = q.popleft()
        for c in children[u]:
            depth[c] = max(depth[c], depth[u] + 1)
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    md = max(depth) if n else 0
    tiers = tuple(frozenset(i for i in range(n) if depth[i] == d) for d in range(md + 1))
    return tuple(t for t in tiers if t)


def _nonedge_with_common_neighbour(dag: DAG) -> tuple[int, int] | None:
    """A true non-adjacent pair that shares a common neighbour (an unshielded triple's outer pair) --
    the interesting forbidden-adjacent case, since forbidding it suppresses a would-be v-structure."""
    n = dag.n_vars
    adj = dag.adjacency()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                continue
            for k in range(n):
                if k not in (i, j) and adj[i, k] and adj[j, k]:
                    return (i, j)
    return None


def _consistent_bks(dag: DAG, rng: np.random.Generator) -> list[tuple[str, BackgroundKnowledge]]:
    """A menu of DAG-consistent background instances exercising each BK facet and a combination."""
    edges = list(dag.directed_edges())
    ne = _nonedge_with_common_neighbour(dag)
    bks: list[tuple[str, BackgroundKnowledge]] = []
    if edges:
        req = frozenset(
            edges[k] for k in rng.choice(len(edges), size=min(2, len(edges)), replace=False)
        )
        bks.append(("required", BackgroundKnowledge(required_directed=req)))
        forb = frozenset(
            (b, a)
            for (a, b) in (edges[k] for k in rng.choice(len(edges), size=min(3, len(edges)), replace=False))
        )
        bks.append(("forbidden_directed", BackgroundKnowledge(forbidden_directed=forb)))
    if ne is not None:
        bks.append(
            ("forbidden_adjacent", BackgroundKnowledge(forbidden_adjacent=frozenset([frozenset(ne)])))
        )
    bks.append(("tiers", BackgroundKnowledge(tiers=_tiers_from_dag(dag))))
    if edges and ne is not None:
        forb = frozenset([(edges[1][1], edges[1][0])]) if len(edges) > 1 else frozenset()
        bks.append(
            (
                "combined",
                BackgroundKnowledge(
                    required_directed=frozenset([edges[0]]),
                    forbidden_directed=forb,
                    forbidden_adjacent=frozenset([frozenset(ne)]),
                    tiers=_tiers_from_dag(dag),
                ),
            )
        )
    return bks


def test_marvel_matches_pc_with_background() -> None:
    """On random DAGs, ``marvel`` and ``pc`` return the identical CPDAG under each consistent
    background instance (every BK facet + a combination). Any mismatch is a wiring bug."""
    rng = np.random.default_rng(99)
    checks = collections.Counter()
    for _ in range(60):
        p = int(rng.integers(7, 15))
        dag = _rand_dag(p, int(p * 1.3), rng)
        oracle = DSeparationOracle(dag)
        data = np.zeros((5, p))
        for name, bg in _consistent_bks(dag, rng):
            m = marvel(data, ci_test=oracle, alpha=0.5, background=bg)
            g = pc(data, ci_test=oracle, alpha=0.5, background=bg)
            assert _shd(m.endpoints, g.endpoints) == 0, (
                f"[{name}] p={p}: marvel(bg) != pc(bg)\nMARVEL:\n{m.endpoints}\nPC:\n{g.endpoints}"
            )
            checks[name] += 1
    # Every facet was actually exercised (guards against the menu silently skipping cases).
    for facet in ("required", "forbidden_directed", "forbidden_adjacent", "tiers", "combined"):
        assert checks[facet] > 0, f"{facet} never exercised"


def test_marvel_forbidden_adjacent_suppresses_collider_like_pc() -> None:
    """A hand-built case: forbidding a true collider's parent pair must drop that v-structure in
    *both* marvel and pc (pc has no separating set to classify it; marvel drops its recorded one)."""
    # 0 -> 2 <- 1 with 0,1 non-adjacent: forbidding {0,1} removes the 0->2<-1 orientation.
    dag = DAG.from_directed_edges(3, [(0, 2), (1, 2)])
    oracle = DSeparationOracle(dag)
    data = np.zeros((5, 3))
    bg = BackgroundKnowledge(forbidden_adjacent=frozenset([frozenset({0, 1})]))
    m = marvel(data, ci_test=oracle, alpha=0.5, background=bg)
    g = pc(data, ci_test=oracle, alpha=0.5, background=bg)
    assert _shd(m.endpoints, g.endpoints) == 0
    # And it genuinely differs from the no-background CPDAG (the collider is otherwise identified).
    m0 = marvel(data, ci_test=oracle, alpha=0.5)
    assert _shd(m.endpoints, m0.endpoints) != 0


def test_marvel_background_none_is_regression() -> None:
    """Passing ``background=None`` changes nothing: identical to the default call and to ``pc``."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        dag = _rand_dag(10, 13, rng)
        oracle = DSeparationOracle(dag)
        data = np.zeros((5, 10))
        a = marvel(data, ci_test=oracle, alpha=0.5)
        b = marvel(data, ci_test=oracle, alpha=0.5, background=None)
        g = pc(data, ci_test=oracle, alpha=0.5, background=None)
        assert _shd(a.endpoints, b.endpoints) == 0
        assert _shd(b.endpoints, g.endpoints) == 0


def test_marvel_forbidden_adjacent_does_not_change_ci_count() -> None:
    """forbidden_adjacent is a skeleton/orientation post-filter, not a CI-test pruner: MARVEL's
    recorded CI count is identical with and without it (a forbidden-adjacent pair may be co-parents,
    so its boundary tests are still needed for correct removability). Documents the (d) finding."""
    rng = np.random.default_rng(5)
    exercised = 0
    for _ in range(30):
        p = int(rng.integers(8, 14))
        dag = _rand_dag(p, int(p * 1.3), rng)
        ne = _nonedge_with_common_neighbour(dag)
        if ne is None:
            continue
        oracle = DSeparationOracle(dag)
        data = np.zeros((5, p))
        bg = BackgroundKnowledge(forbidden_adjacent=frozenset([frozenset(ne)]))
        r0, r1 = InMemoryRecorder(), InMemoryRecorder()
        marvel(data, ci_test=oracle, alpha=0.5, recorder=r0)
        marvel(data, ci_test=oracle, alpha=0.5, background=bg, recorder=r1)
        assert r1.metrics()["n_ci_total"] == r0.metrics()["n_ci_total"]
        exercised += 1
    assert exercised > 0
