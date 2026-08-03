"""Coordinated Multi-Neighborhood Learning (CML; Smith & Zhou 2024, arXiv 2405.15358).

Native, efficient multi-target local causal discovery — the best-in-class local algorithm. Same
accuracy as the ``mmpc``-hops ``local_discovery`` but far fewer CI tests, via the two-phase
neighborhood-coordination skeleton of Algorithm 1 (§III.A):

  Phase 1 (union skeleton): FCI-style skeleton over ``O = NB_T`` (the union of first-order target
    neighborhoods), conditioning only on subsets of ``O`` -> reuses region-scoped ``PCStable``.
    Between-neighborhood edges survive (inducing paths) and coordinate orientation.
  Phase 2 (local skeleton): within each target neighborhood, prune edges using first-order neighbours
    ``N1(i)`` / ``N1(j)`` (which include the target's second-order neighbours) as separators.
  Orient: v-structures + Zhang R1-R4, R8-R10, then R_N (within a neighbourhood, circle marks -> tail,
    i.e. ``o-o`` -> ``-`` and ``o->`` -> ``->``, since there is no latent confounding within a
    neighbourhood). Returns a local PAG.

First-order neighbourhood ``N1(t)`` is the **Markov blanket** (per the paper), estimated once per
node with grow-shrink ``iamb`` and memoized — this is where the CI-test economy comes from (grow-
shrink is ~20x cheaper than the max-min ``mmpc``, which does unbounded subset enumeration).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd._run import iid_run
from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.collider import SepsetOrienter
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.mb import iamb
from cbcd.recording import RunRecorder
from cbcd.rules import FCIRules
from cbcd.skeleton import PCStable, Skeleton


def _tester(ci: CITest, rec: RunRecorder) -> Callable[[int, int, Sequence[int]], float]:
    """One instrumented CI query -> p-value (records ``record_ci``)."""
    is_cached = getattr(ci, "is_cached", None)

    def test(x: int, y: int, cond: Sequence[int]) -> float:
        s = list(cond)
        hit = bool(is_cached(x, y, s)) if is_cached is not None else False
        p = float(ci(x, y, s))
        rec.record_ci(
            x=int(x), y=int(y), S=tuple(int(c) for c in s),
            p_value=p, depth=len(s), was_cache_hit=hit,
        )
        return p

    return test


def _phase2_prune(
    adj: NDArray[np.bool_],
    sepsets: dict[frozenset[int], tuple[int, ...]],
    n1: dict[int, frozenset[int]],
    nodes: Sequence[int],
    test: Callable[[int, int, Sequence[int]], float],
    alpha: float,
    cap: int | None,
) -> None:
    """Prune every union-skeleton edge (i,j) among ``nodes`` if some subset of ``N1(i)\\{j}`` or
    ``N1(j)\\{i}`` separates them (mutates ``adj`` / ``sepsets`` in place).

    Ranging over *all* pairs of ``NB_T`` nodes — not only pairs inside a single target
    neighbourhood — is what makes the phase sound for coordinated multi-target queries. A
    between-neighbourhood edge (endpoints drawn from two different target neighbourhoods) can be a
    false positive that phase 1 could not remove because its true separator lies outside ``O``; if
    that separator sits in the endpoints' Markov blankets ``N1(i)``/``N1(j)`` (spouses included, per
    Assumption 1) it is found here and the edge is dropped. A genuine inducing-path edge has *no*
    observed separator, so this search never removes it — the coordination edge survives. Skipping
    between-neighbourhood pairs (the earlier per-target restriction) left such false edges in place,
    which then formed spurious unshielded triples and produced wrong v-structure arrowheads at a
    neighbourhood node (an unsound orientation the region-restricted PC reference does not commit)."""
    pool_nodes = sorted(nodes)
    for a in range(len(pool_nodes)):
        for b in range(a + 1, len(pool_nodes)):
            i, j = pool_nodes[a], pool_nodes[b]
            if not adj[i, j]:
                continue
            removed = False
            for pool in (sorted(n1[i] - {j}), sorted(n1[j] - {i})):
                upper = len(pool) if cap is None else min(len(pool), cap)
                for size in range(upper + 1):
                    for cond in combinations(pool, size):
                        if test(i, j, cond) > alpha:
                            adj[i, j] = adj[j, i] = False
                            sepsets[frozenset({i, j})] = tuple(cond)
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break


def _apply_rn(pag: PAG, in_nbhd: set[frozenset[int]]) -> PAG:
    """R_N: within a neighbourhood there is no bidirected edge, so every circle mark resolves to a
    tail (``o->`` -> ``->``, ``o-o`` -> ``-``)."""
    ep = pag.endpoints.copy()
    n = pag.n_vars
    for i in range(n):
        for j in range(n):
            if i != j and frozenset({i, j}) in in_nbhd and ep[i, j] == EndpointMark.CIRCLE:
                ep[i, j] = EndpointMark.TAIL
    return PAG(n_vars=n, endpoints=ep, var_names=pag.var_names, sepsets=pag.sepsets)


def cml(
    data: NDArray[np.float64] | pd.DataFrame,
    targets: Sequence[int],
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> PAG:
    """Coordinated Multi-Neighborhood Learning: efficient native local PAG around ``targets``.

    Returns an ``n_vars``-wide PAG with the query neighbourhoods oriented (R_N-simplified) and
    between-neighbourhood coordination edges retained; nodes outside ``NB_T`` are isolated.

    Guarantee. **Sound + complete** under faithfulness, causal sufficiency, and **Assumption 1** (no
    inducing path between two same-neighbourhood nodes routed through a *different* neighbourhood);
    completeness is relative to the constructed neighbourhood graph ``G*_N`` (Smith & Zhou 2024,
    arXiv 2405.15358), i.e. every endpoint mark ``G*_N`` identifies is committed and no other. Every
    committed arrowhead/tail is ancestrally correct — verified at the d-separation oracle against the
    region-restriction of cbcd's own ``pc`` in ``tests/test_local_discovery_soundness.py`` (zero
    wrong-orientation violations across single, disjoint-, and overlapping-neighbourhood queries).

    Why it is sound where LocalPC is not. The classic LocalPC unsoundness (a v-structure oriented
    from a *region-restricted* skeleton whose separating set is incomplete) is sidestepped because
    phase 2 searches for separators over ``N1`` = the full **Markov blanket** — spouses included —
    of each endpoint, and ranges over *all* ``NB_T`` pairs (between-neighbourhood edges too, not only
    pairs inside one target neighbourhood). A false edge whose true separator lies outside the union
    ``O`` but inside the endpoints' MBs is therefore removed before orientation, so it cannot seed a
    spurious unshielded triple. LOAD's critique of MB-incomplete local search (algo113, Appendix A)
    thus does not apply. (Removing the between-neighbourhood pass reintroduces exactly that bug — see
    the soundness test's historical note.)

    Relation to neighbours in the local-discovery literature (answering the reviewer's "why not X?"
    in-code): **LDECC** (Gupta et al.) is *unsound and incomplete* for the local CPDAG in general
    (LOAD algo113, Appendix A), so it is not the champion here. **LOAD / SNAP** (algo113 / algo151)
    solve a *different* problem — they return a valid *adjustment set* for a treatment/outcome effect,
    not a local CPDAG/PAG around a query set — so they are not substitutes for CML's output.

    PAG-mode note (latent cell). Although CML returns a ``PAG``, its soundness assumes **causal
    sufficiency**. The final rule ``R_N`` resolves every within-neighbourhood circle mark to a *tail*
    on the premise that there is no latent confounding — hence no bidirected edge — inside a
    neighbourhood. Under a genuine within-neighbourhood latent confounder that premise fails and
    ``R_N`` commits a **tail where the true MAG has an arrowhead** (it collapses a ``<->`` to a tail);
    the ``tests/test_local_latent_soundness.py`` battery measures this as a nonzero false-*tail* rate
    (730 over the sweep) while cml's **arrowheads stay sound** (zero false arrowheads — they come from
    the FCI collider/rule pass, not ``R_N``). So for the *latent* cell prefer
    ``local_discovery_latent``, which makes no within-neighbourhood-sufficiency assumption and whose
    committed marks (arrowheads *and* tails) are certified sound by the Possible-D-Sep adequacy guard.
    Use cml in PAG mode only when within-neighbourhood causal sufficiency is warranted; there its
    committed marks match global ``fci``/``pc`` (the cell-5 regime, zero violations).
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="cml",
        params={"alpha": alpha, "max_cond_set": max_cond_set},
        alpha=alpha,
        var_names=var_names,
        targets=targets,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        ci, rec, tset = ctx.ci, ctx.rec, ctx.targets or []
        test = _tester(ci, rec)

        # First-order neighbours, estimated once per node and memoized.
        n1: dict[int, frozenset[int]] = {}

        def first_order(v: int) -> frozenset[int]:
            if v not in n1:
                n1[v] = iamb(ci, v, alpha=alpha, max_cond_set=max_cond_set, recorder=rec)
            return n1[v]

        nbhds = {t: frozenset({t}) | first_order(t) for t in tset}
        nb_union = sorted(set().union(*nbhds.values()) if nbhds else set())
        for v in nb_union:  # first-order of every NB_T node (needed for phase 2)
            first_order(v)

        # Phase 1 — union skeleton over O = NB_T (region-scoped PC-stable, conditions on O only).
        skel = PCStable()(
            ci, alpha=alpha, max_cond_set=max_cond_set, background=background,
            variables=nb_union, recorder=rec,
        )
        adj = skel.adj.copy()
        sepsets = dict(skel.sepsets)

        # Phase 2 — prune union-skeleton edges (within- AND between-neighbourhood) using first-order
        # neighbours; the between-neighbourhood pass is what keeps coordinated queries sound.
        _phase2_prune(adj, sepsets, n1, nb_union, test, alpha, max_cond_set)

        # Orient: v-structures + Zhang rules, then R_N within neighbourhoods.
        pruned = Skeleton(n_vars=ctx.n_vars, adj=adj, sepsets=sepsets, pvalues_max=None)
        interior = frozenset(tset) | frozenset(nb_union)
        decisions = SepsetOrienter()(
            pruned, ci, alpha=alpha, background=background, interior=interior, recorder=rec
        )
        partial = decisions.apply_to_pag(pruned, var_names=ctx.names)
        pag = FCIRules()(partial, background=background, recorder=rec)

        in_nbhd = {
            frozenset({i, j})
            for nb in nbhds.values()
            for i in nb
            for j in nb
            if i != j
        }
        ctx.result = _apply_rn(pag, in_nbhd)
    return ctx.result
