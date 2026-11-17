"""LatentLCD — local causal discovery without causal sufficiency.

Ling, Yu, Zhang, Cheng, Zhou, Wu, Jiang and Yu (2024). The smallest of the latent local methods.
It does not walk outward. It asks for the variables adjacent to the target, decides the mark on
each of those edges, and stops.

The object it returns is LocalMAG(T), which splits the neighbours of T three ways instead of two
(Definition 5)::

    Pa(T)   parents
    Ch(T)   children
    EL(T)   "effects of latent causes" — joined to T by a bidirected edge

The third group is the point. Under sufficiency it is empty and the object collapses to the
parents and children (Theorem 3). Without sufficiency its members are adjacent to T with an
arrowhead at T, exactly as a parent is, so a method that assumes sufficiency returns them and
calls them causes. They are neither cause nor effect.

Three steps, from the Methodology section:

1. ``mmpc`` for adj(T), which Proposition 1 says already contains EL(T) — the adjacency step
   needs no change, only the reading of it does;
2. orient the edges at T from the unshielded colliders, then ``FindBI`` to separate the parents
   from EL(T);
3. Zhang's R1-R4 and R8-R10 for the rest.

HOW FindBI WORKS. A single collider at T puts an arrowhead at T and says nothing about the far
end. Two overlapping colliders, one at T and one at the neighbour, put an arrowhead at both ends
of the same edge, and an edge with two arrowheads is bidirected. So the test for a hidden common
cause is a second collider one step further out (Proposition 3).

PAPER TYPO. Proposition 3 reads "Assuming {X, Y} subset of adj(T) and Z in adj(X)". For the
triple {T, Y, Z} to be an unshielded triple with Y in the middle, Z has to be adjacent to Y, and
the proof confirms it by conditioning on sep(T, Z) with Y as the collider. Implemented as
adj(Y).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd._run import iid_run
from cbcd.algorithms._sequential import ARROW, CIRCLE, NO_EDGE, harvest, settled_subgraph
from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError
from cbcd.graph.pag import PAG, PartialPAG
from cbcd.mb import mmpc
from cbcd.recording import RunRecorder
from cbcd.rules import FCIRules

PCLearner = Callable[..., "frozenset[int]"]

# Zhang's rules minus R5-R7, which only apply when selection variables are present. LatentLCD
# allows latents but not selection, and the paper names exactly this subset.
_LATENTLCD_RULES = frozenset({"R1", "R2", "R3", "R4", "R8", "R9", "R10"})


def _find_sepset(
    ci: CITest,
    x: int,
    y: int,
    pools: Sequence[Sequence[int]],
    alpha: float,
    cap: int | None,
    rec: RunRecorder,
) -> tuple[int, ...] | None:
    """Smallest subset of either pool that separates ``x`` and ``y``, or None."""
    is_cached = getattr(ci, "is_cached", None)
    for pool in pools:
        cand = [v for v in pool if v not in (x, y)]
        upper = len(cand) if cap is None else min(len(cand), cap)
        for size in range(upper + 1):
            for S in itertools.combinations(sorted(cand), size):
                hit = bool(is_cached(x, y, list(S))) if is_cached is not None else False
                p = ci(x, y, list(S))
                rec.record_ci(
                    x=x,
                    y=y,
                    S=S,
                    p_value=p,
                    depth=size,
                    was_cache_hit=hit,
                    phase="latentlcd_sepset",
                )
                if p > alpha:
                    return S
    return None


def _dependent_given(
    ci: CITest, x: int, y: int, cond: Sequence[int], alpha: float, rec: RunRecorder
) -> bool:
    """Equation 1 — the second half of the collider test, ``X`` and ``Y`` dependent given
    ``sep(X, Y)`` with the middle vertex added."""
    is_cached = getattr(ci, "is_cached", None)
    S = tuple(sorted(cond))
    hit = bool(is_cached(x, y, list(S))) if is_cached is not None else False
    p = ci(x, y, list(S))
    rec.record_ci(
        x=x, y=y, S=S, p_value=p, depth=len(S), was_cache_hit=hit, phase="latentlcd_collider"
    )
    return p <= alpha


def latentlcd(
    data: NDArray[np.float64] | pd.DataFrame,
    target: int,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    pc_learner: PCLearner = mmpc,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> PAG:
    """LatentLCD (Ling et al., 2024). LocalMAG around ``target``, latents allowed.

    Returns an ``n_vars``-wide PAG. A bidirected edge at the target marks a member of EL(T), a
    neighbour whose association with the target runs through a hidden common cause rather than
    through a direct effect in either direction.

    ``pc_learner`` defaults to ``mmpc``, which is what the paper specifies. Proposition 1 is what
    licenses leaving the adjacency step unchanged.
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="latentlcd",
        params={
            "alpha": alpha,
            "max_cond_set": max_cond_set,
            "pc_learner": getattr(pc_learner, "__name__", str(pc_learner)),
        },
        alpha=alpha,
        var_names=var_names,
        targets=[target],
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        t = int(target)
        if not (0 <= t < ctx.n_vars):
            raise CBCDInputError(f"target must be in [0, {ctx.n_vars}); got {target}")
        n = ctx.n_vars
        rec = ctx.rec

        # --- step 1: adj(T), and the neighbours of each neighbour, which FindBI needs ---------
        adj_t = sorted(pc_learner(ctx.ci, t, alpha=alpha, max_cond_set=max_cond_set, recorder=rec))
        adj: dict[int, list[int]] = {t: adj_t}
        for v in adj_t:
            adj[v] = sorted(
                pc_learner(ctx.ci, v, alpha=alpha, max_cond_set=max_cond_set, recorder=rec)
            )
        ctx.rec.record_region(target=t, region=sorted({t, *adj_t}), boundary=adj_t)

        ep = np.full((n, n), NO_EDGE, dtype=np.int8)
        for a, nbrs in adj.items():
            for b in nbrs:
                ep[a, b] = ep[b, a] = CIRCLE

        def adjacent(a: int, b: int) -> bool:
            return bool(ep[a, b] != NO_EDGE)

        # --- step 2: colliders, from the unshielded triples of the local graph ---------------
        # Every unshielded collider in the local graph is oriented, not only those at T. Zhang's
        # rules in step 3 are sound only on a graph whose v-structures are all in place; orienting
        # just the ones at T and then running R1-R10 gave 2 wrong marks in 59 at an oracle, and
        # R3 was the rule that misfired every time (found from the `rules` table).
        settled = {t, *adj_t}
        sepsets: dict[frozenset[int], tuple[int, ...]] = {}
        for x, y in itertools.combinations(sorted(settled), 2):
            if adjacent(x, y):
                continue
            S = _find_sepset(
                ctx.ci, x, y, (adj.get(x, []), adj.get(y, [])), alpha, max_cond_set, rec
            )
            if S is None:
                continue
            sepsets[frozenset((x, y))] = S
            rec.record_adjacency(
                x=x, y=y, decision="deleted", depth=len(S), sepset=S, phase="latentlcd_sepset"
            )
            for m in sorted(settled):
                if m in (x, y) or not (adjacent(x, m) and adjacent(y, m)) or m in S:
                    continue
                if _dependent_given(ctx.ci, x, y, (*S, m), alpha, rec):
                    ep[x, m] = ARROW
                    ep[y, m] = ARROW
                    rec.record_collider(
                        triple=(x, m, y), classification="collider", orienter="sepset"
                    )

        # --- FindBI: a second collider at the neighbour makes the edge bidirected -------------
        candidates = [y for y in adj_t if ep[y, t] == ARROW]
        for y in candidates:
            for z in adj.get(y, []):
                if z == t or adjacent(t, z):
                    continue  # {T, Y, Z} must be an unshielded triple
                S = _find_sepset(
                    ctx.ci, t, z, (adj.get(t, []), adj.get(z, [])), alpha, max_cond_set, rec
                )
                if S is None or y in S:
                    continue
                if _dependent_given(ctx.ci, t, z, (*S, y), alpha, rec):
                    ep[t, y] = ARROW  # arrowhead at BOTH ends -> bidirected -> y in EL(T)
                    rec.record_collider(
                        triple=(t, y, z), classification="collider", orienter="findbi"
                    )
                    break

        # --- step 3: Zhang's rules, on the part whose adjacency is known ---------------------
        closed = np.asarray(
            FCIRules(rules=_LATENTLCD_RULES)(
                PartialPAG(
                    n_vars=n,
                    endpoints=settled_subgraph(ep, settled, n),
                    var_names=ctx.names,
                    sepsets=sepsets,
                ),
                background=background,
                recorder=rec,
            ).endpoints,
            dtype=np.int8,
        )
        ep = harvest(ep, closed, settled)

        ctx.result = PAG(n_vars=n, endpoints=ep, var_names=ctx.names, sepsets=sepsets)
    return ctx.result
