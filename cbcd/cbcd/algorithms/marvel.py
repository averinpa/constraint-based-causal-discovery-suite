"""MARVEL (Mokhtarian et al. 2021; corpus algo81) — recursive Markov-boundary elimination.

Native, CI-test-efficient constraint-based discovery for cell 1 (full / i.i.d. / causally
sufficient -> CPDAG). MARVEL is the speed champion against the PC-stable anchor: it recovers the
same equivalence class with substantially fewer CI tests on sparse / bounded-in-degree graphs.

The algorithm recursively eliminates a *removable* variable (Def 3 / Thm 5 of the paper). Per
iteration it sorts the remaining variables ascending by ``|Mb_X|`` and scans for the first removable
one, testing removability with CI tests confined to subsets of the Markov boundary ``Mb_X`` -- this
boundedness is where the CI-test economy comes from (Lemma 12: a removable variable has
``|Mb_X| <= Delta_in``). Removability is decided by four lemmas:

  * **Lemma 7** — partition ``Mb_X`` into neighbours ``N_X`` and co-parents. ``Y`` is a neighbour iff
    no subset of ``Mb_X\\{Y}`` separates it from ``X``; otherwise it is a co-parent and we record the
    separating set ``S_XT``.
  * **Lemma 9 / Condition 1** — for every pair of neighbours ``Z, W`` and every ``S <= Mb_X\\{Z,W}``,
    ``Z`` stays dependent on ``W`` given ``S ∪ {X}``.
  * **Lemma 8** — v-structures ``X->Y<-T``: co-parent ``T`` (sepset ``S_XT``), neighbour ``Y`` with
    ``Y ∉ S_XT`` and ``Y`` inseparable from ``T`` over ``(Mb_X ∪ {X})\\{Y,T}``.
  * **Lemma 10 / Condition 2** — for each v-structure, each ``Z in N_X\\{Y}`` and ``S <= Mb_X\\{Z,Y,T}``,
    ``Z`` stays dependent on ``T`` given ``S ∪ {X,Y}``.

Both conditions holding makes ``X`` removable. We then record the skeleton edges ``X—N_X``, remove
``X``, and incrementally update the Markov boundaries of its neighbours (Eq 8): a neighbour pair
``{Y,Z}`` that becomes independent given ``Mb_W\\{X,Y,Z}`` (``W`` the smaller-boundary endpoint) is
dropped from each other's boundary, and their separating set is stored.

The recovered skeleton plus the collision information carried in the recorded separating sets is
completed exactly the way ``pc`` completes its skeleton — ``SepsetOrienter`` (unshielded-collider
rule) followed by ``MeekRules`` closure — so the returned CPDAG is, by Thm 13, identical to the one
``pc`` returns on the same (oracle-noise-free) input.

Optimisation notes (this is the point of MARVEL): every CI query is routed through the run's cached
CI test (duplicate ``(x, y, S)`` triples are never recomputed) and through the recorder (so the
efficiency win is *measured*, not asserted); removability verdicts are memoised per variable and
recomputed only for variables whose boundary actually changed (the eliminated variable's former
neighbours); subset enumeration stops as soon as a lemma's "for all S ... dependent" clause is
violated; and every conditioning set is bounded by the Markov boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd._run import iid_run
from cbcd.algorithms._recursive import _CI, EliminationInfo, find_adjacent, recursive_skeleton
from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.collider import SepsetOrienter
from cbcd.exceptions import CBCDInputError
from cbcd.graph.cpdag import CPDAG
from cbcd.mb import grow_shrink, iamb, inter_iamb
from cbcd.recording import RunRecorder
from cbcd.rules import MeekRules
from cbcd.skeleton import Skeleton

_MB_ALGOS = {
    "grow_shrink": grow_shrink,
    "iamb": iamb,
    "inter_iamb": inter_iamb,
}


def _analyze(x: int, mb_x: set[int], ci: _CI) -> EliminationInfo:
    """Decide whether ``x`` is removable given its Markov boundary ``mb_x`` (Lemmas 7-10).

    Returns the verdict together with the neighbour set (``adjacent``) and the co-parent separating
    sets (``coparents``) -- everything the elimination loop needs to lay down ``x``'s skeleton edges
    and collider separating sets on removal. The v-structures ``x->y<-t`` used by Conditions 1/2 are
    internal to the removability decision and are re-derived at orientation time from the sepsets.
    """
    mb = sorted(mb_x)

    # --- Lemma 7 (shared FindAdjacent): neighbours vs co-parents, recording co-parent sepsets. ---
    neighbours, coparents = find_adjacent(x, mb_x, ci)

    # --- Lemma 9 / Condition 1: no S ∪ {x} separates two neighbours. ---
    for i in range(len(neighbours)):
        z = neighbours[i]
        for j in range(i + 1, len(neighbours)):
            w = neighbours[j]
            pool = [v for v in mb if v != z and v != w]
            if ci.separable_with(z, w, pool, extra=(x,)):
                return EliminationInfo(False, neighbours, coparents)

    # --- Lemma 8: v-structures x->y<-t for co-parent t, neighbour y (y adjacent to t). ---
    mb_plus_x = mb + [x]
    vstructures: list[tuple[int, int]] = []
    for t, sep_xt in coparents.items():
        for y in neighbours:
            if y in sep_xt:
                continue
            pool = [v for v in mb_plus_x if v != y and v != t]
            if ci.find_sepset(y, t, pool) is None:  # y and t inseparable -> adjacent
                vstructures.append((y, t))

    # --- Lemma 10 / Condition 2: no S ∪ {x, y} separates a neighbour z from a co-parent t. ---
    for y, t in vstructures:
        for z in neighbours:
            if z == y:
                continue
            pool = [v for v in mb if v != z and v != y and v != t]
            if ci.separable_with(z, t, pool, extra=(x, y)):
                return EliminationInfo(False, neighbours, coparents)

    return EliminationInfo(True, neighbours, coparents)


def _update_mb(
    chosen: int,
    info: EliminationInfo,
    mb: dict[int, set[int]],
    sepsets: dict[frozenset[int], tuple[int, ...]],
    remaining: set[int],
    ci: _CI,
) -> set[int]:
    """MARVEL boundary update (Eq 8): drop a neighbour pair ``{y,z}`` from each other's boundary when
    they become independent given ``Mb_W\\{chosen,y,z}`` (``W`` the smaller-boundary endpoint), and
    store their separating set. Returns the variables whose boundary changed."""
    touched: set[int] = set()
    nbrs = [y for y in info.adjacent if y in remaining]
    for a in range(len(nbrs)):
        y = nbrs[a]
        for b in range(a + 1, len(nbrs)):
            z = nbrs[b]
            if z not in mb[y]:  # already separated
                continue
            w = y if len(mb[y]) <= len(mb[z]) else z
            cond = sorted(mb[w] - {chosen, y, z})
            if ci.indep(y, z, cond):
                mb[y].discard(z)
                mb[z].discard(y)
                sepsets.setdefault(frozenset({y, z}), tuple(cond))
                touched.add(y)
                touched.add(z)
    return touched


def marvel(
    data: NDArray[np.float64] | pd.DataFrame,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    mb_algo: Literal["grow_shrink", "iamb", "inter_iamb"] = "grow_shrink",
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> CPDAG:
    """MARVEL: recursive Markov-boundary elimination -> CPDAG.

    Recovers the same essential graph as :func:`cbcd.pc` (Thm 13) with far fewer CI tests on sparse
    graphs, by confining every conditioning set to a Markov boundary and eliminating variables in
    ascending boundary-size order.

    Parameters mirror the shared algorithm conventions (``ci_test``, ``alpha``, ``background``,
    ``recorder``, ``run_id``). ``mb_algo`` selects the initial Markov-boundary routine (grow-shrink by
    default; also ``iamb`` / ``inter_iamb``). ``max_cond_set`` caps the Markov-boundary size used
    during the initial boundary pass.

    ``background`` is applied exactly as :func:`cbcd.pc` applies it, so MARVEL-with-background returns
    the identical CPDAG. Forbidden adjacencies are removed from the recovered skeleton and their
    separating sets dropped -- mirroring ``PCStable``, which pre-removes those edges and records no
    separating set (so an unshielded triple whose outer pair is forbidden-adjacent is left
    unoriented, since there is no witness to classify it). Required/forbidden directed edges and tier
    ordering are honoured downstream by ``SepsetOrienter`` and ``MeekRules`` -- the same phases, with
    the same ``background``, that ``pc`` threads them through. Like ``pc``/``PCStable``, MARVEL does
    *not* force a required edge into the skeleton against the CI tests; under background consistent
    with a faithful DAG a required edge is a true edge and survives on its own.
    """
    if mb_algo not in _MB_ALGOS:
        raise CBCDInputError(
            f"unknown mb_algo {mb_algo!r}; known: {sorted(_MB_ALGOS)}"
        )
    mb_fn = _MB_ALGOS[mb_algo]

    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="marvel",
        params={"alpha": alpha, "mb_algo": mb_algo, "max_cond_set": max_cond_set},
        alpha=alpha,
        var_names=var_names,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n = ctx.n_vars
        rec = ctx.rec
        ci = _CI(ctx.ci, rec, alpha)

        # --- Initialisation: Markov boundary of every variable (the recursion's input). ---
        mb: dict[int, set[int]] = {
            v: set(mb_fn(ctx.ci, v, alpha=alpha, max_cond_set=max_cond_set, recorder=rec))
            for v in range(n)
        }

        # --- Recursive elimination (shared core): removability = MARVEL Lemmas 7-10, boundary
        #     update = Eq 8. Yields the skeleton adjacency + co-parent/collider separating sets. ---
        adj, sepsets = recursive_skeleton(
            n, mb, ci, analyze=_analyze, update_mb=_update_mb
        )

        # Background: prune forbidden adjacencies from the skeleton and drop *any* separating set
        # recorded for such a pair, reproducing PCStable's output exactly -- PCStable pre-removes the
        # edge and never searches, so it records no witness. This must happen even when the pair was
        # already non-adjacent here: MARVEL records a separating set for every co-parent pair (needed
        # for its removability tests), and a forbidden-adjacent co-parent's leftover witness would
        # otherwise let SepsetOrienter classify an unshielded triple that pc leaves unoriented.
        # Note this does not save CI tests: forbidden-adjacency means "no edge", not "conditionally
        # independent" (the pair may be co-parents), so the boundary / co-parent CI tests are still
        # required for correct removability. The elimination above is therefore background-agnostic.
        if background is not None:
            for pair in background.forbidden_adjacent:
                if len(pair) != 2:
                    continue
                i, j = sorted(pair)
                if i < 0 or j >= n:
                    continue
                adj[i, j] = adj[j, i] = False
                sepsets.pop(frozenset({i, j}), None)

        # --- Completion: identical path to pc() -- unshielded colliders then Meek closure. ---
        skel = Skeleton(n_vars=n, adj=adj, sepsets=sepsets, pvalues_max=None)
        decisions = SepsetOrienter()(skel, ctx.ci, alpha=alpha, background=background, recorder=rec)
        partial = decisions.apply_to_cpdag(skel, var_names=ctx.names)
        ctx.result = MeekRules()(partial, background=background, recorder=rec)

    return ctx.result
