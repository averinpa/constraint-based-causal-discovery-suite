"""L-MARVEL (Akbari, Mokhtarian, Ghassami, Kiyavash, NeurIPS 2021; corpus algo147) -> PAG.

The cell-2 (full / i.i.d. / causally *non-sufficient*) member of the recursive Markov-boundary
elimination family: an FCI-analog that handles latent confounders and returns a PAG, and the
CI-test-efficiency champion against the FCI anchor on sparse / bounded-degree graphs.

It shares the recursive elimination core with :func:`cbcd.marvel` (see ``_recursive.py``) -- the same
sort-by-boundary loop and the same ``FindAdjacent`` boundary partition -- and differs in two places:

1. **Removability = Theorem 2 (the MAG version).** ``X`` is removable iff for every ``Y in Adj(X)``
   and every ``Z in Mb(X)`` at least one of these holds (checked in this order):

     * **C1:** some ``W ⊆ Mb(X)\\{Y,Z}`` gives ``Y ⫫ Z | W``;
     * **C2:** every ``W ⊆ Mb(X)\\{Y,Z}`` keeps ``Y`` and ``Z`` dependent given ``W ∪ {X}``.

   If some ``(Y,Z)`` violates both, ``X`` is not removable this round.
2. **Orientation = FCI rules R0-R10.** The recovered adjacency + separating-set store ``A`` is fed
   straight into the FCI orientation path (``SepsetOrienter`` collider rule R0, then Zhang's R1-R10)
   -- the *same* orientation :func:`cbcd.fci` runs. This substitution is exact here (unlike MARVEL's
   Meek case) because FCI orientation is entirely sepset-driven and ``A`` supplies the sepsets; by
   Theorem 3 the result is the identical PAG to ``fci`` on the same (oracle-noise-free) input.

Separating-set store ``A``: initialised so that for every ``X`` and ``Y ∉ Mb(X)`` the set ``Mb(X)``
is recorded as a witness (the Markov property guarantees ``X ⫫ Y | Mb(X)``); ``FindAdjacent`` adds
co-parent witnesses; the boundary update adds witnesses for pairs it separates. Every non-adjacent
pair therefore carries a valid m-separating witness, which is all FCI's collider and
discriminating-path rules need.

Scope: **latents only, no selection bias** (``S = ∅``). Theorem 2 requires the selection-induced
undirected subgraph to be chordal; with ``S = ∅`` there are no undirected edges so it holds trivially,
matching :func:`cbcd.fci`'s default regime. Selection bias is a future extension (it needs the
chordality check plus undirected-edge handling in the boundary update). Caveat: finite-sample
Markov-boundary errors propagate (as in MARVEL); the ``2^|Mb|`` factor is bounded by the in-degree of
removable vars, so the win is on sparse / bounded-degree MAGs and degrades on dense ones.
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
from cbcd.graph.pag import PAG
from cbcd.mb import grow_shrink, iamb, inter_iamb
from cbcd.recording import RunRecorder
from cbcd.rules import FCIRules
from cbcd.skeleton import Skeleton

_MB_ALGOS = {
    "grow_shrink": grow_shrink,
    "iamb": iamb,
    "inter_iamb": inter_iamb,
}


def _analyze(x: int, mb_x: set[int], ci: _CI) -> EliminationInfo:
    """L-MARVEL removability (Theorem 2). Returns the verdict with ``x``'s adjacencies and co-parent
    separating sets (recorded for orientation)."""
    mb = sorted(mb_x)

    # Shared FindAdjacent: which boundary members are truly adjacent vs co-parents (with witnesses).
    adjacent, coparents = find_adjacent(x, mb_x, ci)

    # Theorem 2: for every Y in Adj(X), Z in Mb(X), require C1 or C2; else not removable.
    for y in adjacent:
        for z in mb:
            if z == y:
                continue
            pool = [v for v in mb if v != y and v != z]
            # C1: some W ⊆ Mb(X)\{Y,Z} separates Y, Z.
            if ci.find_sepset(y, z, pool) is not None:
                continue
            # C2: no W ⊆ Mb(X)\{Y,Z} separates Y, Z given W ∪ {X} (they stay dependent).
            if not ci.separable_with(y, z, pool, extra=(x,)):
                continue
            return EliminationInfo(False, adjacent, coparents)

    return EliminationInfo(True, adjacent, coparents)


def _update_mb(
    chosen: int,
    info: EliminationInfo,
    mb: dict[int, set[int]],
    sepsets: dict[frozenset[int], tuple[int, ...]],
    remaining: set[int],
    ci: _CI,
) -> set[int]:
    """L-MARVEL boundary update: for each pair ``{y,z}`` in the removed variable's boundary that is
    still mutual, one CI test ``y ⫫ z | Mb_W\\{chosen,y,z}`` (``W`` the smaller-boundary endpoint);
    on independence, drop them from each other's boundary and store the witness. Returns the changed
    variables."""
    touched: set[int] = set()
    members = [v for v in (set(info.adjacent) | set(info.coparents)) if v in remaining]
    for a in range(len(members)):
        y = members[a]
        for b in range(a + 1, len(members)):
            z = members[b]
            if z not in mb[y]:  # not mutually in-boundary -> nothing to drop
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


def _init_sepsets(
    n: int, mb: dict[int, set[int]]
) -> dict[frozenset[int], tuple[int, ...]]:
    """Seed the separating-set store: for every ``X`` and ``Y ∉ Mb(X)``, ``Mb(X)`` separates them (the
    Markov property), so record it. No CI tests -- pure bookkeeping that guarantees every 'far' pair
    has a witness for the FCI orientation."""
    sep: dict[frozenset[int], tuple[int, ...]] = {}
    for x in range(n):
        cond = tuple(sorted(mb[x]))
        mbx = mb[x]
        for y in range(n):
            if y == x or y in mbx:
                continue
            sep.setdefault(frozenset({x, y}), cond)
    return sep


def lmarvel(
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
) -> PAG:
    """L-MARVEL: recursive Markov-boundary elimination -> PAG (latents, no selection bias).

    Recovers the same PAG as :func:`cbcd.fci` (Theorem 3) with far fewer CI tests on sparse graphs,
    by confining every conditioning set to a Markov boundary and eliminating variables in ascending
    boundary-size order, then orienting with the identical FCI rule set.

    Parameters mirror the ``marvel`` / ``fci`` conventions. ``mb_algo`` selects the initial
    Markov-boundary routine (grow-shrink by default; also ``iamb`` / ``inter_iamb``). ``max_cond_set``
    caps the Markov-boundary size in the initial boundary pass.

    ``background`` is threaded exactly as :func:`cbcd.fci` threads it: forbidden adjacencies are pruned
    from the recovered skeleton (with their witnesses dropped, matching FAS's pre-removal), and
    required/forbidden orientation and tier ordering are enforced by the shared ``SepsetOrienter`` and
    ``FCIRules``. So ``lmarvel(background=bg)`` returns the identical PAG to ``fci(background=bg)``.
    """
    if mb_algo not in _MB_ALGOS:
        raise CBCDInputError(f"unknown mb_algo {mb_algo!r}; known: {sorted(_MB_ALGOS)}")
    mb_fn = _MB_ALGOS[mb_algo]

    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="lmarvel",
        params={"alpha": alpha, "mb_algo": mb_algo, "max_cond_set": max_cond_set},
        alpha=alpha,
        var_names=var_names,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        n = ctx.n_vars
        rec = ctx.rec
        ci = _CI(ctx.ci, rec, alpha)

        # --- Initialisation: Markov boundary of every variable, and the 'far pair' witnesses. ---
        mb: dict[int, set[int]] = {
            v: set(mb_fn(ctx.ci, v, alpha=alpha, max_cond_set=max_cond_set, recorder=rec))
            for v in range(n)
        }
        init_sep = _init_sepsets(n, mb)

        # --- Recursive elimination (shared core): removability = Theorem 2, boundary update as above.
        adj, sepsets = recursive_skeleton(
            n, mb, ci, analyze=_analyze, update_mb=_update_mb, init_sepsets=init_sep
        )

        # Background: prune forbidden adjacencies and drop their witnesses, reproducing FAS's output
        # (edge pre-removed, no witness). As in MARVEL this saves no CI tests -- forbidden-adjacency
        # is "no edge", not "conditionally independent"; the elimination is background-agnostic.
        if background is not None:
            for pair in background.forbidden_adjacent:
                if len(pair) != 2:
                    continue
                i, j = sorted(pair)
                if i < 0 or j >= n:
                    continue
                adj[i, j] = adj[j, i] = False
                sepsets.pop(frozenset({i, j}), None)

        # --- Completion: identical orientation to fci() -- FCI colliders (R0) then Zhang R1-R10. The
        #     recovered skeleton is already the (refined) MAG skeleton, so no Possible-D-Sep pass. ---
        skel = Skeleton(n_vars=n, adj=adj, sepsets=sepsets, pvalues_max=None)
        decisions = SepsetOrienter()(skel, ctx.ci, alpha=alpha, background=background, recorder=rec)
        partial = decisions.apply_to_pag(skel, var_names=ctx.names)
        ctx.result = FCIRules()(partial, background=background, recorder=rec)

    return ctx.result
