"""MMB-by-MMB — local causal structure learning in the presence of latent variables.

Xie, Li, Wu, Zeng, Liu and Geng (2024). The predecessor of :mod:`cbcd.algorithms.locals_` from
the same group, and the latent-case analogue of Wang et al.'s MB-by-MB. Same walk, one Markov
blanket at a time; what differs is which part of each local structure is kept, how the waitlist
grows and when the walk stops.

Algorithm 1 of the paper::

    Waitlist := {T}; Donelist := {}; P := {}
    repeat
        V_X <- head of Waitlist
        MMB+(V_X) <- MMB_alg(V_X)
        if MMB+(V_X) is inside a region already learned        -> reuse that substructure
        else if MMB(V_X) is entirely in Donelist               -> reuse P
        else                                                   -> learn L_X over MMB+(V_X)
        P <- edges connected to V_X and the V-structures containing V_X
        P <- orient maximally with Zhang's rules
        Waitlist <- Waitlist + (Adj(V_X) \\ (Waitlist + Donelist))
    until a stopping rule fires

WHAT IS PRESERVED. Theorem 1 says an edge at ``X`` can be decided inside ``MMB+(X)``, and
Theorem 2 that the V-structures involving ``X`` found there are true ones, in two forms: S1 with
``X`` as the collider, S2 with ``X`` as an arm. Remark 2 is explicit that nothing else carries
over, and gives both failure modes — a V-structure not involving ``X`` can be wrong, and a true
V-structure at ``X`` can be unfindable when the separating set for its arms lies outside the
blanket.

PAPER DEFINITION INCOMPLETE. Section 3.1 defines MMB(T) as the parents of T, the children of T,
and the vertices not adjacent to T with a collider path to T. That omits the vertices joined to
T by a bidirected edge, which are adjacent and so excluded by the third clause while being
neither parent nor child. Nothing can m-separate an adjacent vertex from T, so the set as written
is not a Markov blanket. Measured over 163 targets on random MAGs, it differs from the correct
blanket on 21 of them, always on dense graphs; adding the bidirected neighbours makes it agree
everywhere. The same group's later Definition 9 (LoCaLS) states the four-clause version that is
correct, so this module uses ``total_conditioning``, which returns it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd._run import iid_run
from cbcd.algorithms._sequential import (
    ARROW,
    CIRCLE,
    NO_EDGE,
    collider_at,
    harvest,
    local_learn,
    settled_subgraph,
    unshielded,
)
from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError
from cbcd.graph.pag import PAG, PartialPAG
from cbcd.mb import total_conditioning
from cbcd.recording import RunRecorder
from cbcd.rules import FCIRules

MBLearner = Callable[..., "frozenset[int]"]


def _preserve(
    keep: NDArray[np.int8], local: NDArray[np.int8], x: int, blanket: frozenset[int]
) -> NDArray[np.int8]:
    """Keep the edges at ``x`` (Theorem 1) and the V-structures involving ``x`` (Theorem 2)."""
    out = keep.copy()
    n = out.shape[0]

    # Theorem 1 — an edge at x is decidable inside MMB+(x), present or absent.
    for v in sorted(blanket):
        if local[x, v] == NO_EDGE:
            out[x, v] = out[v, x] = NO_EDGE
        elif out[x, v] == NO_EDGE:
            out[x, v] = out[v, x] = CIRCLE
    for v in range(n):
        if v != x and v not in blanket:
            out[x, v] = out[v, x] = NO_EDGE

    def commit(i: int, j: int, mark: int) -> None:
        if local[i, j] == NO_EDGE or mark == CIRCLE:
            return
        if out[i, j] == NO_EDGE:
            out[i, j] = out[j, i] = CIRCLE
        out[i, j] = mark

    nbrs = [v for v in range(n) if v != x and local[x, v] != NO_EDGE]

    # Theorem 2, S1 — x is the collider: V_a *-> x <-* V_b, arms non-adjacent.
    for a, b in itertools.combinations(sorted(nbrs), 2):
        if unshielded(local, a, b, x) and collider_at(local, a, x, b):
            commit(a, x, ARROW)
            commit(b, x, ARROW)

    # Theorem 2, S2 — x is an arm: x *-> V_a <-* V_b, with x and V_b non-adjacent.
    for a in nbrs:
        for b in range(n):
            if b in (x, a) or local[a, b] == NO_EDGE:
                continue
            if unshielded(local, x, b, a) and collider_at(local, x, a, b):
                commit(x, a, ARROW)
                commit(b, a, ARROW)
    return out


def _r3_satisfied(ep: NDArray[np.int8], t: int) -> bool:
    """Stopping rule R3 — every path out of ``t`` that still carries an undetermined mark at
    ``t`` is blocked further along by an arrowhead. Nothing beyond such a block can propagate an
    orientation back, so no further blanket could settle the edges at ``t``."""
    n = ep.shape[0]
    open_at_t = [w for w in range(n) if w != t and ep[t, w] != NO_EDGE and ep[w, t] == CIRCLE]
    if not open_at_t:
        return True
    for start in open_at_t:
        seen, stack = {t, start}, [start]
        while stack:
            cur = stack.pop()
            for nxt in range(n):
                if nxt in seen or ep[cur, nxt] == NO_EDGE:
                    continue
                if ep[nxt, cur] == ARROW:  # arrowhead into nxt blocks the path
                    continue
                seen.add(nxt)
                stack.append(nxt)
        if len(seen) > 2:  # the path out of t reaches something unblocked
            return False
    return True


def mmb_by_mmb(
    data: NDArray[np.float64] | pd.DataFrame,
    target: int,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    mb_learner: MBLearner = total_conditioning,
    max_cond_set: int | None = None,
    max_rounds: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> PAG:
    """MMB-by-MMB (Xie et al., 2024). Local PAG around ``target``, latents allowed.

    Returns an ``n_vars``-wide PAG carrying committed marks only where the walk certified them.
    The guarantee is agreement with global discovery on the direct causes and effects of
    ``target``, not on the whole graph.

    ``max_rounds`` caps the number of blankets learned. Not in the paper, where the walk runs to
    a stopping rule; it bounds an instrumented run on a large graph. A capped run stays sound,
    since each mark was certified when written, and is less informative.
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="mmb_by_mmb",
        params={
            "alpha": alpha,
            "max_cond_set": max_cond_set,
            "max_rounds": max_rounds,
            "mb_learner": getattr(mb_learner, "__name__", str(mb_learner)),
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

        keep = np.full((ctx.n_vars, ctx.n_vars), NO_EDGE, dtype=np.int8)
        sepsets: dict[frozenset[int], tuple[int, ...]] = {}
        learned: dict[int, tuple[frozenset[int], NDArray[np.int8]]] = {}
        waitlist: list[int] = [t]
        done: set[int] = set()
        rounds = 0

        while waitlist:
            if max_rounds is not None and rounds >= max_rounds:
                break
            x = waitlist.pop(0)
            blanket = frozenset(mb_learner(ctx.ci, x, alpha=alpha, recorder=ctx.rec))
            region = sorted(blanket | {x})
            ctx.rec.record_region(target=x, region=region, boundary=sorted(blanket))

            # The two shortcut branches of Algorithm 1. Both avoid a whole local learn, which is
            # where nearly all the CI cost sits, so they change the measured test count a lot.
            reuse = next(
                (ep for y, (bl, ep) in learned.items() if y in done and set(region) <= bl | {y}),
                None,
            )
            if reuse is not None:
                lep = reuse
            elif blanket <= done:
                lep = keep.copy()
            else:
                lep = np.asarray(
                    local_learn(
                        ctx.ci,
                        region,
                        alpha=alpha,
                        max_cond_set=max_cond_set,
                        background=background,
                        n_vars=ctx.n_vars,
                        names=ctx.names,
                        rec=ctx.rec,
                    ).endpoints,
                    dtype=np.int8,
                )
            learned[x] = (blanket, lep)

            keep = _preserve(keep, lep, x, blanket)
            done.add(x)
            closed = np.asarray(
                FCIRules()(
                    PartialPAG(
                        n_vars=ctx.n_vars,
                        endpoints=settled_subgraph(keep, done, ctx.n_vars),
                        var_names=ctx.names,
                        sepsets=sepsets,
                    ),
                    background=background,
                    recorder=ctx.rec,
                ).endpoints,
                dtype=np.int8,
            )
            keep = harvest(keep, closed, done)
            rounds += 1

            # R1 — everything around the target is determined.
            edges_at_t = [w for w in range(ctx.n_vars) if w != t and keep[t, w] != NO_EDGE]
            if edges_at_t and all(
                keep[t, w] != CIRCLE and keep[w, t] != CIRCLE for w in edges_at_t
            ):
                break
            if _r3_satisfied(keep, t):  # R3
                break
            # Waitlist update: the neighbours of x not already queued or processed.
            for v in range(ctx.n_vars):
                if v != x and keep[x, v] != NO_EDGE and v not in done and v not in waitlist:
                    waitlist.append(v)
            # R2 is this list coming back empty.

        ctx.result = PAG(n_vars=ctx.n_vars, endpoints=keep, var_names=ctx.names, sepsets=sepsets)
    return ctx.result
