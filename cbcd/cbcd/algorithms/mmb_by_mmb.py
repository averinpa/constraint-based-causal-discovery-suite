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

PAPER vs CODE. Stopping rule R3 (Theorem 3) is false as stated, so it is off by default
(``stop_r3=False``). R3 stops once every path out of T that leaves by an undetermined mark is
blocked by an arrowhead ``*->`` pointing along it. Its proof (Appendix C.3) only follows marks
propagating back toward T. Rules R9 and R10 work along uncovered potentially directed paths,
and those run straight through such an arrowhead. Witness, at the exact covariance oracle on
MILDEW with dm_3, foto_3, temp_4 and udbytte hidden, target dm_4: R3 fires with dm_4 <-o temp_3
and temp_3 o-> mikro_3, mikro_3 not yet processed. Once mikro_3 and its neighbours are
processed, R9 orients temp_3 -> dm_4 along temp_3 o-> mikro_3 o-> meldug_4 o-> lai_4 -> foto_4
-> dm_4. The global PAG has that tail. The authors' code (``stop_condition_three``) misses the
same mark. With R3 off, the walk stops on R1 or R2 only. Output then equals the true PAG at the
target on all seven anchor targets (MILDEW 20 of 20 marks, ALARM 32 of 32). With R3 on it gets
19 of 20 and 30 of 32.

PAPER vs CODE (our port, fixed). The first port had two bugs, and the authors' code has
neither. (1) R3 treated an unprocessed neighbour with no recorded edges as a dead end. After
round one only T is processed, so every path looked blocked and the walk stopped after one
blanket. The authors' ``stop_condition_three`` returns False there. (2) No separating set
reached the rules. ``FCIRules`` R4 reads a missing sepset as "not in the sepset" and orients
a <-> b <-> c, which put a wrong arrowhead on ALARM (CATECHOL <-> ARTCO2, truth ARTCO2 ->
CATECHOL). The authors record a sepset for every decided pair: the one found in the adjacency
search, or MMB(x) for a vertex outside the blanket of x. This module now does the same. Before
the fix: 8 of 20 marks on MILDEW and 21 of 32 on ALARM, all misses left as circles.
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


def _r3_satisfied(ep: NDArray[np.int8], t: int, done: set[int]) -> bool:
    """Stopping rule R3, in the form its proof uses (Appendix C.3). Off by default, because it
    is incomplete (see the module docstring).

    A path leaves ``t`` by an edge with an undetermined mark, runs through processed vertices
    ``V_1 .. V_n`` and ends at an unprocessed ``Y``. It is blocked where it meets an arrowhead
    pointing along it (``V_i *-> V_{i+1}``, or ``V_n *-> Y``). R3 holds when no unprocessed
    vertex is reachable from ``t`` without crossing such an arrowhead. Edges at ``t`` with both
    marks decided are not followed, as in the authors' ``stop_condition_three``.
    """
    n = ep.shape[0]
    decided = {
        w for w in range(n)
        if w != t and ep[t, w] != NO_EDGE and ep[t, w] != CIRCLE and ep[w, t] != CIRCLE
    }
    seen = {t} | decided
    stack = []
    for w in range(n):
        if w in seen or ep[t, w] == NO_EDGE or ep[t, w] == ARROW:  # ARROW: blocked at w
            continue
        seen.add(w)
        stack.append(w)
    while stack:
        cur = stack.pop()
        if cur not in done:
            return False  # reached an unprocessed vertex without crossing an arrowhead
        for nxt in range(n):
            if nxt in seen or ep[cur, nxt] == NO_EDGE or ep[cur, nxt] == ARROW:
                continue
            seen.add(nxt)
            stack.append(nxt)
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
    stop_r3: bool = False,
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

    ``stop_r3`` turns on the paper's stopping rule R3. It is off by default because R3 can stop
    before R9 or R10 have the path they need (module docstring). Turn it on only to reproduce
    the paper's stopping behaviour. R1 and R2 are always active.
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="mmb_by_mmb",
        params={
            "alpha": alpha,
            "max_cond_set": max_cond_set,
            "max_rounds": max_rounds,
            "stop_r3": stop_r3,
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
                local = local_learn(
                    ctx.ci,
                    region,
                    alpha=alpha,
                    max_cond_set=max_cond_set,
                    background=background,
                    n_vars=ctx.n_vars,
                    names=ctx.names,
                    rec=ctx.rec,
                )
                lep = np.asarray(local.endpoints, dtype=np.int8)
                # A set that separates two vertices inside the region separates them in the full
                # graph too, so every local sepset is kept. The first one recorded wins.
                for pair, sep in (local.sepsets or {}).items():
                    sepsets.setdefault(pair, tuple(sep))
            learned[x] = (blanket, lep)
            # A vertex outside MMB(x) is m-separated from x by MMB(x). The authors' find_mmb
            # records exactly this. R4 needs it: a missing sepset makes it orient a collider.
            for v in range(ctx.n_vars):
                if v != x and v not in blanket:
                    sepsets.setdefault(frozenset({x, v}), tuple(sorted(blanket)))

            before = keep
            keep = _preserve(keep, lep, x, blanket)
            ctx.rec.record_choice(
                kind="preserve", subject=str(x), choice="theorems_1_2",
                phase="mmb_by_mmb",
                detail={
                    "offered": int((lep != NO_EDGE).sum()),
                    "kept": int((keep != NO_EDGE).sum() - (before != NO_EDGE).sum()),
                    "reused": reuse is not None,
                },
            )
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
                ctx.rec.record_choice(kind="stop", subject=str(t), choice="R1",
                                      phase="mmb_by_mmb", detail={"round": rounds})
                break
            if stop_r3 and _r3_satisfied(keep, t, done):  # R3, opt-in
                ctx.rec.record_choice(kind="stop", subject=str(t), choice="R3",
                                      phase="mmb_by_mmb", detail={"round": rounds})
                break
            # Waitlist update: the neighbours of x not already queued or processed.
            for v in range(ctx.n_vars):
                if v != x and keep[x, v] != NO_EDGE and v not in done and v not in waitlist:
                    waitlist.append(v)
            if not waitlist:  # R2
                ctx.rec.record_choice(kind="stop", subject=str(t), choice="R2",
                                      phase="mmb_by_mmb", detail={"round": rounds})

        ctx.result = PAG(n_vars=ctx.n_vars, endpoints=keep, var_names=ctx.names, sepsets=sepsets)
    return ctx.result
