"""LoCaLS — local causal structure learning with latent variables and selection bias.

Li, Zhang, Wang, Cai, Zhang and Xie (2025). Learns the structure around a single ``target``
without building the global PAG, by walking outward one Markov blanket at a time.

The loop is Algorithm 1 of the paper::

    Waitlist <- {T}; Donelist <- {}; P_hat <- {}; Sepsets <- {}
    repeat
        X  <- first of Waitlist
        MB+(X)       <- MBLearn(X)                      # total conditioning, mb.py
        L_X, seps    <- LocalLearn(MB+(X))              # FCI over that region
        P_hat        <- PreserveConsistentInfo(P_hat, L_X, X)
        P_hat        <- OrientPAG(P_hat, Sepsets)       # Zhang R1-R10, rules.py
        Waitlist     <- UpdateWaitlist(T, P_hat, Donelist)
    until a stopping rule fires

Everything except the three LoCaLS-specific steps is composed from existing cbcd pieces, so the
CI queries land in the recorder exactly as they do for ``fci`` or ``pc`` with no extra wiring:
``PCStable`` for the region skeleton, ``SepsetOrienter`` for colliders, ``PossibleDSepRefinement``
for the second pass, and ``FCIRules`` for the rule closure.

WHAT IS PRESERVED, AND WHY ONLY THAT. A PAG learned from the marginal over ``MB+(X)`` is not in
general the global PAG restricted to those vertices. Section 4.2 of the paper says exactly which
parts survive, and this module copies only those three:

* adjacencies incident to ``X`` (Corollary 1),
* uncovered collider paths out of ``X`` (Theorem 2),
* unshielded collider triples with ``X`` as the collider (Theorem 3).

Everything else in ``L_X`` is dropped. Remark 3 gives both failure modes that justify the
restriction: a collider in ``L_X`` not involving ``X`` can be spurious, and a true collider at
``X`` can be unrecoverable when the separating set for its arms lies outside ``MB+(X)``.
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


# ------------------------------------------------------- PreserveConsistentInfo
def _uncovered_collider_paths(ep: NDArray[np.int8], x: int, blanket: frozenset[int]):
    """Every uncovered collider path ``X *-> V1 <-> ... <-* Vi`` out of ``x`` (Theorem 2).

    A collider path has every non-endpoint a collider on it; uncovered means every consecutive
    triple is unshielded. The first edge must carry an arrowhead at ``V1``, which is the ``*->``
    in the statement. Yielded as vertex lists starting at ``x``.
    """
    n = ep.shape[0]
    starts = [v for v in range(n) if v in blanket and ep[x, v] == ARROW]
    stack = [[x, v] for v in starts]
    while stack:
        path = stack.pop()
        yield path
        last, prev = path[-1], path[-2]
        for nxt in range(n):
            if nxt in path or nxt not in blanket or ep[last, nxt] == NO_EDGE:
                continue
            if collider_at(ep, prev, last, nxt) and unshielded(ep, prev, nxt, last):
                stack.append([*path, nxt])


def _preserve_consistent_info(
    keep: NDArray[np.int8], local: NDArray[np.int8], x: int, blanket: frozenset[int]
) -> NDArray[np.int8]:
    """Copy from ``local`` into ``keep`` only what Section 4.2 guarantees is globally consistent."""
    out = keep.copy()

    # Corollary 1 — adjacencies incident to X are exactly the global ones, present or absent.
    for v in sorted(blanket):
        if local[x, v] == NO_EDGE:
            out[x, v] = out[v, x] = NO_EDGE
        else:
            if out[x, v] == NO_EDGE:
                out[x, v] = out[v, x] = CIRCLE
    # X's adjacency is settled, so anything outside the blanket is not adjacent to X.
    for v in range(out.shape[0]):
        if v != x and v not in blanket:
            out[x, v] = out[v, x] = NO_EDGE

    def commit(i: int, j: int, mark: int) -> None:
        """Write a certified mark, creating the edge if the local graph has one.

        A circle never overwrites a committed mark. A later region can be silent about an
        endpoint an earlier one settled, and silence is not evidence. The authors' code does the
        same by passing ``None`` instead of a circle when it merges a region in.
        """
        if local[i, j] == NO_EDGE:
            return
        if out[i, j] == NO_EDGE:
            out[i, j] = out[j, i] = CIRCLE
        if mark != CIRCLE:
            out[i, j] = mark

    # Theorem 2 — uncovered collider paths out of X carry their marks across.
    for path in _uncovered_collider_paths(local, x, blanket):
        for a, b in zip(path, path[1:], strict=False):
            commit(a, b, int(local[a, b]))
            commit(b, a, int(local[b, a]))

    # Theorem 3 — an unshielded collider with X in the middle is a true one.
    nbrs = [v for v in range(local.shape[0]) if local[x, v] != NO_EDGE]
    for a, b in itertools.combinations(sorted(nbrs), 2):
        if unshielded(local, a, b, x) and collider_at(local, a, x, b):
            commit(a, x, ARROW)
            commit(b, x, ARROW)

    return out


# ------------------------------------------------------------- UpdateWaitlist
def _potentially_anterior(ep: NDArray[np.int8], v: int, target: int) -> bool:
    """Is there a potentially anterior path from ``v`` to ``target`` in ``ep``?

    Travelling toward the target, a path is blocked as soon as some vertex has an arrowhead
    pointing back into it — that is Theorem 4's "not potentially anterior toward T". A circle
    does not block, because it may yet resolve to a tail, which is what makes the rule a
    conservative stopping criterion rather than an optimisation.
    """
    if v == target:
        return True
    n = ep.shape[0]
    seen, stack = {v}, [v]
    while stack:
        cur = stack.pop()
        for nxt in range(n):
            if nxt in seen or ep[cur, nxt] == NO_EDGE:
                continue
            if ep[nxt, cur] == ARROW:  # arrowhead back into cur: blocked
                continue
            if nxt == target:
                return True
            seen.add(nxt)
            stack.append(nxt)
    return False


def _update_waitlist(ep: NDArray[np.int8], target: int, done: set[int]) -> list[int]:
    """Algorithm 4. Keep only unprocessed vertices that can still reach the target."""
    n = ep.shape[0]
    return [
        v
        for v in range(n)
        if v not in done
        and any(ep[v, w] != NO_EDGE for w in range(n))
        and _potentially_anterior(ep, v, target)
    ]


def _target_marks_decided(ep: NDArray[np.int8], target: int) -> bool:
    """Stopping rule R1 — every edge mark incident to the target is committed."""
    n = ep.shape[0]
    edges = [w for w in range(n) if w != target and ep[target, w] != NO_EDGE]
    if not edges:
        return False
    return all(ep[target, w] != CIRCLE and ep[w, target] != CIRCLE for w in edges)


# -------------------------------------------------------------------- LoCaLS
def locals_(
    data: NDArray[np.float64] | pd.DataFrame,
    target: int,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    mb_learner: MBLearner = total_conditioning,
    max_cond_set: int | None = None,
    max_rounds: int | None = None,
    as_published: bool = False,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> PAG:
    """LoCaLS (Li et al., 2025). Local PAG around ``target``, latents and selection allowed.

    Named with a trailing underscore because ``locals`` is a Python builtin.

    Returns an ``n_vars``-wide PAG in which only the part the walk certified carries committed
    marks; everything else is a circle or no edge. The guarantee the paper proves is agreement
    with global discovery on the edges incident to ``target``, not on the whole graph.

    ``mb_learner`` defaults to ``total_conditioning``, which is what the paper specifies and is
    the only blanket primitive in ``cbcd.mb`` that is valid in a MAG. The others characterise a
    DAG blanket and will silently return the wrong set under latents or selection.

    ``as_published`` reproduces the defect in Algorithm 1 as written, which runs the orientation
    rules over the whole accumulated graph. A missing edge there is usually a pair nothing has
    examined, not a proved absence, and every Zhang rule has a nonadjacency premise, so the rules
    fire almost everywhere. For measuring what the published specification does, not for use.

    ``max_rounds`` caps the number of blankets learned. It is not in the paper, where the walk
    runs until a stopping rule fires; it exists so an instrumented run on a large graph can be
    bounded. A capped run is still sound, since every mark was certified when it was written,
    but it is less informative than an uncapped one.
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="locals",
        params={
            "alpha": alpha,
            "max_cond_set": max_cond_set,
            "max_rounds": max_rounds,
            "as_published": as_published,
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
        absent: set[frozenset[int]] = set()
        waitlist: list[int] = [t]
        done: set[int] = set()
        rounds = 0

        while waitlist:
            if max_rounds is not None and rounds >= max_rounds:
                break
            x = waitlist[0]
            blanket = mb_learner(ctx.ci, x, alpha=alpha, recorder=ctx.rec)
            region = sorted(set(blanket) | {x})
            ctx.rec.record_region(target=x, region=region, boundary=sorted(blanket))

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
            if local.sepsets:
                sepsets.update(local.sepsets)
            # Every pair this region separated is a proved absence from here on. The authors
            # accumulate exactly this, as their global sepset store, and gate the rules on it.
            lep = np.asarray(local.endpoints, dtype=np.int8)
            for a, b in itertools.combinations(region, 2):
                if lep[a, b] == NO_EDGE:
                    absent.add(frozenset((a, b)))

            before = keep
            keep = _preserve_consistent_info(keep, lep, x, frozenset(blanket))
            # D5 — what the local structure offered against what Section 4.2 lets us keep.
            offered = int((lep != NO_EDGE).sum())
            taken = int(((keep != before) | ((keep != NO_EDGE) & (before != NO_EDGE))).sum())
            ctx.rec.record_choice(
                kind="preserve", subject=str(x), choice="section_4_2",
                phase="locals", detail={"offered": offered, "kept": taken},
            )
            done.add(x)
            closed = np.asarray(
                FCIRules()(
                    PartialPAG(
                        n_vars=ctx.n_vars,
                        endpoints=keep if as_published
                        else settled_subgraph(keep, done, ctx.n_vars),
                        var_names=ctx.names,
                        sepsets=sepsets,
                    ),
                    background=background,
                    recorder=ctx.rec,
                ).endpoints,
                dtype=np.int8,
            )
            keep = closed if as_published else harvest(keep, closed, done)
            rounds += 1
            if _target_marks_decided(keep, t):  # R1
                ctx.rec.record_choice(
                    kind="stop", subject=str(t), choice="R1",
                    phase="locals", detail={"round": rounds},
                )
                break
            waitlist = _update_waitlist(keep, t, done)
            if not waitlist:  # R2
                ctx.rec.record_choice(
                    kind="stop", subject=str(t), choice="R2",
                    phase="locals", detail={"round": rounds},
                )

        ctx.result = PAG(n_vars=ctx.n_vars, endpoints=keep, var_names=ctx.names, sepsets=sepsets)
    return ctx.result
