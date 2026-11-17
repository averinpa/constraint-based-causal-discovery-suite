"""Shared machinery for the sequential local algorithms.

LoCaLS and MMB-by-MMB are the same walk — pop a vertex, learn its Markov blanket, learn the
structure over that region, keep the part the marginal licenses, orient, update the waitlist,
test a stopping rule. They differ only in what gets kept, how the waitlist is updated and when
to stop. What is common lives here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.collider import SepsetOrienter
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PartialPAG
from cbcd.recording import RunRecorder
from cbcd.refinement import PossibleDSepRefinement
from cbcd.rules import FCIRules
from cbcd.skeleton import PCStable, Skeleton

NO_EDGE = int(EndpointMark.NO_EDGE)
ARROW = int(EndpointMark.ARROW)
TAIL = int(EndpointMark.TAIL)
CIRCLE = int(EndpointMark.CIRCLE)


def unshielded(ep: NDArray[np.int8], a: int, b: int, c: int) -> bool:
    """``a`` and ``b`` non-adjacent, with ``c`` the middle vertex of the triple."""
    return ep[a, b] == NO_EDGE and ep[b, a] == NO_EDGE


def collider_at(ep: NDArray[np.int8], a: int, m: int, b: int) -> bool:
    return ep[a, m] == ARROW and ep[b, m] == ARROW


def local_learn(
    ci: CITest,
    region: Sequence[int],
    *,
    alpha: float,
    max_cond_set: int | None,
    background: BackgroundKnowledge | None,
    n_vars: int,
    names: tuple[str, ...] | None,
    rec: RunRecorder,
) -> PartialPAG:
    """FCI over ``region`` only, which is LoCaLS's ``LocalLearn``.

    The same two-pass composition ``cbcd.fci`` uses — skeleton, colliders, Possible-D-Sep
    refinement, re-run colliders on the refined skeleton, rule closure — with ``PCStable``
    restricted to the region. Vertices outside it stay isolated, so the returned graph is
    ``n_vars`` wide and index positions are preserved throughout.
    """
    skel = PCStable()(
        ci,
        alpha=alpha,
        max_cond_set=max_cond_set,
        background=background,
        variables=list(region),
        recorder=rec,
    )
    coll = SepsetOrienter()
    partial = coll(skel, ci, alpha=alpha, background=background, recorder=rec).apply_to_pag(
        skel, var_names=names
    )
    partial = PossibleDSepRefinement()(
        partial, ci, alpha=alpha, max_cond_set=max_cond_set, recorder=rec
    )
    # Refinement can delete edges, which changes which triples are unshielded, so the collider
    # classification from before it is stale (the D13 two-pass shape, as in fci()).
    refined = Skeleton(
        n_vars=partial.n_vars,
        adj=(partial.endpoints != NO_EDGE).astype(bool),
        sepsets=partial.sepsets if partial.sepsets is not None else {},
        pvalues_max=None,
    )
    partial = coll(refined, ci, alpha=alpha, background=background, recorder=rec).apply_to_pag(
        refined, var_names=names
    )
    closed = FCIRules()(partial, background=background, recorder=rec)
    return PartialPAG(
        n_vars=n_vars,
        endpoints=np.asarray(closed.endpoints, dtype=np.int8),
        var_names=names,
        sepsets=closed.sepsets,
    )


# ------------------------------------------------------- PreserveConsistentInfo


def settled_subgraph(keep: NDArray[np.int8], done: set[int], n: int) -> NDArray[np.int8]:
    """Induced subgraph on the processed vertices, the only graph the rules are sound on.

    A missing edge anywhere else in ``keep`` is an unexamined pair, not a nonadjacency, and
    every Zhang rule needs a real one. Running them over all of ``keep`` gave 50 of 140 target
    marks wrong at an oracle, here 0. Corollary 1 settles adjacency at a processed vertex, so
    between two of them absence is proved.

    PAPER vs CODE. Algorithm 1 says only ``OrientPAG(P_hat, Sepsets)``. The authors' own
    implementation gates every rule on a recorded separating set, which the paper never says.
    """
    out = np.full((n, n), NO_EDGE, dtype=np.int8)
    for i in sorted(done):
        for j in sorted(done):
            if i != j:
                out[i, j] = keep[i, j]
    return out


def harvest(keep: NDArray[np.int8], closed: NDArray[np.int8], done: set[int]) -> NDArray[np.int8]:
    """Take rule-derived marks back, on settled edges only."""
    out = keep.copy()
    for i in sorted(done):
        for j in sorted(done):
            if i != j and out[i, j] != NO_EDGE:
                out[i, j] = closed[i, j]
    return out
