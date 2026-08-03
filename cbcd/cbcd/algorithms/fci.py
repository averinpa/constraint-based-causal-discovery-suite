"""FCI / RFCI / anytime-FCI composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd._run import iid_run
from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.collider import ColliderOrienter, SepsetOrienter
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.recording import RunRecorder
from cbcd.refinement import PAGSkeletonRefinement, PossibleDSepRefinement
from cbcd.rules import FCIRules, PAGRules
from cbcd.skeleton import FAS, Skeleton, SkeletonAlgorithm


def fci(
    data: NDArray[np.float64] | pd.DataFrame,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    skeleton: SkeletonAlgorithm | None = None,
    collider: ColliderOrienter | None = None,
    refinement: PAGSkeletonRefinement | None | Literal["default"] = "default",
    rules: PAGRules | None = None,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
    _algo_label: str = "fci",
) -> PAG:
    """Fast Causal Inference (Spirtes/Zhang). Returns a PAG.

    Two-pass pipeline: skeleton → collider → ``apply_to_pag`` → refinement
    → re-run collider on refined skeleton → ``apply_to_pag`` → FCIRules.

    The two-pass shape (D13) follows Zhang/Spirtes pseudocode and the
    ``causal-learn`` reference: Possible-D-Sep refinement may remove edges
    that change which triples are unshielded, so colliders must be
    re-classified before the rule-fixpoint.

    ``refinement`` defaults to ``PossibleDSepRefinement()``; pass ``None`` to
    skip refinement (the RFCI shape).
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm=_algo_label,
        params={"alpha": alpha, "max_cond_set": max_cond_set, "n_jobs": n_jobs},
        alpha=alpha,
        var_names=var_names,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        skel_algo = skeleton if skeleton is not None else FAS()
        coll_algo = collider if collider is not None else SepsetOrienter()
        rules_algo = rules if rules is not None else FCIRules()
        if refinement == "default":
            refine_algo: PAGSkeletonRefinement | None = PossibleDSepRefinement()
        else:
            refine_algo = refinement

        skel = skel_algo(
            ctx.ci,
            alpha=alpha,
            max_cond_set=max_cond_set,
            background=background,
            recorder=ctx.rec,
            n_jobs=n_jobs,
        )
        decisions = coll_algo(skel, ctx.ci, alpha=alpha, background=background, recorder=ctx.rec)
        partial = decisions.apply_to_pag(skel, var_names=ctx.names)

        if refine_algo is not None:
            partial = refine_algo(
                partial,
                ctx.ci,
                alpha=alpha,
                max_cond_set=max_cond_set,
                recorder=ctx.rec,
                n_jobs=n_jobs,
            )
            # Re-run collider on the refined skeleton: removed edges may have
            # changed which triples are unshielded, so the prior collider
            # classification is stale. Build a Skeleton from the refined adjacency,
            # carrying through the sepset witnesses recorded by refinement.
            refined_adj = (partial.endpoints != EndpointMark.NO_EDGE).astype(bool)
            refined_skel = Skeleton(
                n_vars=partial.n_vars,
                adj=refined_adj,
                sepsets=partial.sepsets if partial.sepsets is not None else {},
                pvalues_max=None,
            )
            decisions = coll_algo(
                refined_skel, ctx.ci, alpha=alpha, background=background, recorder=ctx.rec
            )
            partial = decisions.apply_to_pag(refined_skel, var_names=ctx.names)

        ctx.result = rules_algo(partial, background=background, recorder=ctx.rec)
    return ctx.result


def rfci(
    data: NDArray[np.float64] | pd.DataFrame,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
) -> PAG:
    """RFCI (Colombo et al. 2012): FCI without Possible-D-Sep refinement, with
    only Zhang's R1–R4. Faster and more conservative than ``fci()``."""
    return fci(
        data,
        ci_test=ci_test,
        alpha=alpha,
        refinement=None,
        rules=FCIRules(rules=frozenset({"R1", "R2", "R3", "R4"})),
        max_cond_set=max_cond_set,
        background=background,
        var_names=var_names,
        recorder=recorder,
        run_id=run_id,
        n_jobs=n_jobs,
        _algo_label="rfci",
    )


def anytime_fci(
    data: NDArray[np.float64] | pd.DataFrame,
    max_cond_set: int,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
) -> PAG:
    """Anytime-FCI (Spirtes 2001): ``fci()`` with a hard depth cap. Sound but
    possibly incomplete. ``max_cond_set`` is positional and required to make
    the trade-off explicit at the call site."""
    return fci(
        data,
        ci_test=ci_test,
        alpha=alpha,
        max_cond_set=max_cond_set,
        background=background,
        var_names=var_names,
        recorder=recorder,
        run_id=run_id,
        n_jobs=n_jobs,
        _algo_label="anytime_fci",
    )
