"""FCI / RFCI / anytime-FCI composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd.algorithms._data import _normalize_data
from cbcd.background import BackgroundKnowledge
from cbcd.citest.cached import CachedCITest
from cbcd.citest.factory import make_ci_test
from cbcd.citest.protocol import CITest
from cbcd.collider import ColliderOrienter, SepsetOrienter
from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.recording import RunRecorder, _resolve_recorder
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
    n_jobs: int = 1,
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
    if not (0.0 < alpha < 1.0):
        raise CBCDInputError(f"alpha must be in (0, 1), got {alpha}")

    array, names = _normalize_data(data, var_names)

    if isinstance(ci_test, str):
        inner: CITest = make_ci_test(ci_test, array)
    else:
        inner = ci_test
        if inner.n_vars != array.shape[1]:
            raise CBCDInputError(
                f"ci_test.n_vars ({inner.n_vars}) does not match data columns ({array.shape[1]})"
            )

    cached = CachedCITest(inner)
    _resolve_recorder(recorder)

    skel_algo = skeleton if skeleton is not None else FAS()
    coll_algo = collider if collider is not None else SepsetOrienter()
    rules_algo = rules if rules is not None else FCIRules()
    if refinement == "default":
        refine_algo: PAGSkeletonRefinement | None = PossibleDSepRefinement()
    else:
        refine_algo = refinement

    skel = skel_algo(
        cached,
        alpha=alpha,
        max_cond_set=max_cond_set,
        background=background,
        n_jobs=n_jobs,
    )
    decisions = coll_algo(skel, cached, alpha=alpha, background=background)
    partial = decisions.apply_to_pag(skel, var_names=names)

    if refine_algo is not None:
        partial = refine_algo(
            partial,
            cached,
            alpha=alpha,
            max_cond_set=max_cond_set,
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
        decisions = coll_algo(refined_skel, cached, alpha=alpha, background=background)
        partial = decisions.apply_to_pag(refined_skel, var_names=names)

    return rules_algo(partial, background=background)


def rfci(
    data: NDArray[np.float64] | pd.DataFrame,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
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
        n_jobs=n_jobs,
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
        n_jobs=n_jobs,
    )
