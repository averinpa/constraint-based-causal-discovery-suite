"""PC algorithm composition (vanilla)."""

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
from cbcd.graph.cpdag import CPDAG
from cbcd.recording import RunRecorder
from cbcd.rules import CPDAGRules, MeekRules
from cbcd.skeleton import PCStable, SkeletonAlgorithm


def pc(
    data: NDArray[np.float64] | pd.DataFrame,
    *,
    ci_test: CITest | Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    skeleton: SkeletonAlgorithm | None = None,
    collider: ColliderOrienter | None = None,
    rules: CPDAGRules | None = None,
    max_cond_set: int | None = None,
    background: BackgroundKnowledge | None = None,
    var_names: Sequence[str] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
) -> CPDAG:
    """PC algorithm: skeleton via PC-stable, colliders via sepset rule, Meek closure.

    Parameters mirror decisions D1 (data input), D2 (CI factory), D5 (background
    validation), D7 (n_jobs plumbed), D10 (max_cond_set), D11 (recorder),
    D12 (cache + recording fused via CachedCITest).
    """
    with iid_run(
        data,
        ci_test=ci_test,
        algorithm="pc",
        params={"alpha": alpha, "max_cond_set": max_cond_set, "n_jobs": n_jobs},
        alpha=alpha,
        var_names=var_names,
        recorder=recorder,
        run_id=run_id,
    ) as ctx:
        skel_algo = skeleton if skeleton is not None else PCStable()
        coll_algo = collider if collider is not None else SepsetOrienter()
        rules_algo = rules if rules is not None else MeekRules()

        skel = skel_algo(
            ctx.ci,
            alpha=alpha,
            max_cond_set=max_cond_set,
            background=background,
            recorder=ctx.rec,
            n_jobs=n_jobs,
        )
        decisions = coll_algo(skel, ctx.ci, alpha=alpha, background=background, recorder=ctx.rec)
        partial = decisions.apply_to_cpdag(skel, var_names=ctx.names)
        ctx.result = rules_algo(partial, background=background, recorder=ctx.rec)
    return ctx.result
