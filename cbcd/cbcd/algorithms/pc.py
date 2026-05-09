"""PC algorithm composition (vanilla)."""

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
from cbcd.graph.cpdag import CPDAG
from cbcd.recording import RunRecorder, _resolve_recorder
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
    n_jobs: int = 1,
) -> CPDAG:
    """PC algorithm: skeleton via PC-stable, colliders via sepset rule, Meek closure.

    Parameters mirror decisions D1 (data input), D2 (CI factory), D5 (background
    validation), D7 (n_jobs plumbed), D10 (max_cond_set), D11 (recorder),
    D12 (cache + recording fused via CachedCITest).
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
    _resolve_recorder(recorder)  # Validates type; NullRecorder default.

    skel_algo = skeleton if skeleton is not None else PCStable()
    coll_algo = collider if collider is not None else SepsetOrienter()
    rules_algo = rules if rules is not None else MeekRules()

    skel = skel_algo(
        cached,
        alpha=alpha,
        max_cond_set=max_cond_set,
        background=background,
        n_jobs=n_jobs,
    )
    decisions = coll_algo(skel, cached, alpha=alpha, background=background)
    partial = decisions.apply_to_cpdag(skel, var_names=names)
    return rules_algo(partial, background=background)
