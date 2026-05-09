"""Time-series algorithm composition: pcmci()."""

from __future__ import annotations

from typing import Literal

import numpy as np

from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark
from cbcd.recording import RunRecorder, _resolve_recorder
from cbcd.timeseries.citest import (
    CachedLaggedCITest,
    LaggedCITest,
    make_lagged_ci_test,
)
from cbcd.timeseries.graph import TimeSeriesCPDAG
from cbcd.timeseries.lagged import LaggedBackgroundKnowledge, LaggedDataset, LaggedVar
from cbcd.timeseries.skeleton import LaggedSkeletonAlgorithm, PC1Skeleton


def pcmci(
    data: LaggedDataset,
    *,
    ci_test: LaggedCITest | Literal["parcorr"] = "parcorr",
    alpha: float = 0.05,
    pc_alpha: float | None = None,
    skeleton: LaggedSkeletonAlgorithm | None = None,
    background: LaggedBackgroundKnowledge | None = None,
    recorder: RunRecorder | None = None,
    n_jobs: int = 1,
) -> TimeSeriesCPDAG:
    """PCMCI (Runge et al. 2019) — vanilla, lagged-only.

    Two-stage:

    1. **PC₁** — for each target ``Y_t``, prune candidate lagged parents
       via per-pair CI tests with growing conditioning sets.
    2. **MCI** — for each candidate ``(X_{t-τ}, Y_t)`` with τ ∈ [1, max_lag],
       condition on ``̂P(Y_t) ∪ {shifted parents of X_{t-τ}}`` and test;
       the edge exists iff the test rejects independence at ``alpha``.

    Returns a ``TimeSeriesCPDAG`` with all lagged edges directed
    past→present and no contemporaneous edges (decision: vanilla PCMCI
    assumes contemporaneous independence; use ``pcmci_plus`` for
    contemporaneous discovery — deferred).

    ``pc_alpha`` defaults to ``alpha`` when ``None``. Tigramite's grid
    auto-tune (open question O4) is deferred.
    """
    if not (0.0 < alpha < 1.0):
        raise CBCDInputError(f"alpha must be in (0, 1), got {alpha}")
    if pc_alpha is None:
        pc_alpha = alpha
    if not (0.0 < pc_alpha < 1.0):
        raise CBCDInputError(f"pc_alpha must be in (0, 1), got {pc_alpha}")
    if n_jobs != 1:
        raise CBCDInputError("n_jobs != 1 not yet implemented in this slice; pass n_jobs=1")

    if isinstance(ci_test, str):
        inner: LaggedCITest = make_lagged_ci_test(ci_test, data)
    else:
        inner = ci_test
        if inner.n_vars != data.n_vars:
            raise CBCDInputError(
                f"ci_test.n_vars ({inner.n_vars}) does not match data.n_vars ({data.n_vars})"
            )
        if inner.max_lag != data.max_lag:
            raise CBCDInputError(
                f"ci_test.max_lag ({inner.max_lag}) does not match data.max_lag ({data.max_lag})"
            )

    cached = CachedLaggedCITest(inner)
    _resolve_recorder(recorder)

    skel_algo = skeleton if skeleton is not None else PC1Skeleton()
    skel = skel_algo(
        cached,
        alpha=pc_alpha,
        background=background,
        n_jobs=n_jobs,
    )

    n = data.n_vars
    max_lag = data.max_lag
    endpoints = np.zeros((max_lag + 1, n, n), dtype=np.int8)

    # MCI step: test every candidate lagged edge (X_{t-τ}, Y_t) using the
    # conditioning set ̂P(Y) ∪ shifted ̂P(X). Add the edge iff the test
    # rejects independence at α.
    for y_var in range(n):
        target = LaggedVar(y_var, 0)
        py = skel.parents.get(target, frozenset())
        for x_var in range(n):
            x_parents = skel.parents.get(LaggedVar(x_var, 0), frozenset())
            for tau in range(1, max_lag + 1):
                candidate = LaggedVar(x_var, -tau)
                if background is not None and background.is_forbidden_lagged(candidate, target):
                    continue
                # Shifted parents of X at lag -τ: each (Z, -σ) ∈ ̂P(X)
                # becomes (Z, -(τ + σ)) in Y's frame.
                shifted = {
                    LaggedVar(p.var, p.lag - tau) for p in x_parents if -p.lag + tau <= max_lag
                }
                S_set = (py | shifted) - {candidate, target}
                p = cached(candidate, target, sorted(S_set, key=lambda lv: (lv.var, lv.lag)))
                if p <= alpha:
                    endpoints[tau, x_var, y_var] = EndpointMark.ARROW

    return TimeSeriesCPDAG(
        n_vars=n,
        max_lag=max_lag,
        endpoints=endpoints,
        var_names=data.var_names,
    )
