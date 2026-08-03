"""Shared algorithm scaffold: one consistent entry-point protocol + run context managers.

Every top-level algorithm follows the same shape — normalize data, build a cached CI test, resolve
the recorder, bracket the run with ``begin_run``/``finish_run``, and return a graph. That boilerplate
lives here once, so an algorithm body collapses to just its phase composition (which is already
protocol-based: SkeletonAlgorithm / ColliderOrienter / CPDAGRules / PAGRules / ...).

Two data models -> two context managers:
  * ``iid_run``    for i.i.d. array / DataFrame input (``CITest`` + ``CachedCITest``).
  * ``lagged_run`` for ``LaggedDataset`` input (``LaggedCITest`` + ``CachedLaggedCITest``).

Usage::

    with iid_run(data, ci_test=ci_test, algorithm="pc", params={...}, alpha=alpha,
                 var_names=var_names, recorder=recorder, run_id=run_id) as ctx:
        skel = PCStable()(ctx.ci, alpha=alpha, recorder=ctx.rec)
        ...
        ctx.result = rules(partial, recorder=ctx.rec)   # scaffold summarizes + finish_run on exit
    return ctx.result
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd._data import _graph_summary, _normalize_data
from cbcd.citest.cached import CachedCITest
from cbcd.citest.factory import make_ci_test
from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError
from cbcd.recording import RunRecorder, _resolve_recorder


@runtime_checkable
class Algorithm(Protocol):
    """A cbcd discovery algorithm. Conformance is structural: any callable with this shape counts.

    ``data`` is the input (array/DataFrame for i.i.d., ``LaggedDataset`` for time series); the
    remaining keywords are the shared conventions (``ci_test``, ``alpha``, ``background``,
    ``recorder``, ``run_id``). Algorithm-specific options ride in ``**kwargs``. Returns a graph.
    """

    def __call__(self, data: Any, **kwargs: Any) -> Any: ...


def _new_run_id(run_id: str | None) -> str:
    return run_id if run_id is not None else uuid.uuid4().hex


def _check_alpha(alpha: float | None) -> None:
    if alpha is not None and not (0.0 < alpha < 1.0):
        raise CBCDInputError(f"alpha must be in (0, 1), got {alpha}")


@dataclass
class RunContext:
    """State shared with an i.i.d. algorithm body. Set ``result`` (and optionally ``summary``)
    before the ``with`` block exits; the scaffold records ``finish_run`` from them."""

    ci: CachedCITest
    rec: RunRecorder
    n_vars: int
    n_samples: int
    names: tuple[str, ...] | None
    array: NDArray[np.float64]
    targets: list[int] | None = None
    result: Any = None
    summary: dict[str, Any] | None = None


@contextmanager
def iid_run(
    data: NDArray[np.float64] | pd.DataFrame,
    *,
    ci_test: CITest | str,
    algorithm: str,
    params: dict[str, Any],
    alpha: float | None = None,
    var_names: Sequence[str] | None = None,
    targets: Sequence[int] | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> Iterator[RunContext]:
    """Scaffold for an i.i.d. algorithm: normalize data, build the cached CI test, bracket the run.

    ``targets`` (when given) is validated against the variable count, deduped and sorted, and exposed
    as ``ctx.targets``. On normal exit the result graph is summarized (``ctx.summary`` if set, else
    variable/edge counts) and ``finish_run`` is recorded. On exception nothing is finalized.
    """
    _check_alpha(alpha)
    array, names = _normalize_data(data, var_names)
    n = int(array.shape[1])

    if isinstance(ci_test, str):
        inner: CITest = make_ci_test(ci_test, array)
        ci_name = ci_test
    else:
        inner = ci_test
        ci_name = type(inner).__name__
        if inner.n_vars != n:
            raise CBCDInputError(
                f"ci_test.n_vars ({inner.n_vars}) does not match data columns ({n})"
            )

    q: list[int] | None = None
    if targets is not None:
        q = sorted({int(t) for t in targets})
        if not q:
            raise CBCDInputError("targets must be non-empty")
        if q[0] < 0 or q[-1] >= n:
            raise CBCDInputError(f"targets must be indices in [0, {n}); got {sorted(targets)}")

    rec = _resolve_recorder(recorder)
    ctx = RunContext(
        ci=CachedCITest(inner),
        rec=rec,
        n_vars=n,
        n_samples=int(array.shape[0]),
        names=names,
        array=array,
        targets=q,
    )
    rec.begin_run(
        run_id=_new_run_id(run_id),
        algorithm=algorithm,
        params=params,
        n_samples=ctx.n_samples,
        n_vars=n,
        ci_test=ci_name,
        targets=q,
    )
    yield ctx
    summary = ctx.summary
    if summary is None and ctx.result is not None:
        summary = _graph_summary(ctx.result.endpoints, ctx.result.n_vars)
    rec.finish_run(result_summary=summary if summary is not None else {})


@dataclass
class LaggedRunContext:
    """State shared with a time-series algorithm body (nodes are the lagged grid)."""

    ci: Any  # CachedLaggedCITest
    rec: RunRecorder
    n_series: int
    max_lag: int
    grid_n: int
    n_samples: int
    result: Any = None
    summary: dict[str, Any] = field(default_factory=dict)


@contextmanager
def lagged_run(
    data: Any,  # LaggedDataset
    *,
    ci_test: Any,  # LaggedCITest | str
    algorithm: str,
    params: dict[str, Any],
    alpha: float | None = None,
    recorder: RunRecorder | None = None,
    run_id: str | None = None,
) -> Iterator[LaggedRunContext]:
    """Scaffold for a time-series algorithm over a ``LaggedDataset``. ``n_vars`` recorded is the
    lagged-grid size ``n_series*(max_lag+1)``. The body sets ``ctx.summary`` (time-series graphs vary)."""
    from cbcd.timeseries.citest import CachedLaggedCITest, make_lagged_ci_test

    _check_alpha(alpha)
    if isinstance(ci_test, str):
        inner = make_lagged_ci_test(ci_test, data)
        ci_name = ci_test
    else:
        inner = ci_test
        ci_name = type(inner).__name__
        if inner.n_vars != data.n_vars or inner.max_lag != data.max_lag:
            raise CBCDInputError("ci_test (n_vars, max_lag) does not match the dataset")

    n, lag = data.n_vars, data.max_lag
    grid_n = n * (lag + 1)
    rec = _resolve_recorder(recorder)
    ctx = LaggedRunContext(
        ci=CachedLaggedCITest(inner),
        rec=rec,
        n_series=n,
        max_lag=lag,
        grid_n=grid_n,
        n_samples=data.n_samples,
    )
    rec.begin_run(
        run_id=_new_run_id(run_id),
        algorithm=algorithm,
        params=params,
        n_samples=data.n_samples,
        n_vars=grid_n,
        ci_test=ci_name,
    )
    yield ctx
    rec.finish_run(result_summary=ctx.summary)
