"""Lagged CI test layer: LaggedCITest Protocol, ParCorr, caching, factory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from cbcd.exceptions import CBCDDataError, CBCDInputError
from cbcd.timeseries.lagged import LaggedDataset, LaggedVar

_R_CLIP = 1.0 - 1e-15


@dataclass(frozen=True, slots=True)
class LaggedCITestResult:
    """Outcome of a single lagged conditional-independence call.

    ``n_effective`` is particularly important for time-series tests: it's
    typically ``T - max_lag`` (the number of usable training rows after
    aligning the lagged design), possibly smaller if the test does extra
    preprocessing.
    """

    p_value: float
    statistic: float | None = None
    df: int | None = None
    n_effective: int | None = None
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class LaggedCITest(Protocol):
    """Conditional-independence test bound to a ``LaggedDataset``.

    Conditioning sets are sequences of ``LaggedVar``. Implementations must
    be deterministic for fixed ``(x, y, S)`` and stationary in the sense
    that the result depends on relative lags, not absolute time.
    """

    n_vars: int
    max_lag: int

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float: ...

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> LaggedCITestResult: ...


def _cache_key(
    x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]
) -> tuple[LaggedVar, LaggedVar, frozenset[LaggedVar]]:
    a, b = (x, y) if (x.var, -x.lag) <= (y.var, -y.lag) else (y, x)
    return (a, b, frozenset(S))


class CachedLaggedCITest:
    """Per-instance cache wrapper for any ``LaggedCITest``.

    Mirrors ``CachedCITest`` for the i.i.d. layer. The cache key is
    ``(min, max, frozenset(S))`` on ``LaggedVar`` tuples — no fragile
    data-content hashing.
    """

    n_vars: int
    max_lag: int

    def __init__(self, inner: LaggedCITest, *, cache: bool = True) -> None:
        self._inner = inner
        self.n_vars = inner.n_vars
        self.max_lag = inner.max_lag
        self._cache_enabled = cache
        self._cache: dict[
            tuple[LaggedVar, LaggedVar, frozenset[LaggedVar]],
            LaggedCITestResult,
        ] = {}

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> LaggedCITestResult:
        if not self._cache_enabled:
            return self._inner.details(x, y, S)
        key = _cache_key(x, y, S)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._inner.details(x, y, S)
        self._cache[key] = result
        return result

    def is_cached(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> bool:
        return _cache_key(x, y, S) in self._cache


class ParCorr:
    """Linear partial-correlation test on the lagged design matrix.

    Construction stacks the dataset into a wide ``(T - max_lag, n_vars *
    (max_lag + 1))`` design where column ``τ * n_vars + v`` holds
    ``data[max_lag - τ : T - τ, v]`` (the realisation of variable ``v`` at
    lag ``-τ``). The correlation matrix of this design is computed once.

    Each ``ci(x, y, S)`` call maps the ``LaggedVar`` arguments to design-
    column indices, takes the Schur-complement of the relevant submatrix
    to obtain the partial correlation, and applies the Fisher-Z transform
    with ``df = n_effective - |S| - 3``.
    """

    n_vars: int
    max_lag: int

    def __init__(self, dataset: LaggedDataset) -> None:
        self.n_vars = dataset.n_vars
        self.max_lag = dataset.max_lag
        T = dataset.n_samples
        ml = dataset.max_lag
        n_eff = T - ml
        if n_eff < 4:
            raise CBCDDataError(
                f"ParCorr needs at least 4 effective samples (T - max_lag); "
                f"got T={T}, max_lag={ml}, T-max_lag={n_eff}"
            )
        if np.any(np.isnan(dataset.data)):
            raise CBCDDataError("ParCorr does not support NaN entries")
        if np.any(np.isinf(dataset.data)):
            raise CBCDDataError("ParCorr does not support infinite entries")

        n_design = self.n_vars * (ml + 1)
        design = np.empty((n_eff, n_design), dtype=np.float64)
        for tau in range(ml + 1):
            block = dataset.data[ml - tau : T - tau, :]
            design[:, tau * self.n_vars : (tau + 1) * self.n_vars] = block

        centered = design - design.mean(axis=0, keepdims=True)
        std = centered.std(axis=0, ddof=1, keepdims=True)
        if np.any(std == 0):
            raise CBCDDataError(
                "ParCorr: a lagged column has zero variance "
                "(constant variable in the training window)"
            )
        normed = centered / std
        self._corr: NDArray[np.float64] = (normed.T @ normed) / (n_eff - 1)
        self._n_effective = n_eff

    def _col(self, lv: LaggedVar) -> int:
        tau = -lv.lag
        if not (0 <= tau <= self.max_lag):
            raise CBCDInputError(f"LaggedVar {lv} has lag outside [-max_lag={self.max_lag}, 0]")
        if not (0 <= lv.var < self.n_vars):
            raise CBCDInputError(f"LaggedVar {lv} has var outside [0, n_vars={self.n_vars})")
        return tau * self.n_vars + lv.var

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> LaggedCITestResult:
        S_tuple = tuple(S)
        x_col = self._col(x)
        y_col = self._col(y)
        S_cols = [self._col(z) for z in S_tuple]
        if x_col == y_col or x_col in S_cols or y_col in S_cols:
            raise CBCDInputError(f"ParCorr requires distinct columns: x={x}, y={y}, S={S_tuple}")

        df = self._n_effective - len(S_tuple) - 3
        if df <= 0:
            return LaggedCITestResult(
                p_value=1.0,
                statistic=0.0,
                df=df,
                n_effective=self._n_effective,
            )

        if not S_cols:
            r = float(self._corr[x_col, y_col])
        else:
            idx = [x_col, y_col, *S_cols]
            sub = self._corr[np.ix_(idx, idx)]
            try:
                inv = np.linalg.inv(sub)
            except np.linalg.LinAlgError:
                return LaggedCITestResult(
                    p_value=1.0,
                    statistic=0.0,
                    df=df,
                    n_effective=self._n_effective,
                )
            denom = inv[0, 0] * inv[1, 1]
            if denom <= 0:
                return LaggedCITestResult(
                    p_value=1.0,
                    statistic=0.0,
                    df=df,
                    n_effective=self._n_effective,
                )
            r = float(-inv[0, 1] / np.sqrt(denom))

        r = max(min(r, _R_CLIP), -_R_CLIP)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        stat = abs(z) * np.sqrt(df)
        p_value = 2.0 * float(norm.sf(stat))
        return LaggedCITestResult(
            p_value=p_value,
            statistic=float(stat),
            df=df,
            n_effective=self._n_effective,
            extra={"r": r},
        )


# --- factory + registry ---------------------------------------------------


_LAGGED_REGISTRY: dict[str, Callable[..., LaggedCITest]] = {}


def register_lagged_ci_test(name: str, factory: Callable[..., LaggedCITest]) -> None:
    """Register a user-defined lagged CI test under ``name``.

    Raises ``CBCDInputError`` on duplicate registration.
    """
    if name in _LAGGED_REGISTRY:
        raise CBCDInputError(f"lagged CI test {name!r} is already registered")
    _LAGGED_REGISTRY[name] = factory


def make_lagged_ci_test(name: str, dataset: LaggedDataset, **kwargs: object) -> LaggedCITest:
    """Resolve a lagged CI-test name to a bound instance.

    Built-in (this slice ships only one):
        "parcorr" -> ParCorr
    """
    factory = _LAGGED_REGISTRY.get(name)
    if factory is None:
        raise CBCDInputError(
            f"unknown lagged CI test {name!r}; registered: {sorted(_LAGGED_REGISTRY)}"
        )
    return factory(dataset, **kwargs)


_LAGGED_REGISTRY["parcorr"] = ParCorr
