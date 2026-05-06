"""Continuous and discrete CI tests with citk-native implementations.

* :class:`FisherZ` and :class:`Spearman` use a self-contained partial-
  correlation + Fisher-Z transform — no causal-learn import.
* :class:`ChiSq` and :class:`GSq` wrap causal-learn's vectorised
  ``Chisq_or_Gsq`` implementation. They're available only when the
  optional ``[causallearn]`` extra is installed; otherwise they're
  ``None`` placeholders that raise on instantiation.

Auto-registration with causal-learn's PC dispatch goes through
:func:`citk.tests._register.maybe_register`, which silently no-ops when
causal-learn is missing.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import CITKResult, CITKTest, NO_SPECIFIED_PARAMETERS_MSG, inner_test_kwargs
from ._register import maybe_register

_R_CLIP = 1.0 - 1e-15


def _fisher_z_partial_correlation(
    corr: np.ndarray, n_samples: int, X: int, Y: int, condition_set: list[int]
) -> CITKResult:
    """Schur-complement partial correlation + Fisher-Z transform → p-value.

    Behaves identically to causal-learn's ``FisherZ`` implementation on
    well-conditioned input; falls back to ``p_value=1.0`` (treat as
    independent) on numerically singular submatrices.
    """
    df = n_samples - len(condition_set) - 3
    if df <= 0:
        return CITKResult(p_value=1.0, statistic=0.0, df=df, n_effective=n_samples)
    if not condition_set:
        r = float(corr[X, Y])
    else:
        idx = [X, Y, *condition_set]
        sub = corr[np.ix_(idx, idx)]
        try:
            inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            return CITKResult(p_value=1.0, statistic=0.0, df=df, n_effective=n_samples)
        denom = inv[0, 0] * inv[1, 1]
        if denom <= 0:
            return CITKResult(p_value=1.0, statistic=0.0, df=df, n_effective=n_samples)
        r = float(-inv[0, 1] / np.sqrt(denom))
    r = max(min(r, _R_CLIP), -_R_CLIP)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    stat = abs(z) * np.sqrt(df)
    p_value = 2.0 * float(norm.sf(stat))
    return CITKResult(
        p_value=p_value,
        statistic=float(stat),
        df=df,
        n_effective=n_samples,
        extra={"r": r},
    )


def _correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Sample correlation matrix with safety guards (zero variance, NaN)."""
    if np.any(np.isnan(data)):
        raise ValueError("FisherZ does not support NaN entries.")
    if np.any(np.isinf(data)):
        raise ValueError("FisherZ does not support infinite entries.")
    centered = data - data.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1, keepdims=True)
    if np.any(std == 0):
        raise ValueError("FisherZ: a column has zero variance.")
    normed = centered / std
    n = data.shape[0]
    return (normed.T @ normed) / (n - 1)


class FisherZ(CITKTest):
    """Fisher-Z partial correlation test (continuous, Gaussian).

    Native implementation; no causal-learn dependency. Compatible with
    cbcd via the structural :class:`cbcd.CITest` Protocol — pass an
    instance directly to ``cbcd.pc(data, ci_test=FisherZ(data))``.
    """

    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("fisherz_citk", NO_SPECIFIED_PARAMETERS_MSG)
        if data.shape[0] < 4:
            raise ValueError(
                f"FisherZ needs at least 4 samples for a usable test; got {data.shape[0]}"
            )
        self._corr = _correlation_matrix(data)

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> float:
        cs = self._normalize_condition_set(condition_set)
        return _fisher_z_partial_correlation(
            self._corr, self.sample_size, X, Y, cs
        ).p_value

    def details(
        self,
        X: int,
        Y: int,
        condition_set: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> CITKResult:
        cs = self._normalize_condition_set(condition_set)
        return _fisher_z_partial_correlation(self._corr, self.sample_size, X, Y, cs)


maybe_register("fisherz_citk", FisherZ)


class Spearman(CITKTest):
    """Spearman partial correlation: Fisher-Z applied to ranked data.

    Native implementation. Captures monotonic but non-linear dependence
    that vanilla Fisher-Z would miss.
    """

    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        ranked = pd.DataFrame(data).rank().to_numpy()
        super().__init__(ranked, **kwargs)
        self.check_cache_method_consistent("spearman", NO_SPECIFIED_PARAMETERS_MSG)
        if ranked.shape[0] < 4:
            raise ValueError(
                f"Spearman needs at least 4 samples; got {ranked.shape[0]}"
            )
        self._corr = _correlation_matrix(ranked)

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> float:
        cs = self._normalize_condition_set(condition_set)
        return _fisher_z_partial_correlation(
            self._corr, self.sample_size, X, Y, cs
        ).p_value

    def details(
        self,
        X: int,
        Y: int,
        condition_set: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> CITKResult:
        cs = self._normalize_condition_set(condition_set)
        return _fisher_z_partial_correlation(self._corr, self.sample_size, X, Y, cs)


maybe_register("spearman", Spearman)


# ---------------------------------------------------------------------------
# Discrete tests (require causal-learn for the underlying Chisq_or_Gsq math).
# ---------------------------------------------------------------------------

try:
    from causallearn.utils.cit import Chisq_or_Gsq

    class GSq(CITKTest):
        """G-squared CI test for discrete data. Wraps causal-learn's
        ``Chisq_or_Gsq``; available only with the ``[causallearn]`` extra."""

        supported_dtypes = {"discrete"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            super().__init__(data, **kwargs)
            self.check_cache_method_consistent("gsq", NO_SPECIFIED_PARAMETERS_MSG)
            self.test_instance = Chisq_or_Gsq(
                data, method_name="gsq", **inner_test_kwargs(kwargs)
            )

        def _compute(
            self,
            X: int,
            Y: int,
            condition_set: Optional[list[int]] = None,
            **kwargs: Any,
        ) -> float:
            return float(self.test_instance(X, Y, condition_set))

    class ChiSq(CITKTest):
        """Chi-squared CI test for discrete data. Wraps causal-learn's
        ``Chisq_or_Gsq``; available only with the ``[causallearn]`` extra."""

        supported_dtypes = {"discrete"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            super().__init__(data, **kwargs)
            self.check_cache_method_consistent("chisq", NO_SPECIFIED_PARAMETERS_MSG)
            self.test_instance = Chisq_or_Gsq(
                data, method_name="chisq", **inner_test_kwargs(kwargs)
            )

        def _compute(
            self,
            X: int,
            Y: int,
            condition_set: Optional[list[int]] = None,
            **kwargs: Any,
        ) -> float:
            return float(self.test_instance(X, Y, condition_set))

    maybe_register("gsq", GSq)
    maybe_register("chisq", ChiSq)
except ImportError:
    GSq = None  # type: ignore[assignment]
    ChiSq = None  # type: ignore[assignment]
