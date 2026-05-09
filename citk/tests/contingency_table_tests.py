"""Contingency-table CI tests (survey family): ChiSq, GSq.

Both tests wrap causal-learn's vectorised ``Chisq_or_Gsq``
implementation, so they require the optional ``[causallearn]`` extra.
When causal-learn isn't installed the classes are bound to ``None``
and module import does not fail.
"""

from typing import Any, Optional

import numpy as np

from ._register import maybe_register
from .base import CITKTest, NO_SPECIFIED_PARAMETERS_MSG, inner_test_kwargs

__all__ = ["ChiSq", "GSq"]


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
