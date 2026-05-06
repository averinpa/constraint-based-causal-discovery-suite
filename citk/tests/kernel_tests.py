"""Kernel CI tests (survey family).

The :class:`KCI` wrapper requires causal-learn (uses its KCI implementation
internally). When the optional ``[causallearn]`` extra is not installed,
``KCI`` is bound to ``None`` and the import does not fail. ``RCIT`` and
``RCoT`` are R-based; their availability depends on the ``[r]`` extra.
"""

from typing import Any, List, Optional

import numpy as np

from ._register import maybe_register
from .base import CITKTest, NO_SPECIFIED_PARAMETERS_MSG, inner_test_kwargs
from .r_based_tests import RCIT, RCoT  # noqa: F401  (re-exported)

try:
    from causallearn.utils.cit import KCI as KCI_test
    _HAS_CAUSALLEARN = True
except ImportError:
    _HAS_CAUSALLEARN = False


if _HAS_CAUSALLEARN:

    class KCI(CITKTest):
        """Kernel Conditional Independence (KCI) test, wrapping causal-learn."""

        supported_dtypes = {"continuous"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            super().__init__(data, **kwargs)
            self.check_cache_method_consistent("kci", NO_SPECIFIED_PARAMETERS_MSG)
            self.kci_instance = KCI_test(data, **inner_test_kwargs(kwargs))

        def _compute(
            self,
            X: int,
            Y: int,
            condition_set: Optional[List[int]] = None,
            **kwargs: Any,
        ) -> float:
            return float(self.kci_instance(X, Y, condition_set))

    maybe_register("kci", KCI)
else:
    KCI = None  # type: ignore[assignment]


__all__ = ["KCI", "RCIT", "RCoT"]
