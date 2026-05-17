"""Kernel CI tests (survey family): KCI, RCIT, RCoT.

* :class:`KCI` wraps causal-learn's KCI implementation; available only
  with the optional ``[causallearn]`` extra.
* :class:`RCIT` and :class:`RCoT` are randomized kernel CI tests from
  the R RCIT package (Strobl et al., 2019); require the ``[r]`` extra
  and the R RCIT package installed in your R environment.
"""

from typing import Any, List, Optional

import numpy as np

from ._register import maybe_register
from ._backends import (
    _extract_rcit_p_value,
    _load_rcit_package,
    _to_r_matrix,
    _to_r_vector,
)
from .base import CITKTest, NO_SPECIFIED_PARAMETERS_MSG, inner_test_kwargs

__all__ = ["KCI", "RCIT", "RCoT"]


class _RCITBase(CITKTest):
    supported_dtypes = {"continuous"}
    accepted_kwargs: set = set()
    method_name = ""
    rcit_func_name = ""

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent(self.method_name, NO_SPECIFIED_PARAMETERS_MSG)

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> float:
        ro, rcit_pkg = _load_rcit_package()

        x = _to_r_vector(ro, self.data[:, X])
        y = _to_r_vector(ro, self.data[:, Y])
        if condition_set:
            z = _to_r_matrix(ro, self.data[:, condition_set])
            result = getattr(rcit_pkg, self.rcit_func_name)(x, y, z)
        else:
            result = getattr(rcit_pkg, self.rcit_func_name)(x, y)

        return _extract_rcit_p_value(result)


class RCoT(_RCITBase):
    method_name = "rcot"
    rcit_func_name = "RCoT"


class RCIT(_RCITBase):
    method_name = "rcit"
    rcit_func_name = "RCIT"


maybe_register("rcot", RCoT)
maybe_register("rcit", RCIT)


try:
    from causallearn.utils.cit import KCI as KCI_test

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
except ImportError:
    KCI = None  # type: ignore[assignment]
