"""Nearest-neighbor CI tests (survey family): CMIknn, CMIknnMixed, MCMIknn.

* :class:`CMIknn` and :class:`CMIknnMixed` wrap tigramite's k-NN
  conditional mutual information estimators (require optional
  ``tigramite``).
* :class:`MCMIknn` wraps the vendored mCMIkNN implementation
  (Hügle et al., 2023).
"""

from typing import Any, List, Optional

import numpy as np

from ._register import maybe_register
from ._backends import _TigramiteBase
from .base import CITKTest, hash_parameters

__all__ = ["CMIknn", "CMIknnMixed", "MCMIknn"]


class CMIknn(_TigramiteBase):
    """Tigramite's k-NN conditional mutual information CI test."""

    method_name = "cmiknn"
    class_candidates = ["tigramite.independence_tests.cmiknn.CMIknn"]


class CMIknnMixed(_TigramiteBase):
    """Tigramite's mixed-data k-NN CMI variant (continuous + discrete)."""

    method_name = "cmiknn_mixed"
    class_candidates = [
        "tigramite.independence_tests.cmiknn.CMIknnMixed",
        "tigramite.independence_tests.cmiknn_mixed.CMIknnMixed",
    ]


class MCMIknn(CITKTest):
    """Wrapper around the vendored mCMIkNN kNN-CMI test (Hügle et al., 2023)."""

    supported_dtypes = {"continuous", "discrete"}
    accepted_kwargs = {"test_kwargs"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.test_kwargs = kwargs.get("test_kwargs", {})
        self.check_cache_method_consistent(
            "mcmiknn", hash_parameters({"test_kwargs": self.test_kwargs})
        )

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> float:
        from citests._vendor.indeptests import mCMIkNN

        x = self.data[:, X]
        y = self.data[:, Y]
        z = self.data[:, condition_set] if condition_set else None

        test = mCMIkNN(**self.test_kwargs)
        return float(test.compute_pval(x, y, z))


maybe_register("cmiknn", CMIknn)
maybe_register("cmiknn_mixed", CMIknnMixed)
maybe_register("mcmiknn", MCMIknn)
