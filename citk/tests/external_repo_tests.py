from typing import Any, List, Optional

import numpy as np
from causallearn.utils.cit import register_ci_test

from .base import CITKTest, hash_parameters


class MCMIknn(CITKTest):
    """Wrapper around the vendored mCMIkNN kNN-CMI test (Hügle et al., 2023)."""

    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.test_kwargs = kwargs.get("test_kwargs", {})
        self.check_cache_method_consistent(
            "mcmiknn", hash_parameters({"test_kwargs": self.test_kwargs})
        )

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any) -> float:
        from citk._vendor.indeptests import mCMIkNN

        x = self.data[:, X]
        y = self.data[:, Y]
        z = self.data[:, condition_set] if condition_set else None

        test = mCMIkNN(**self.test_kwargs)
        return float(test.compute_pval(x, y, z))


register_ci_test("mcmiknn", MCMIknn)
