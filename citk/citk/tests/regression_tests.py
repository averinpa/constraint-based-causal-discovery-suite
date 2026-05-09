"""Regression-based CI tests (survey family): RegressionCI, CiMM.

* :class:`RegressionCI` — tigramite's mixed-data regression-residual
  test (requires the optional ``tigramite`` dependency).
* :class:`CiMM` — symmetric ``ci.mm`` test from R MXM
  (Tsagris et al., 2018; requires the ``[r]`` extra and the R MXM
  package installed in your R environment).
"""

from typing import Any, List, Optional

import numpy as np

from citk.exceptions import CITKComputationError
from ._register import maybe_register
from ._backends import _TigramiteBase, _load_mxm_package
from .base import CITKTest, hash_parameters

__all__ = ["RegressionCI", "CiMM"]


class RegressionCI(_TigramiteBase):
    """Tigramite's mixed-data regression-residual CI test."""

    method_name = "regci"
    class_candidates = ["tigramite.independence_tests.regressionCI.RegressionCI"]


class CiMM(CITKTest):
    """Symmetric regression-based CI test from R MXM package (Tsagris et al., 2018).

    ci.mm automatically selects the regression model based on variable type:
    linear regression for continuous, logistic for binary/categorical. It runs
    two asymmetric likelihood-ratio tests (X→Y and Y→X) and combines them.
    Handles mixed continuous/categorical data natively.
    """

    supported_dtypes = {"continuous", "discrete"}
    accepted_kwargs = {"data_type"}

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.data_type = kwargs.get("data_type", None)
        self.check_cache_method_consistent(
            "ci_mm", hash_parameters({"data_type": self.data_type})
        )

    def _get_mxm_type(self, col_indices):
        """Build MXM type string for the given columns.

        Uses data_type array (0=continuous, 1=discrete) if available.
        Falls back to checking if column values are all integers with few unique values.
        """
        types = []
        for j in col_indices:
            if self.data_type is not None:
                is_discrete = int(self.data_type[0, j]) == 1
            else:
                col = self.data[:, j]
                is_discrete = np.all(col == col.astype(int)) and len(np.unique(col)) < 20
            types.append("nominal" if is_discrete else "gaussian")
        return types

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> float:
        ro, _ = _load_mxm_package()

        all_cols = [X, Y] + (condition_set or [])
        sub = self.data[:, all_cols]
        mxm_types = self._get_mxm_type(all_cols)

        n_rows = sub.shape[0]  # noqa: F841 (kept for parity with R-side bookkeeping)
        r_cols = {}
        as_factor = ro.r("as.factor")
        for i in range(sub.shape[1]):
            col = sub[:, i]
            if mxm_types[i] == "nominal":
                # ci.mm dispatches on R column class; factor needed for K>=3 multinom.
                r_cols[f"v{i}"] = as_factor(ro.IntVector(col.astype(int)))
            else:
                r_cols[f"v{i}"] = ro.FloatVector(col.ravel())

        r_df = ro.DataFrame(r_cols)
        ro.globalenv["dat"] = r_df
        ro.globalenv["type_vec"] = ro.StrVector(mxm_types)

        if condition_set:
            cs_r = ro.IntVector(list(range(3, 3 + len(condition_set))))
            ro.globalenv["cs_r"] = cs_r
            result = ro.r("ci.mm(ind1=1, ind2=2, cs=cs_r, dat=dat, type=type_vec)")
        else:
            result = ro.r("ci.mm(ind1=1, ind2=2, cs=NULL, dat=dat, type=type_vec)")

        # Result: [test_stat, logged_p_value, df]
        logged_p = float(result[1])
        p = float(np.exp(logged_p))
        if not np.isfinite(p):
            raise CITKComputationError(
                f"CiMM produced non-finite p (log_p={logged_p}) for "
                f"X={X}, Y={Y}, S={condition_set}; likely multinom/ordinal fit degeneracy."
            )
        return p


maybe_register("regci", RegressionCI)
maybe_register("ci_mm", CiMM)
