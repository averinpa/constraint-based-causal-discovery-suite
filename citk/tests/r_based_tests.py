from typing import List, Optional

import numpy as np
import pandas as pd
from causallearn.utils.cit import (
    Chisq_or_Gsq,
    NO_SPECIFIED_PARAMETERS_MSG,
    register_ci_test,
)

from citk.exceptions import CITKComputationError, CITKDependencyError
from .base import CITKTest, hash_parameters, inner_test_kwargs


def _load_rcit_package():
    """
    Lazy-load rpy2 + RCIT package and raise a clear actionable error if missing.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "R-based CI tests require optional dependency 'rpy2'. "
            "Install with: pip install 'citk[r]' (or uv sync --extra r)."
        ) from exc

    try:
        rcit_pkg = importr("RCIT")
    except Exception as exc:
        raise CITKDependencyError(
            "R package 'RCIT' is required for RCoT/RCIT tests. "
            "Install in R from GitHub: ericstrobl/RCIT."
        ) from exc

    return ro, rcit_pkg


def _load_bnlearn_package():
    try:
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "Hartemink CI test requires optional dependency 'rpy2'. "
            "Install with: pip install 'citk[r]' (or uv sync --extra r)."
        ) from exc

    try:
        bnlearn_pkg = importr("bnlearn")
    except Exception as exc:
        raise CITKDependencyError(
            "R package 'bnlearn' is required for Hartemink discretization. "
            "Install from CRAN in your R environment."
        ) from exc
    return pandas2ri, bnlearn_pkg


def _to_r_vector(ro, arr: np.ndarray):
    return ro.FloatVector(np.asarray(arr, dtype=float).ravel())


def _to_r_matrix(ro, arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    return ro.r.matrix(ro.FloatVector(arr.ravel(order="F")), nrow=arr.shape[0], ncol=arr.shape[1])


def _extract_p_value(result) -> float:
    try:
        return float(result.rx2("p")[0])
    except Exception as exc:
        raise CITKComputationError("Could not extract 'p' from RCIT result.") from exc


class _RCITBase(CITKTest):
    supported_dtypes = {"continuous"}
    method_name = ""
    rcit_func_name = ""

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent(self.method_name, NO_SPECIFIED_PARAMETERS_MSG)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        ro, rcit_pkg = _load_rcit_package()

        x = _to_r_vector(ro, self.data[:, X])
        y = _to_r_vector(ro, self.data[:, Y])
        if condition_set:
            z = _to_r_matrix(ro, self.data[:, condition_set])
            result = getattr(rcit_pkg, self.rcit_func_name)(x, y, z)
        else:
            result = getattr(rcit_pkg, self.rcit_func_name)(x, y)

        return _extract_p_value(result)


class RCoT(_RCITBase):
    method_name = "rcot"
    rcit_func_name = "RCoT"


class RCIT(_RCITBase):
    method_name = "rcit"
    rcit_func_name = "RCIT"


class HarteminkChiSq(CITKTest):
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        self.breaks = kwargs.get("breaks", 4)
        self.ibreaks = kwargs.get("ibreaks", 10)
        discretized = self._hartemink_discretize(data)
        super().__init__(discretized, **kwargs)
        self.check_cache_method_consistent(
            "hartemink_chisq",
            hash_parameters({"breaks": self.breaks, "ibreaks": self.ibreaks}),
        )
        self.test_instance = Chisq_or_Gsq(self.data, method_name="chisq", **inner_test_kwargs(kwargs))

    def _hartemink_discretize(self, data: np.ndarray) -> np.ndarray:
        pandas2ri, bnlearn_pkg = _load_bnlearn_package()
        from rpy2.robjects import default_converter
        from rpy2.robjects.conversion import localconverter

        frame = pd.DataFrame(data, columns=[f"v{i}" for i in range(data.shape[1])])
        with localconverter(default_converter + pandas2ri.converter):
            r_frame = pandas2ri.py2rpy(frame)
            disc_df = bnlearn_pkg.discretize(
                r_frame,
                method="hartemink",
                breaks=self.breaks,
                ibreaks=self.ibreaks,
            )
        if not isinstance(disc_df, pd.DataFrame):
            disc_df = pd.DataFrame(disc_df)
        out = np.zeros((len(disc_df), disc_df.shape[1]), dtype=int)
        for j, col in enumerate(disc_df.columns):
            out[:, j] = pd.Categorical(disc_df[col]).codes
        return out

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        return float(self.test_instance(X, Y, condition_set))


register_ci_test("rcot", RCoT)
register_ci_test("rcit", RCIT)
register_ci_test("hartemink_chisq", HarteminkChiSq)


def _load_mxm_package():
    """Lazy-load rpy2 + MXM R package."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "R-based MXM tests require optional dependency 'rpy2'. "
            "Install with: pip install 'citk[r]' (or uv sync --extra r)."
        ) from exc
    try:
        ro.r("library(MXM)")
    except Exception as exc:
        raise CITKDependencyError(
            "R package 'MXM' is required for ci.mm test. "
            "Install from CRAN: install.packages('MXM')."
        ) from exc
    return ro, pandas2ri


class CiMM(CITKTest):
    """Symmetric regression-based CI test from R MXM package (Tsagris et al., 2018).

    ci.mm automatically selects the regression model based on variable type:
    linear regression for continuous, logistic for binary/categorical. It runs
    two asymmetric likelihood-ratio tests (X→Y and Y→X) and combines them.
    Handles mixed continuous/categorical data natively.
    """
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
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

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        ro, _ = _load_mxm_package()

        all_cols = [X, Y] + (condition_set or [])
        sub = self.data[:, all_cols]
        mxm_types = self._get_mxm_type(all_cols)

        # Build R data.frame column by column
        n_rows = sub.shape[0]
        r_cols = {}
        for i in range(sub.shape[1]):
            col = sub[:, i]
            if mxm_types[i] == "nominal":
                r_cols[f"v{i}"] = ro.IntVector(col.astype(int))
            else:
                r_cols[f"v{i}"] = ro.FloatVector(col.ravel())

        r_df = ro.DataFrame(r_cols)
        ro.globalenv["dat"] = r_df
        ro.globalenv["type_vec"] = ro.StrVector(mxm_types)

        # ci.mm(ind1, ind2, cs, dat, type)
        # Indices are 1-based in R
        if condition_set:
            cs_r = ro.IntVector(list(range(3, 3 + len(condition_set))))
            ro.globalenv["cs_r"] = cs_r
            result = ro.r("ci.mm(ind1=1, ind2=2, cs=cs_r, dat=dat, type=type_vec)")
        else:
            result = ro.r("ci.mm(ind1=1, ind2=2, cs=NULL, dat=dat, type=type_vec)")

        # Result: [test_stat, logged_p_value, df]
        logged_p = float(result[1])
        return float(np.exp(logged_p))


register_ci_test("ci_mm", CiMM)
