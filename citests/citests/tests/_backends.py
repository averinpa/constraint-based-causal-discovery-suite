"""Private backend loaders shared across family-named CI test modules.

Several CI test families dispatch to the same external backend
(rpy2/R for RCIT, bnlearn, MXM; tigramite for CMIknn, RegressionCI;
etc.). Family modules import from here rather than each other so the
public taxonomy stays clean — `partial_correlation_tests`,
`kernel_tests`, etc. — without inflating each module with duplicated
loaders.

Nothing in this module is part of citests's public API.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

import numpy as np

from citests.exceptions import CITKComputationError, CITKDependencyError
from .base import CITKTest, hash_parameters

# ---------------------------------------------------------------------------
# rpy2 / R-package loaders
# ---------------------------------------------------------------------------


def _load_rcit_package():
    """Lazy-load rpy2 + R RCIT package; clear actionable error if missing."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "R-based CI tests require optional dependency 'rpy2'. "
            "Install with: pip install 'citests[r]' (or uv sync --extra r)."
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
    """Lazy-load rpy2 + R bnlearn package (Hartemink discretization)."""
    try:
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "Hartemink CI test requires optional dependency 'rpy2'. "
            "Install with: pip install 'citests[r]' (or uv sync --extra r)."
        ) from exc

    try:
        bnlearn_pkg = importr("bnlearn")
    except Exception as exc:
        raise CITKDependencyError(
            "R package 'bnlearn' is required for Hartemink discretization. "
            "Install from CRAN in your R environment."
        ) from exc
    return pandas2ri, bnlearn_pkg


def _load_mxm_package():
    """Lazy-load rpy2 + R MXM package (ci.mm test)."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "R-based MXM tests require optional dependency 'rpy2'. "
            "Install with: pip install 'citests[r]' (or uv sync --extra r)."
        ) from exc
    try:
        ro.r("library(MXM)")
    except Exception as exc:
        raise CITKDependencyError(
            "R package 'MXM' is required for ci.mm test. "
            "Install from CRAN: install.packages('MXM')."
        ) from exc
    return ro, pandas2ri


def _to_r_vector(ro, arr: np.ndarray):
    return ro.FloatVector(np.asarray(arr, dtype=float).ravel())


def _to_r_matrix(ro, arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    return ro.r.matrix(
        ro.FloatVector(arr.ravel(order="F")),
        nrow=arr.shape[0],
        ncol=arr.shape[1],
    )


def _extract_rcit_p_value(result) -> float:
    try:
        return float(result.rx2("p")[0])
    except Exception as exc:
        raise CITKComputationError("Could not extract 'p' from RCIT result.") from exc


# ---------------------------------------------------------------------------
# tigramite loader + base class
# ---------------------------------------------------------------------------


def _load_tigramite():
    try:
        tp = importlib.import_module("tigramite.data_processing")
    except ModuleNotFoundError as exc:
        raise CITKDependencyError(
            "Tigramite-based CI tests require optional dependency 'tigramite'. "
            "Install with: pip install tigramite."
        ) from exc
    return tp


def _load_tigramite_test_class(candidates: List[str]):
    last_exc: Optional[Exception] = None
    for path in candidates:
        module_name, class_name = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            last_exc = exc
    raise CITKDependencyError(
        f"Could not import tigramite test class from {candidates}"
    ) from last_exc


def _extract_tigramite_pvalue(result) -> float:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return float(result[1])
        raise CITKComputationError("Unexpected tuple result from tigramite run_test.")
    if isinstance(result, dict):
        for key in ("pval", "p_value", "p"):
            if key in result:
                return float(result[key])
    if isinstance(result, (float, np.floating)):
        return float(result)
    raise CITKComputationError(f"Unexpected tigramite result type: {type(result)}")


class _TigramiteBase(CITKTest):
    """Shared base for tigramite-backed CI tests."""

    supported_dtypes = {"continuous", "discrete"}
    accepted_kwargs = {"test_kwargs", "data_type"}
    method_name = ""
    class_candidates: List[str] = []

    def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
        super().__init__(data, **kwargs)
        self.test_kwargs = kwargs.get("test_kwargs", {})
        self.data_type = kwargs.get("data_type", None)
        self.check_cache_method_consistent(
            self.method_name,
            hash_parameters({"test_kwargs": self.test_kwargs, "data_type": self.data_type}),
        )

    def _compute(
        self,
        X: int,
        Y: int,
        condition_set: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> float:
        tp = _load_tigramite()
        test_cls = _load_tigramite_test_class(self.class_candidates)

        dataframe = tp.DataFrame(self.data, data_type=self.data_type)
        test = test_cls(**self.test_kwargs)
        test.set_dataframe(dataframe)
        z = [(int(c), 0) for c in (condition_set or [])]
        result = test.run_test(X=[(int(X), 0)], Y=[(int(Y), 0)], Z=z, tau_max=0)
        return _extract_tigramite_pvalue(result)
