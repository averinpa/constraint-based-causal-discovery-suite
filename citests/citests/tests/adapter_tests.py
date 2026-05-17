"""Adapter-strategy CI tests (survey family).

Tests in this module pre-process the data before delegating to an
underlying test:

* :class:`DiscChiSq`, :class:`DiscGSq` — equal-frequency discretize
  continuous columns, then apply chi-square / G-square.
* :class:`DummyFisherZ` — one-hot expand categorical columns, then
  Fisher-Z on the dummy-coded matrix.
* :class:`HarteminkChiSq` — joint Hartemink discretization (preserves
  pairwise mutual information) via R bnlearn, then chi-square via
  causal-learn.

All four require the optional ``[causallearn]`` extra; HarteminkChiSq
additionally requires the ``[r]`` extra and the R ``bnlearn`` package.
When causal-learn is missing, the classes are bound to ``None``
placeholders.
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

from citests.exceptions import CITKDependencyError
from ._backends import _load_bnlearn_package
from ._register import maybe_register
from .base import (
    CITKTest,
    _is_categorical_column,
    hash_parameters,
    inner_test_kwargs,
)

__all__ = ["DiscChiSq", "DiscGSq", "DummyFisherZ", "HarteminkChiSq"]


def _equal_frequency_discretize(data: np.ndarray, n_bins: int = 5) -> np.ndarray:
    df = pd.DataFrame(data)
    out = np.zeros_like(df.to_numpy(), dtype=int)
    for j in range(df.shape[1]):
        col = df.iloc[:, j]
        if _is_categorical_column(col.to_numpy()):
            out[:, j] = pd.Categorical(col).codes
        else:
            binned = pd.qcut(col, q=n_bins, labels=False, duplicates="drop")
            out[:, j] = np.asarray(binned, dtype=int)
    return out


try:
    from causallearn.utils.cit import CIT, Chisq_or_Gsq
    _HAS_CAUSALLEARN = True
except ImportError:
    _HAS_CAUSALLEARN = False


if _HAS_CAUSALLEARN:

    class DiscChiSq(CITKTest):
        supported_dtypes = {"continuous", "discrete"}
        accepted_kwargs = {"n_bins"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            self.n_bins = kwargs.get("n_bins", 5)
            disc_data = _equal_frequency_discretize(data, n_bins=self.n_bins)
            super().__init__(disc_data, **kwargs)
            self.check_cache_method_consistent(
                "disc_chisq", hash_parameters({"n_bins": self.n_bins})
            )
            self.test_instance = Chisq_or_Gsq(
                self.data, method_name="chisq", **inner_test_kwargs(kwargs)
            )

        def _compute(
            self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any
        ) -> float:
            return float(self.test_instance(X, Y, condition_set))

    class DiscGSq(CITKTest):
        supported_dtypes = {"continuous", "discrete"}
        accepted_kwargs = {"n_bins"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            self.n_bins = kwargs.get("n_bins", 5)
            disc_data = _equal_frequency_discretize(data, n_bins=self.n_bins)
            super().__init__(disc_data, **kwargs)
            self.check_cache_method_consistent(
                "disc_gsq", hash_parameters({"n_bins": self.n_bins})
            )
            self.test_instance = Chisq_or_Gsq(
                self.data, method_name="gsq", **inner_test_kwargs(kwargs)
            )

        def _compute(
            self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any
        ) -> float:
            return float(self.test_instance(X, Y, condition_set))

    class DummyFisherZ(CITKTest):
        supported_dtypes = {"continuous", "discrete"}
        accepted_kwargs = {"max_levels"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            self.max_levels = kwargs.get("max_levels", 10)
            expanded_blocks = []
            self.col_map = {}
            cursor = 0
            for j in range(data.shape[1]):
                col = data[:, j]
                if _is_categorical_column(col, max_levels=self.max_levels):
                    dummies = pd.get_dummies(
                        pd.Series(col).astype("category"), drop_first=True
                    )
                    if dummies.shape[1] == 0:
                        block = np.zeros((len(col), 1), dtype=float)
                    else:
                        block = dummies.to_numpy(dtype=float)
                else:
                    block = col.reshape(-1, 1).astype(float)
                expanded_blocks.append(block)
                self.col_map[j] = list(range(cursor, cursor + block.shape[1]))
                cursor += block.shape[1]

            expanded = np.hstack(expanded_blocks)
            super().__init__(expanded, **kwargs)
            self.check_cache_method_consistent(
                "dummy_fisherz", hash_parameters({"max_levels": self.max_levels})
            )
            self.test_instance = CIT(
                self.data, method_name="fisherz", **inner_test_kwargs(kwargs)
            )

        def _compute(
            self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs: Any
        ) -> float:
            x_cols = self.col_map[X]
            y_cols = self.col_map[Y]
            z_cols: list[int] = []
            for z in (condition_set or []):
                z_cols.extend(self.col_map[z])

            p_vals = []
            for x_col in x_cols:
                for y_col in y_cols:
                    cond = [c for c in z_cols if c != x_col and c != y_col]
                    p_vals.append(float(self.test_instance(x_col, y_col, cond)))

            if not p_vals:
                return 1.0
            if len(p_vals) == 1:
                return p_vals[0]
            safe_p_vals = np.clip(np.asarray(p_vals, dtype=float), 1e-300, 1.0)
            combined_p = combine_pvalues(safe_p_vals, method="fisher")[1]
            return float(np.clip(combined_p, 0.0, 1.0))

    class HarteminkChiSq(CITKTest):
        """Hartemink discretization (R/bnlearn) + chi-squared CI (causal-learn).

        Requires both ``[r]`` and ``[causallearn]`` extras. Construction
        raises :class:`CITKDependencyError` if causal-learn is missing.
        """

        supported_dtypes = {"continuous", "discrete"}
        accepted_kwargs = {"breaks", "ibreaks", "data_type"}

        def __init__(self, data: np.ndarray, **kwargs: Any) -> None:
            self.breaks = kwargs.get("breaks", 4)
            self.ibreaks = kwargs.get("ibreaks", 10)
            self.data_type = kwargs.get("data_type", None)
            discretized = self._hartemink_discretize(data)
            super().__init__(discretized, **kwargs)
            self.check_cache_method_consistent(
                "hartemink_chisq",
                hash_parameters({
                    "breaks": self.breaks,
                    "ibreaks": self.ibreaks,
                    "data_type": self.data_type,
                }),
            )
            self.test_instance = Chisq_or_Gsq(
                self.data, method_name="chisq", **inner_test_kwargs(kwargs)
            )

        def _hartemink_discretize(self, data: np.ndarray) -> np.ndarray:
            # Partition columns by type (0=continuous, 1=discrete in data_type array).
            # bnlearn's joint Hartemink discretizer rejects integer/categorical columns,
            # so categoricals are passed through unchanged and only continuous columns
            # are discretized. With >=2 continuous columns we use joint Hartemink (preserves
            # cross-column MI among continuous variables); with 1 continuous column we fall
            # back to equal-frequency binning (Hartemink's joint algorithm needs >=2 cols).
            n_rows, n_cols = data.shape
            if self.data_type is not None:
                cont_idx = [j for j in range(n_cols) if int(self.data_type[0, j]) == 0]
            else:
                cont_idx = []
                for j in range(n_cols):
                    col = data[:, j]
                    looks_continuous = not (
                        np.all(col == col.astype(int)) and len(np.unique(col)) < 20
                    )
                    if looks_continuous:
                        cont_idx.append(j)
            cat_idx = [j for j in range(n_cols) if j not in cont_idx]

            out = np.zeros((n_rows, n_cols), dtype=int)

            if len(cont_idx) >= 2:
                pandas2ri, bnlearn_pkg = _load_bnlearn_package()
                from rpy2.robjects import default_converter
                from rpy2.robjects.conversion import localconverter

                cont_data = data[:, cont_idx]
                frame = pd.DataFrame(cont_data, columns=[f"v{i}" for i in range(len(cont_idx))])
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
                for i, j in enumerate(cont_idx):
                    out[:, j] = pd.Categorical(disc_df.iloc[:, i]).codes
            elif len(cont_idx) == 1:
                j = cont_idx[0]
                binned = pd.qcut(data[:, j], q=self.breaks, labels=False, duplicates="drop")
                out[:, j] = np.asarray(binned, dtype=int)

            for j in cat_idx:
                out[:, j] = pd.Categorical(data[:, j]).codes

            return out

        def _compute(
            self,
            X: int,
            Y: int,
            condition_set: Optional[List[int]] = None,
            **kwargs: Any,
        ) -> float:
            return float(self.test_instance(X, Y, condition_set))

    maybe_register("disc_chisq", DiscChiSq)
    maybe_register("disc_gsq", DiscGSq)
    maybe_register("dummy_fisherz", DummyFisherZ)
    maybe_register("hartemink_chisq", HarteminkChiSq)
else:
    DiscChiSq = None  # type: ignore[assignment]
    DiscGSq = None  # type: ignore[assignment]
    DummyFisherZ = None  # type: ignore[assignment]
    HarteminkChiSq = None  # type: ignore[assignment]
