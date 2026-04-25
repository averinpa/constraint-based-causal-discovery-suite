from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

from causallearn.utils.cit import (
    CIT,
    Chisq_or_Gsq,
    register_ci_test,
)
from .base import CITKTest


def _is_categorical_column(values: np.ndarray, max_levels: int = 10) -> bool:
    unique_vals = np.unique(values[~np.isnan(values)]) if np.issubdtype(values.dtype, np.number) else np.unique(values)
    if len(unique_vals) <= max_levels:
        if np.issubdtype(values.dtype, np.integer):
            return True
        if np.issubdtype(values.dtype, np.floating):
            return np.allclose(values, np.round(values), equal_nan=True)
    return False


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


class DiscChiSq(CITKTest):
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        self.n_bins = kwargs.get("n_bins", 5)
        disc_data = _equal_frequency_discretize(data, n_bins=self.n_bins)
        super().__init__(disc_data, **kwargs)
        params = f"n_bins={self.n_bins}"
        self.check_cache_method_consistent("disc_chisq", params)
        self.test_instance = Chisq_or_Gsq(self.data, method_name="chisq", **kwargs)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        return float(self.test_instance(X, Y, condition_set))


register_ci_test("disc_chisq", DiscChiSq)


class DiscGSq(CITKTest):
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        self.n_bins = kwargs.get("n_bins", 5)
        disc_data = _equal_frequency_discretize(data, n_bins=self.n_bins)
        super().__init__(disc_data, **kwargs)
        params = f"n_bins={self.n_bins}"
        self.check_cache_method_consistent("disc_gsq", params)
        self.test_instance = Chisq_or_Gsq(self.data, method_name="gsq", **kwargs)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        return float(self.test_instance(X, Y, condition_set))


register_ci_test("disc_gsq", DiscGSq)


class DummyFisherZ(CITKTest):
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        self.max_levels = kwargs.get("max_levels", 10)
        expanded_blocks = []
        self.col_map = {}
        cursor = 0
        for j in range(data.shape[1]):
            col = data[:, j]
            if _is_categorical_column(col, max_levels=self.max_levels):
                dummies = pd.get_dummies(pd.Series(col).astype("category"), drop_first=True)
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
        params = f"max_levels={self.max_levels}"
        self.check_cache_method_consistent("dummy_fisherz", params)
        self.test_instance = CIT(self.data, method_name="fisherz", **kwargs)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        x_cols = self.col_map[X]
        y_cols = self.col_map[Y]
        z_cols = []
        for z in condition_set:
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


register_ci_test("dummy_fisherz", DummyFisherZ)
