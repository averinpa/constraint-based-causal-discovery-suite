from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from causallearn.utils.cit import (
    CIT,
    Chisq_or_Gsq,
    NO_SPECIFIED_PARAMETERS_MSG,
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


def _gcm_p_value(res_x: np.ndarray, res_y: np.ndarray) -> float:
    prod = np.asarray(res_x) * np.asarray(res_y)
    denom = np.std(prod, ddof=1)
    if denom <= 0:
        return 1.0
    stat = np.sqrt(len(prod)) * np.mean(prod) / denom
    return float(2.0 * norm.sf(abs(stat)))


def _fit_residuals(model, target: np.ndarray, z: np.ndarray) -> np.ndarray:
    if z.shape[1] == 0:
        return target - np.mean(target)
    model.fit(z, target)
    return target - model.predict(z)


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


class GCMLinear(CITKTest):
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("gcm_linear", NO_SPECIFIED_PARAMETERS_MSG)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        z = self.data[:, condition_set] if condition_set else np.empty((len(self.data), 0))
        rx = _fit_residuals(LinearRegression(), self.data[:, X], z)
        ry = _fit_residuals(LinearRegression(), self.data[:, Y], z)
        return _gcm_p_value(rx, ry)


# Not registered — pycomets-based wrapper in ml_based_tests.py is used instead


class GCMRF(CITKTest):
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.n_estimators = kwargs.get("n_estimators", 200)
        self.random_state = kwargs.get("random_state", 42)
        self.n_splits = kwargs.get("n_splits", 5)
        params = f"n_estimators={self.n_estimators},seed={self.random_state},splits={self.n_splits}"
        self.check_cache_method_consistent("gcm_rf", params)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        z = self.data[:, condition_set] if condition_set else np.empty((len(self.data), 0))
        if z.shape[1] == 0:
            return _gcm_p_value(self.data[:, X] - np.mean(self.data[:, X]),
                                self.data[:, Y] - np.mean(self.data[:, Y]))
        rx = np.zeros(len(self.data))
        ry = np.zeros(len(self.data))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, test_idx) in enumerate(kf.split(z)):
            rf_x = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state + 10 * fold, n_jobs=-1)
            rf_y = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state + 10 * fold + 1, n_jobs=-1)
            rf_x.fit(z[train_idx], self.data[train_idx, X])
            rf_y.fit(z[train_idx], self.data[train_idx, Y])
            rx[test_idx] = self.data[test_idx, X] - rf_x.predict(z[test_idx])
            ry[test_idx] = self.data[test_idx, Y] - rf_y.predict(z[test_idx])
        return _gcm_p_value(rx, ry)


# Not registered — pycomets-based wrapper in ml_based_tests.py is used instead


class WGCMRF(CITKTest):
    supported_dtypes = {"continuous"}

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.n_estimators = kwargs.get("n_estimators", 200)
        self.random_state = kwargs.get("random_state", 42)
        self.n_splits = kwargs.get("n_splits", 2)
        params = f"n_estimators={self.n_estimators},seed={self.random_state},splits={self.n_splits}"
        self.check_cache_method_consistent("wgcm_rf", params)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        z = self.data[:, condition_set] if condition_set else np.empty((len(self.data), 0))
        if z.shape[1] == 0:
            return _gcm_p_value(self.data[:, X] - np.mean(self.data[:, X]), self.data[:, Y] - np.mean(self.data[:, Y]))

        rx = np.zeros(len(self.data))
        ry = np.zeros(len(self.data))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, test_idx) in enumerate(kf.split(z)):
            rf_x = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state + 10 * fold, n_jobs=-1
            )
            rf_y = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state + 10 * fold + 1, n_jobs=-1
            )
            rf_x.fit(z[train_idx], self.data[train_idx, X])
            rf_y.fit(z[train_idx], self.data[train_idx, Y])
            rx[test_idx] = self.data[test_idx, X] - rf_x.predict(z[test_idx])
            ry[test_idx] = self.data[test_idx, Y] - rf_y.predict(z[test_idx])

        return _gcm_p_value(rx, ry)


# Not registered — dropped from benchmark (original R wGCM broken on R 4.5.1)
