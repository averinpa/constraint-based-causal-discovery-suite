"""Fisher-Z conditional-independence test for Gaussian data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from cbcd.citest.protocol import CITestResult
from cbcd.exceptions import CBCDDataError, CBCDInputError

_R_CLIP = 1.0 - 1e-15


class FisherZ:
    """Fisher-Z conditional-independence test.

    Pre-computes the sample correlation matrix once. Each query inverts the
    relevant submatrix to obtain the partial correlation, then applies Fisher's
    r-to-z transform with ``df = n - len(S) - 3``.

    Strict Gaussian assumption; for mixed-data, use a different test.
    """

    n_vars: int

    def __init__(
        self,
        data: NDArray[np.float64] | pd.DataFrame,
        var_names: tuple[str, ...] | None = None,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            if var_names is None:
                var_names = tuple(str(c) for c in data.columns)
            data = data.to_numpy(dtype=np.float64)
        else:
            data = np.ascontiguousarray(data, dtype=np.float64)

        if data.ndim != 2:
            raise CBCDInputError(f"data must be 2-D, got shape {data.shape}")
        if np.any(np.isnan(data)):
            raise CBCDDataError("Fisher-Z does not support NaN entries")
        if np.any(np.isinf(data)):
            raise CBCDDataError("Fisher-Z does not support infinite entries")

        self.n_samples, self.n_vars = data.shape
        if self.n_samples < 4:
            raise CBCDDataError(
                f"Fisher-Z needs at least 4 samples for a usable test, got {self.n_samples}"
            )

        # Sample correlation matrix.
        centered = data - data.mean(axis=0, keepdims=True)
        std = centered.std(axis=0, ddof=1, keepdims=True)
        if np.any(std == 0):
            raise CBCDDataError("Fisher-Z: a column has zero variance")
        normed = centered / std
        self._corr: NDArray[np.float64] = (normed.T @ normed) / (self.n_samples - 1)
        self.var_names = var_names

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        S_tuple = tuple(int(s) for s in S)
        if x == y or x in S_tuple or y in S_tuple:
            raise CBCDInputError(
                f"FisherZ requires x, y distinct and not in S; got x={x}, y={y}, S={S_tuple}"
            )

        df = self.n_samples - len(S_tuple) - 3
        if df <= 0:
            return CITestResult(p_value=1.0, statistic=0.0, df=df, n_effective=self.n_samples)

        if not S_tuple:
            r = float(self._corr[x, y])
        else:
            idx = [x, y, *S_tuple]
            sub = self._corr[np.ix_(idx, idx)]
            try:
                inv = np.linalg.inv(sub)
            except np.linalg.LinAlgError:
                return CITestResult(p_value=1.0, statistic=0.0, df=df, n_effective=self.n_samples)
            denom = inv[0, 0] * inv[1, 1]
            if denom <= 0:
                return CITestResult(p_value=1.0, statistic=0.0, df=df, n_effective=self.n_samples)
            r = float(-inv[0, 1] / np.sqrt(denom))

        r = max(min(r, _R_CLIP), -_R_CLIP)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        stat = abs(z) * np.sqrt(df)
        p_value = 2.0 * float(norm.sf(stat))
        return CITestResult(
            p_value=p_value,
            statistic=float(stat),
            df=df,
            n_effective=self.n_samples,
            extra={"r": r},
        )
