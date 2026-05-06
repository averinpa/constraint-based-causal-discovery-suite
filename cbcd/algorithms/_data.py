"""Internal data normalization (decision D1)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError


def _normalize_data(
    data: NDArray[np.float64] | pd.DataFrame,
    var_names: Sequence[str] | None = None,
) -> tuple[NDArray[np.float64], tuple[str, ...] | None]:
    """Coerce data to ``(ndarray, var_names | None)``.

    DataFrames yield ``var_names`` from their columns unless one was supplied
    explicitly. ndarrays produce ``var_names = None`` unless one was supplied.
    """
    if isinstance(data, pd.DataFrame):
        names: tuple[str, ...] | None
        if var_names is None:
            names = tuple(str(c) for c in data.columns)
        else:
            names = tuple(var_names)
            if len(names) != data.shape[1]:
                raise CBCDInputError(
                    f"var_names length {len(names)} does not match data columns {data.shape[1]}"
                )
        arr = data.to_numpy(dtype=np.float64)
    else:
        arr = np.ascontiguousarray(data, dtype=np.float64)
        names = tuple(var_names) if var_names is not None else None
        if names is not None and len(names) != arr.shape[1]:
            raise CBCDInputError(
                f"var_names length {len(names)} does not match data columns {arr.shape[1]}"
            )
    if arr.ndim != 2:
        raise CBCDInputError(f"data must be 2-D, got shape {arr.shape}")
    return arr, names
