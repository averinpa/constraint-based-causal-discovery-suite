"""GraphLike Protocol — the cross-package contract for graph input."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class GraphLike(Protocol):
    """A graph backed by a square int8 endpoint-mark matrix.

    Conformance is duck-typed: any object that exposes the three
    attributes below satisfies the Protocol. No inheritance, no import
    of bnm required. cbcd's ``DAG``, ``CPDAG``, ``PAG`` instances
    conform with zero adaptation.

    Attributes
    ----------
    n_vars : int
        Number of variables (rows/columns of `endpoints`).
    endpoints : NDArray[np.int8]
        Square ``(n_vars, n_vars)`` endpoint-mark matrix. Marks follow
        the bnm.EndpointMark convention. Must be an attribute, not a
        property — bnm metric code reads this in hot loops.
    var_names : tuple[str, ...] | None
        Optional human-readable variable names; ``None`` means
        index-only.
    """

    n_vars: int
    endpoints: NDArray[np.int8]
    var_names: tuple[str, ...] | None
