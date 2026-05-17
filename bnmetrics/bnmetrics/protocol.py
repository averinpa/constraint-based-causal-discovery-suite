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
    of bnmetrics required. cbcd's ``DAG``, ``CPDAG``, ``PAG`` instances
    conform with zero adaptation.

    The required attributes are:

    - ``n_vars`` — number of variables (rows / columns of
      ``endpoints``).
    - ``endpoints`` — square ``(n_vars, n_vars)`` endpoint-mark matrix
      following the :class:`bnmetrics.EndpointMark` convention. Must be an
      attribute, not a property — bnmetrics metric code reads this in hot
      loops.
    - ``var_names`` — optional human-readable variable names; ``None``
      means index-only.
    """

    n_vars: int
    endpoints: NDArray[np.int8]
    var_names: tuple[str, ...] | None
