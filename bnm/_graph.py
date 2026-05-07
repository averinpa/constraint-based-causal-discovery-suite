"""Internal concrete `_Graph` implementing GraphLike.

Used by:
    - `bnm.adapter._to_endpoints` to box the normalised triple into a
      Protocol-conforming object when callers need one.
    - `bnm.markov_blanket` to return sub-graphs.
    - `tests/fixtures.py` for hand-built canonical fixtures.

Not part of the public API. Callers passing their own graph types
(cbcd's DAG/CPDAG/PAG, custom wrappers) never see this class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class _Graph:
    """A concrete `GraphLike` over an int8 endpoint matrix."""

    n_vars: int
    endpoints: NDArray[np.int8]
    var_names: tuple[str, ...] | None
