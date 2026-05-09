"""CI test factory and registry (decision D2).

Public ``make_ci_test(name, data, **kwargs)`` resolves a string name to a
concrete ``CITest`` instance. ``register_ci_test(name, factory)`` extends the
registry. Only ``"fisherz"`` ships in this slice; chisq/gsq/KCI/partialcorr
are deferred.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cbcd.citest.fisherz import FisherZ
from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError

_REGISTRY: dict[str, Callable[..., CITest]] = {}


def register_ci_test(name: str, factory: Callable[..., CITest]) -> None:
    """Register a CI-test factory under a given name."""
    _REGISTRY[name] = factory


def make_ci_test(
    name: str,
    data: NDArray[np.float64] | pd.DataFrame,
    **kwargs: object,
) -> CITest:
    """Construct a registered CI test by name."""
    factory = _REGISTRY.get(name)
    if factory is None:
        known = sorted(_REGISTRY.keys())
        raise CBCDInputError(f"unknown CI test {name!r}; known: {known}")
    return factory(data, **kwargs)


register_ci_test("fisherz", FisherZ)
