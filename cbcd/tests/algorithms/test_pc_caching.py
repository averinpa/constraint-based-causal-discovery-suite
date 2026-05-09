"""CachedCITest dedupes repeated CI calls inside pc()."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cbcd import pc
from cbcd.citest.protocol import CITestResult


class CountingFisherZ:
    """Wraps FisherZ but counts every uncached CI call."""

    n_vars: int

    def __init__(self, data: np.ndarray) -> None:
        from cbcd.citest import FisherZ

        self._inner = FisherZ(data)
        self.n_vars = self._inner.n_vars
        self.calls = 0

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        self.calls += 1
        return self._inner.details(x, y, S)


def test_pc_dedupes_ci_calls_via_cache() -> None:
    from tests.algorithms.test_pc_fisherz import _sample_linear_gaussian
    from tests.fixtures import ALL_FIXTURES

    rng = np.random.default_rng(0)
    dag, _ = ALL_FIXTURES["m_structure"]()
    data = _sample_linear_gaussian(dag, 5000, rng)

    counting = CountingFisherZ(data)
    pc(data, ci_test=counting, alpha=0.05)

    # No duplicate uncached calls: every (x, y, S) was unique upstream of the cache.
    seen: set[tuple[int, int, frozenset[int]]] = set()

    # We re-run pc() and intercept calls to verify uniqueness via a second
    # counting wrapper that records (x, y, S) tuples it sees.
    class TrackingFisherZ(CountingFisherZ):
        seen_keys: set[tuple[int, int, frozenset[int]]]

        def __init__(self, data: np.ndarray) -> None:
            super().__init__(data)
            self.seen_keys = seen

        def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
            a, b = (x, y) if x <= y else (y, x)
            key = (a, b, frozenset(S))
            assert key not in self.seen_keys, (
                f"CachedCITest leaked duplicate call to inner CI: {key}"
            )
            self.seen_keys.add(key)
            return super().details(x, y, S)

    pc(data, ci_test=TrackingFisherZ(data), alpha=0.05)
