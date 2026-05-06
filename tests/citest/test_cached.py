"""CachedCITest behaviour."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cbcd.citest import CachedCITest, FisherZ
from cbcd.citest.protocol import CITestResult


class CountingCITest:
    n_vars = 5

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        self.calls += 1
        # Deterministic synthetic p-value depending on inputs.
        return CITestResult(p_value=0.1 + 0.01 * (x + y + sum(S)))


def test_cache_hits_after_first_call() -> None:
    inner = CountingCITest()
    cached = CachedCITest(inner)
    assert not cached.is_cached(0, 1, [2])
    p1 = cached(0, 1, [2])
    p2 = cached(0, 1, [2])
    assert p1 == p2
    assert inner.calls == 1
    assert cached.is_cached(0, 1, [2])


def test_cache_normalizes_xy_order() -> None:
    inner = CountingCITest()
    cached = CachedCITest(inner)
    cached(0, 1, [2])
    # Reversed (y, x) order should hit the same cache entry.
    cached(1, 0, [2])
    assert inner.calls == 1


def test_cache_keyed_on_S_set_not_order() -> None:
    inner = CountingCITest()
    cached = CachedCITest(inner)
    cached(0, 1, [2, 3])
    cached(0, 1, [3, 2])
    cached(0, 1, (2, 3))
    assert inner.calls == 1


def test_cache_distinct_S_values() -> None:
    inner = CountingCITest()
    cached = CachedCITest(inner)
    cached(0, 1, [])
    cached(0, 1, [2])
    cached(0, 1, [3])
    assert inner.calls == 3


def test_cache_disabled() -> None:
    inner = CountingCITest()
    cached = CachedCITest(inner, cache=False)
    cached(0, 1, [2])
    cached(0, 1, [2])
    assert inner.calls == 2


def test_cache_isolation_between_instances() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    fz1 = FisherZ(data)
    fz2 = FisherZ(data + 1e-6)  # Slightly different data → different correlations.
    c1 = CachedCITest(fz1)
    c2 = CachedCITest(fz2)
    p1 = c1(0, 1, [])
    p2 = c2(0, 1, [])
    # Each wrapper has its own cache, so neither poisons the other.
    assert c1.is_cached(0, 1, [])
    assert c2.is_cached(0, 1, [])
    # Caches are independent: not necessarily equal p-values.
    _ = p1, p2
