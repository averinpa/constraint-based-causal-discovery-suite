"""Caching wrapper for any CITest. Per-instance cache; no fragile data hashing."""

from __future__ import annotations

from collections.abc import Sequence

from cbcd.citest.protocol import CITest, CITestResult


def _cache_key(x: int, y: int, S: Sequence[int]) -> tuple[int, int, frozenset[int]]:
    a, b = (x, y) if x <= y else (y, x)
    return (a, b, frozenset(S))


class CachedCITest:
    """Wraps a ``CITest`` with a per-instance result cache.

    Two ``CachedCITest`` instances do not share state — caches are keyed by
    ``(min(x,y), max(x,y), frozenset(S))`` within a single wrapper. This avoids
    the ``str(ndarray)`` md5 collision bug found in causal-learn (audit pitfall
    #1): the wrapper itself is the dataset boundary.
    """

    n_vars: int

    def __init__(self, inner: CITest, *, cache: bool = True) -> None:
        self._inner = inner
        self.n_vars = inner.n_vars
        self._cache_enabled = cache
        self._cache: dict[tuple[int, int, frozenset[int]], CITestResult] = {}

    def __call__(self, x: int, y: int, S: Sequence[int]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: int, y: int, S: Sequence[int]) -> CITestResult:
        if not self._cache_enabled:
            return self._inner.details(x, y, S)
        key = _cache_key(x, y, S)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._inner.details(x, y, S)
        self._cache[key] = result
        return result

    def is_cached(self, x: int, y: int, S: Sequence[int]) -> bool:
        return _cache_key(x, y, S) in self._cache
