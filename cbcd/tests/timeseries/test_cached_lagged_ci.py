"""CachedLaggedCITest invariants."""

from __future__ import annotations

from collections.abc import Sequence

from cbcd.timeseries import CachedLaggedCITest, LaggedCITestResult, LaggedVar


class CountingInner:
    n_vars = 2
    max_lag = 1

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> float:
        return self.details(x, y, S).p_value

    def details(self, x: LaggedVar, y: LaggedVar, S: Sequence[LaggedVar]) -> LaggedCITestResult:
        self.calls += 1
        return LaggedCITestResult(p_value=0.5)


def test_cache_returns_same_result_each_call() -> None:
    inner = CountingInner()
    cached = CachedLaggedCITest(inner)
    cached(LaggedVar(0, -1), LaggedVar(1, 0), [LaggedVar(0, 0)])
    cached(LaggedVar(0, -1), LaggedVar(1, 0), [LaggedVar(0, 0)])
    assert inner.calls == 1


def test_cache_key_is_unordered_in_xy() -> None:
    inner = CountingInner()
    cached = CachedLaggedCITest(inner)
    cached(LaggedVar(0, -1), LaggedVar(1, 0), [])
    cached(LaggedVar(1, 0), LaggedVar(0, -1), [])
    assert inner.calls == 1


def test_cache_key_is_unordered_in_S() -> None:
    inner = CountingInner()
    cached = CachedLaggedCITest(inner)
    cached(LaggedVar(0, 0), LaggedVar(1, 0), [LaggedVar(0, -1), LaggedVar(1, -1)])
    cached(LaggedVar(0, 0), LaggedVar(1, 0), [LaggedVar(1, -1), LaggedVar(0, -1)])
    assert inner.calls == 1


def test_cache_disabled_passes_through() -> None:
    inner = CountingInner()
    cached = CachedLaggedCITest(inner, cache=False)
    cached(LaggedVar(0, 0), LaggedVar(1, 0), [])
    cached(LaggedVar(0, 0), LaggedVar(1, 0), [])
    assert inner.calls == 2


def test_is_cached_reports_correctly() -> None:
    inner = CountingInner()
    cached = CachedLaggedCITest(inner)
    assert not cached.is_cached(LaggedVar(0, 0), LaggedVar(1, 0), [])
    cached(LaggedVar(0, 0), LaggedVar(1, 0), [])
    assert cached.is_cached(LaggedVar(0, 0), LaggedVar(1, 0), [])
