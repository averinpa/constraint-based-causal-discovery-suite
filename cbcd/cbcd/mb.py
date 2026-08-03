"""Markov-blanket / parents-children discovery primitives (roadmap Phase 1).

Local, per-target CI-test routines that the region-grow (Phase 3) builds on:

* ``iamb`` / ``inter_iamb`` — Markov blanket of a target (grow-shrink; inter- interleaves the shrink).
* ``mmpc`` — parents-and-children set (Max-Min PC forward/backward + AND symmetry correction).

All are ``CITest``-driven (``X ⫫ Y | S`` when ``ci(x, y, S) > alpha``) and accept a ``recorder`` so
every CI query is instrumented for the cost-vs-accuracy frontier. They emit ``record_ci`` only; the
run-level ``begin_run``/``finish_run`` brackets belong to the calling algorithm (Phase 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError
from cbcd.recording import RunRecorder, _resolve_recorder


def _check_target(ci: CITest, target: int) -> int:
    t = int(target)
    if not (0 <= t < ci.n_vars):
        raise CBCDInputError(f"target must be in [0, {ci.n_vars}); got {target}")
    return t


class _Tester:
    """Bundles a CITest with a recorder + cache-hit probe so each call is one instrumented query."""

    def __init__(self, ci: CITest, recorder: RunRecorder | None) -> None:
        self._ci = ci
        self._rec = _resolve_recorder(recorder)
        self._is_cached = getattr(ci, "is_cached", None)

    def p(self, x: int, y: int, cond: Sequence[int]) -> float:
        s = list(cond)
        hit = bool(self._is_cached(x, y, s)) if self._is_cached is not None else False
        val = float(self._ci(x, y, s))
        self._rec.record_ci(
            x=int(x), y=int(y), S=tuple(int(c) for c in s),
            p_value=val, depth=len(s), was_cache_hit=hit,
        )
        return val

    def max_p_over_subsets(
        self, x: int, t: int, pool: Sequence[int], cap: int | None
    ) -> float:
        """Largest p over conditioning subsets of ``pool`` (incl. the empty set), up to size ``cap``.
        A value ``> alpha`` means some subset separates ``x`` from ``t``."""
        pool = list(pool)
        upper = len(pool) if cap is None else min(len(pool), cap)
        best = self.p(x, t, [])
        for size in range(1, upper + 1):
            for subset in combinations(pool, size):
                best = max(best, self.p(x, t, subset))
        return best


def iamb(
    ci: CITest,
    target: int,
    *,
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    recorder: RunRecorder | None = None,
) -> frozenset[int]:
    """IAMB (Tsamardinos et al. 2003): grow the MB by strongest association to the target given the
    current MB, then shrink by dropping any member independent of the target given the rest."""
    t = _check_target(ci, target)
    tester = _Tester(ci, recorder)
    others = [v for v in range(ci.n_vars) if v != t]
    mb: set[int] = set()

    while True:
        if max_cond_set is not None and len(mb) >= max_cond_set:
            break
        cond = sorted(mb)
        best_x: int | None = None
        best_p = alpha
        for x in others:
            if x in mb:
                continue
            p = tester.p(x, t, cond)
            if p <= best_p and (best_x is None or p < best_p or x < best_x):
                best_p, best_x = p, x
        if best_x is None:
            break
        mb.add(best_x)

    for x in sorted(mb):
        if tester.p(x, t, sorted(mb - {x})) > alpha:
            mb.discard(x)
    return frozenset(mb)


def inter_iamb(
    ci: CITest,
    target: int,
    *,
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    recorder: RunRecorder | None = None,
) -> frozenset[int]:
    """Interleaved IAMB: run the shrink step after every grow addition, keeping the MB (and thus the
    conditioning sets) as small as possible during the search."""
    t = _check_target(ci, target)
    tester = _Tester(ci, recorder)
    others = [v for v in range(ci.n_vars) if v != t]
    mb: set[int] = set()

    while True:
        if max_cond_set is not None and len(mb) >= max_cond_set:
            break
        cond = sorted(mb)
        best_x: int | None = None
        best_p = alpha
        for x in others:
            if x in mb:
                continue
            p = tester.p(x, t, cond)
            if p <= best_p and (best_x is None or p < best_p or x < best_x):
                best_p, best_x = p, x
        if best_x is None:
            break
        mb.add(best_x)
        changed = True
        while changed:
            changed = False
            for x in sorted(mb):
                if tester.p(x, t, sorted(mb - {x})) > alpha:
                    mb.discard(x)
                    changed = True
    return frozenset(mb)


def grow_shrink(
    ci: CITest,
    target: int,
    *,
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    recorder: RunRecorder | None = None,
) -> frozenset[int]:
    """Grow-Shrink (Margaritis & Thrun 1999), the original MB algorithm. Grow: repeatedly add any
    variable dependent on the target given the current MB (index order), until a full pass adds
    nothing; then Shrink: drop members independent of the target given the rest.

    Differs from ``iamb`` only in the grow step: GS adds the first-found dependent variable (fast,
    order-dependent), whereas IAMB rescans and adds the *most* dependent (order-independent)."""
    t = _check_target(ci, target)
    tester = _Tester(ci, recorder)
    others = [v for v in range(ci.n_vars) if v != t]
    mb: list[int] = []

    changed = True
    while changed:
        changed = False
        for x in others:
            if x in mb:
                continue
            if max_cond_set is not None and len(mb) >= max_cond_set:
                break
            if tester.p(x, t, sorted(mb)) <= alpha:  # dependent given current MB
                mb.append(x)
                changed = True

    for x in list(mb):
        if tester.p(x, t, sorted(set(mb) - {x})) > alpha:
            mb.remove(x)
    return frozenset(mb)


def _mmpc_raw(tester: _Tester, n: int, t: int, alpha: float, cap: int | None) -> set[int]:
    others = [v for v in range(n) if v != t]
    cpc: list[int] = []

    # Forward (MaxMin): add the candidate that stays most associated under its best separator.
    while True:
        best_x: int | None = None
        best_maxp = alpha
        for x in others:
            if x in cpc:
                continue
            maxp = tester.max_p_over_subsets(x, t, cpc, cap)
            if maxp <= alpha and (best_x is None or maxp < best_maxp or x < best_x):
                best_maxp, best_x = maxp, x
        if best_x is None:
            break
        cpc.append(best_x)

    # Backward: drop x if some subset of the rest separates it from the target.
    for x in list(cpc):
        rest = [c for c in cpc if c != x]
        if tester.max_p_over_subsets(x, t, rest, cap) > alpha:
            cpc.remove(x)
    return set(cpc)


def mmpc(
    ci: CITest,
    target: int,
    *,
    alpha: float = 0.05,
    max_cond_set: int | None = None,
    symmetry: str | None = "and",
    recorder: RunRecorder | None = None,
) -> frozenset[int]:
    """MMPC (Tsamardinos et al. 2006): the parents-and-children set of ``target``.

    ``symmetry='and'`` (default) keeps ``x`` only when the relation is reciprocal
    (``x`` in raw-PC(target) *and* ``target`` in raw-PC(x)) — the standard false-positive filter.
    ``symmetry=None`` returns the raw (asymmetric) forward/backward result.
    """
    t = _check_target(ci, target)
    if symmetry not in (None, "and"):
        raise CBCDInputError(f"symmetry must be None or 'and'; got {symmetry!r}")
    tester = _Tester(ci, recorder)
    n = ci.n_vars
    raw_t = _mmpc_raw(tester, n, t, alpha, max_cond_set)
    if symmetry is None:
        return frozenset(raw_t)
    kept = {x for x in raw_t if t in _mmpc_raw(tester, n, x, alpha, max_cond_set)}
    return frozenset(kept)
