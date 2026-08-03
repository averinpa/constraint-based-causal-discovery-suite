"""Shared recursive Markov-boundary elimination core for the MARVEL family.

Both :func:`cbcd.marvel` (MARVEL, Mokhtarian et al. 2021 -> CPDAG) and :func:`cbcd.lmarvel`
(L-MARVEL, Akbari et al. 2021 -> PAG) recover a skeleton + separating-set store by recursively
removing a *removable* variable, sorting the remaining variables ascending by Markov-boundary size
each round. The pieces that are identical across the two -- the instrumented cache-aware CI tester,
the ``FindAdjacent`` boundary partition (which variables in ``Mb(X)`` are truly adjacent to ``X`` vs
co-parents whose separating set we record), and the sort-by-boundary elimination loop with its
incremental boundary maintenance -- live here. Each algorithm supplies only what differs: an
``analyze`` callback (its removability test) and an ``update_mb`` callback (how boundaries of the
removed variable's context are pruned), plus its own final orientation.

Every CI query is routed through the run's cached CI test and the recorder, so the CI-test-efficiency
win is measured, not asserted; subset scans short-circuit on the first separator; and boundaries are
updated incrementally rather than recomputed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError
from cbcd.recording import RunRecorder


class _CI:
    """One instrumented, cache-aware CI query bundled with the run's alpha threshold.

    ``indep(x, y, S)`` returns ``True`` when ``X ⫫ Y | S`` (p-value ``> alpha``); every call is
    recorded (``record_ci``) so the recorder counts total / unique / cache-hit CI tests.
    """

    def __init__(self, ci: CITest, rec: RunRecorder, alpha: float) -> None:
        self._ci = ci
        self._rec = rec
        self._alpha = alpha
        self._is_cached = getattr(ci, "is_cached", None)

    def indep(self, x: int, y: int, S: Sequence[int]) -> bool:
        s = sorted(int(v) for v in S)
        hit = bool(self._is_cached(x, y, s)) if self._is_cached is not None else False
        p = float(self._ci(x, y, s))
        self._rec.record_ci(
            x=int(x), y=int(y), S=tuple(s), p_value=p, depth=len(s), was_cache_hit=hit
        )
        return p > self._alpha

    def find_sepset(self, x: int, y: int, pool: Sequence[int]) -> tuple[int, ...] | None:
        """Smallest-first subset of ``pool`` that separates ``x`` and ``y`` (empty set included);
        ``None`` if none does. Stops at the first separator found."""
        pool = list(pool)
        for size in range(len(pool) + 1):
            for S in combinations(pool, size):
                if self.indep(x, y, S):
                    return tuple(S)
        return None

    def separable_with(self, x: int, y: int, pool: Sequence[int], extra: Sequence[int]) -> bool:
        """True iff some subset ``S`` of ``pool`` gives ``X ⫫ Y | S ∪ extra``. ``extra`` is a fixed
        add-on to every conditioning set (disjoint from ``pool``). Stops at the first separator."""
        pool = list(pool)
        extra = tuple(int(e) for e in extra)
        for size in range(len(pool) + 1):
            for S in combinations(pool, size):
                if self.indep(x, y, tuple(S) + extra):
                    return True
        return False


class EliminationInfo:
    """Result of analysing one variable for removal in a given round.

    ``adjacent`` are the variables in ``Mb(X)`` found truly adjacent to ``X``; ``coparents`` maps each
    non-adjacent boundary member to a separating set (a witness the caller stores). Algorithm-specific
    analysers may attach extra fields, but the elimination loop reads only these three.
    """

    __slots__ = ("removable", "adjacent", "coparents")

    def __init__(
        self,
        removable: bool,
        adjacent: list[int],
        coparents: dict[int, tuple[int, ...]],
    ) -> None:
        self.removable = removable
        self.adjacent = adjacent
        self.coparents = coparents


def find_adjacent(x: int, mb_x: set[int], ci: _CI) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    """Partition ``Mb(X)`` into adjacencies and co-parents (``FindAdjacent``).

    ``Y in Mb(X)`` is adjacent to ``X`` iff no subset of ``Mb(X)\\{Y}`` separates them; otherwise it
    is a co-parent and the first separating subset found is recorded. Short-circuits on the first
    separator (<= ``|Mb|*2^(|Mb|-1)`` CI tests). Shared verbatim by MARVEL (its Lemma-7 step) and
    L-MARVEL.
    """
    mb = sorted(mb_x)
    adjacent: list[int] = []
    coparents: dict[int, tuple[int, ...]] = {}
    for y in mb:
        rest = [v for v in mb if v != y]
        sep = ci.find_sepset(x, y, rest)
        if sep is None:
            adjacent.append(y)
        else:
            coparents[y] = sep
    return adjacent, coparents


def recursive_skeleton(
    n: int,
    mb: dict[int, set[int]],
    ci: _CI,
    *,
    analyze: Callable[[int, set[int], _CI], EliminationInfo],
    update_mb: Callable[
        [int, EliminationInfo, dict[int, set[int]], dict[frozenset[int], tuple[int, ...]], set[int], _CI],
        set[int],
    ],
    init_sepsets: dict[frozenset[int], tuple[int, ...]] | None = None,
) -> tuple[NDArray[np.bool_], dict[frozenset[int], tuple[int, ...]]]:
    """Recursive MB-sorted elimination -> (adjacency, separating-set store).

    Each round: sort remaining variables ascending by ``|Mb|``, scan for the first ``analyze``-removable
    one, commit its adjacencies and its co-parents' separating sets, then eliminate it -- dropping it
    from every boundary and letting ``update_mb`` prune the boundaries it touched. Removability verdicts
    are memoised and recomputed only for variables whose boundary actually changed (``dirty``), so
    unchanged variables are never re-analysed.

    ``analyze(x, mb_x, ci) -> EliminationInfo`` is the algorithm's removability test; ``update_mb(chosen,
    info, mb, sepsets, remaining, ci) -> set[int]`` prunes boundaries (mutating ``mb`` / ``sepsets``) and
    returns the additional variables whose boundary changed. ``mb`` is mutated in place.
    """
    adj = np.zeros((n, n), dtype=bool)
    sepsets: dict[frozenset[int], tuple[int, ...]] = dict(init_sepsets) if init_sepsets else {}

    remaining: set[int] = set(range(n))
    analysis: dict[int, EliminationInfo] = {}
    dirty: set[int] = set(range(n))

    while remaining:
        order = sorted(remaining, key=lambda v: (len(mb[v]), v))

        chosen: int | None = None
        for v in order:
            if v in dirty or v not in analysis:
                analysis[v] = analyze(v, mb[v], ci)
                dirty.discard(v)
            if analysis[v].removable:
                chosen = v
                break

        if chosen is None:  # theory guarantees a removable variable always exists
            raise CBCDInputError(
                "recursive elimination found no removable variable; the CI test may be "
                "inconsistent with a faithful (M)AG (check Markov boundaries / alpha)."
            )

        info = analysis[chosen]
        for y in info.adjacent:
            adj[chosen, y] = adj[y, chosen] = True
        for t, sep in info.coparents.items():
            sepsets.setdefault(frozenset({chosen, t}), sep)

        remaining.discard(chosen)
        del analysis[chosen]
        old_mb = mb[chosen]
        touched = {v for v in old_mb if v in remaining}
        for v in remaining:
            mb[v].discard(chosen)

        touched |= update_mb(chosen, info, mb, sepsets, remaining, ci)
        dirty |= touched  # boundaries changed -> re-decide removability for these next scan

    return adj, sepsets
