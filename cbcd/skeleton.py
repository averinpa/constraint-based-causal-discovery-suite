"""Skeleton phase: SkeletonAlgorithm Protocol + PCStable."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cbcd.background import BackgroundKnowledge
from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError


@dataclass
class Skeleton:
    """Output of the skeleton-discovery phase.

    ``adj`` is a symmetric boolean adjacency matrix (``True`` = edge present).
    ``sepsets`` maps each removed pair (as a frozenset) to one witness
    conditioning set that rendered them independent. ``pvalues_max`` holds the
    largest p-value seen for each pair during search, used by ``MaxPOrienter``;
    ``None`` when ``track_max_pvalue=False``.
    """

    n_vars: int
    adj: NDArray[np.bool_]
    sepsets: dict[frozenset[int], tuple[int, ...]] = field(default_factory=dict)
    pvalues_max: NDArray[np.float64] | None = None


class SkeletonAlgorithm(Protocol):
    def __call__(
        self,
        ci: CITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        background: BackgroundKnowledge | None = None,
        n_jobs: int = 1,
    ) -> Skeleton: ...


class PCStable:
    """PC-stable skeleton (Colombo & Maathuis 2014).

    At each conditioning-set size ``d``, neighbour sets are frozen at the start
    of the depth so that all pairs see the same adjacency snapshot — removing
    the order-dependence of the original PC algorithm.
    """

    def __init__(self, *, track_max_pvalue: bool = False) -> None:
        self.track_max_pvalue = track_max_pvalue

    def __call__(
        self,
        ci: CITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        background: BackgroundKnowledge | None = None,
        n_jobs: int = 1,
    ) -> Skeleton:
        if n_jobs != 1:
            raise CBCDInputError(
                "n_jobs != 1 not yet implemented in this slice; pass n_jobs=1"
            )
        n = ci.n_vars
        adj = np.ones((n, n), dtype=bool)
        np.fill_diagonal(adj, False)
        if background is not None:
            for u in range(n):
                for v in range(u + 1, n):
                    if background.is_forbidden_adjacent(u, v):
                        adj[u, v] = False
                        adj[v, u] = False

        sepsets: dict[frozenset[int], tuple[int, ...]] = {}
        pvalues_max: NDArray[np.float64] | None = None
        if self.track_max_pvalue:
            pvalues_max = np.full((n, n), -np.inf, dtype=np.float64)

        depth = 0
        while True:
            if max_cond_set is not None and depth > max_cond_set:
                break

            adj_snapshot = adj.copy()
            to_remove: list[tuple[int, int, tuple[int, ...]]] = []
            any_eligible = False

            marked: set[frozenset[int]] = set()
            for x in range(n):
                neighbours_x_snapshot = np.where(adj_snapshot[x])[0]
                if len(neighbours_x_snapshot) - 1 < depth:
                    continue
                for y in neighbours_x_snapshot:
                    y_int = int(y)
                    if y_int == x:
                        continue
                    if not adj[x, y_int]:
                        continue
                    pair = frozenset({x, y_int})
                    if pair in marked:
                        continue
                    candidates = [int(z) for z in neighbours_x_snapshot if z != y_int]
                    if len(candidates) < depth:
                        continue
                    any_eligible = True

                    found_sepset: tuple[int, ...] | None = None
                    for S in _conditioning_sets(candidates, depth):
                        p = ci(x, y_int, S)
                        if pvalues_max is not None and p > pvalues_max[x, y_int]:
                            pvalues_max[x, y_int] = p
                            pvalues_max[y_int, x] = p
                        if p > alpha:
                            found_sepset = S
                            break
                    if found_sepset is not None:
                        to_remove.append((x, y_int, found_sepset))
                        marked.add(pair)

            for x, y, sep in to_remove:
                if not adj[x, y]:
                    continue
                adj[x, y] = False
                adj[y, x] = False
                sepsets[frozenset({x, y})] = sep

            if not any_eligible:
                break
            depth += 1

        return Skeleton(
            n_vars=n,
            adj=adj,
            sepsets=sepsets,
            pvalues_max=pvalues_max,
        )


def _conditioning_sets(candidates: Sequence[int], k: int) -> list[tuple[int, ...]]:
    if k == 0:
        return [()]
    return [tuple(c) for c in combinations(candidates, k)]
