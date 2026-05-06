"""PAG skeleton-refinement step.

After collider orientation produces a ``PartialPAG``, FCI runs a Possible-
D-Sep refinement pass: for each surviving edge, search subsets of
Possible-D-Sep up to ``max_cond_set`` for a separating set; remove the edge
on first independence found.

This replaces the exponential ``removeByPossibleDsep`` in ``causal-learn``
(audited at FCI.py:1000–1058): increasing-size enumeration with early
break, single ``(X, Y)`` pass instead of two, and a clean parameter for
``max_cond_set``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np

from cbcd.citest.protocol import CITest
from cbcd.exceptions import CBCDInputError
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PartialPAG
from cbcd.graph.queries import possible_dsep


class PAGSkeletonRefinement(Protocol):
    def __call__(
        self,
        graph: PartialPAG,
        ci: CITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        n_jobs: int = 1,
    ) -> PartialPAG: ...


class PossibleDSepRefinement:
    """Standard FCI Possible-D-Sep pruning.

    For each surviving edge ``{X, Y}`` in the input ``PartialPAG``, computes
    Possible-D-Sep(X, Y) on the *current* graph, then enumerates subsets in
    increasing size up to ``max_cond_set``, breaking on the first ``S`` with
    ``X ⫫ Y | S``. Records the witness in ``PartialPAG.sepsets``.
    """

    def __init__(self, *, max_cond_set: int | None = None) -> None:
        self.max_cond_set = max_cond_set

    def __call__(
        self,
        graph: PartialPAG,
        ci: CITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        n_jobs: int = 1,
    ) -> PartialPAG:
        if n_jobs != 1:
            raise CBCDInputError("n_jobs != 1 not yet implemented in this slice; pass n_jobs=1")
        cap = max_cond_set if max_cond_set is not None else self.max_cond_set
        n = graph.n_vars
        endpoints = graph.endpoints.copy()
        sepsets: dict[frozenset[int], tuple[int, ...]] = (
            dict(graph.sepsets) if graph.sepsets else {}
        )

        # Snapshot the edge list so we test each edge once against the input
        # graph; later removals don't disturb iteration order.
        edges_to_check: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if endpoints[i, j] != EndpointMark.NO_EDGE:
                    edges_to_check.append((i, j))

        for x, y in edges_to_check:
            if endpoints[x, y] == EndpointMark.NO_EDGE:
                continue  # Already removed by an earlier iteration's effect.
            pds_x = possible_dsep(endpoints, x, y) - {y}
            pds_y = possible_dsep(endpoints, y, x) - {x}
            candidates = sorted(pds_x | pds_y)
            if not candidates:
                continue
            upper = len(candidates) if cap is None else min(len(candidates), cap)
            removed = False
            for size in range(0, upper + 1):
                for S in combinations(candidates, size):
                    p = ci(x, y, list(S))
                    if p > alpha:
                        endpoints[x, y] = EndpointMark.NO_EDGE
                        endpoints[y, x] = EndpointMark.NO_EDGE
                        sepsets[frozenset({x, y})] = tuple(S)
                        removed = True
                        break
                if removed:
                    break

        # Wipe orientations on remaining edges back to CIRCLE-CIRCLE: refinement
        # invalidates the prior collider pass because removed edges may change
        # which triples are unshielded. The caller (``fci()``) re-runs the
        # collider step on the refined skeleton.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if endpoints[i, j] != EndpointMark.NO_EDGE:
                    endpoints[i, j] = EndpointMark.CIRCLE
        np.fill_diagonal(endpoints, EndpointMark.NO_EDGE)

        return PartialPAG(
            n_vars=n,
            endpoints=endpoints,
            var_names=graph.var_names,
            sepsets=sepsets,
        )
