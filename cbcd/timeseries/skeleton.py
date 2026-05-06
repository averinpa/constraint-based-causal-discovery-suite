"""Lagged skeleton phase: LaggedSkeleton, PC1Skeleton.

PC₁ (Runge 2018, §3.1): per-target candidate-parent pruning. For each target
``Y_t``, iteratively remove candidate lagged parents whose CI test against
``Y_t`` succeeds when conditioned on the strongest currently-surviving
parents. "Strength" is tracked as the smallest p-value seen so far (lower
p = stronger evidence of dependence = higher priority as a conditioner).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cbcd.exceptions import CBCDInputError
from cbcd.timeseries.citest import LaggedCITest
from cbcd.timeseries.lagged import LaggedBackgroundKnowledge, LaggedVar


@dataclass
class LaggedSkeleton:
    """Output of the PC₁ stage: per-target estimated parent sets + sepsets.

    Unlike the i.i.d. ``Skeleton`` (which carries an adjacency matrix), the
    time-series skeleton holds *per-target* parent sets — PCMCI is a
    per-target search, not a global skeleton scan.
    """

    n_vars: int
    max_lag: int
    parents: dict[LaggedVar, frozenset[LaggedVar]]
    sepsets: dict[frozenset[LaggedVar], tuple[LaggedVar, ...]] = field(default_factory=dict)


class LaggedSkeletonAlgorithm(Protocol):
    def __call__(
        self,
        ci: LaggedCITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        background: LaggedBackgroundKnowledge | None = None,
        n_jobs: int = 1,
    ) -> LaggedSkeleton: ...


class PC1Skeleton:
    """PCMCI's PC₁ stage: per-target lagged-parent pruning.

    For each target ``Y_t``, candidate lagged parents are
    ``{(X, -τ): X ∈ vars, τ ∈ [1, max_lag]}`` (including ``X = Y`` for
    autocorrelation). Iteration is by depth ``d``:

    1. Sort survivors by ``pval_max`` ascending (smallest p = strongest
       evidence first).
    2. For each Z, condition on the top-``d`` strongest survivors excluding
       Z. Test ``ci(Z, Y, S)``; update ``pval_max[Z] = max(pval_max[Z], p)``.
    3. If ``p > alpha``, mark Z for removal; record the sepset.
    4. Apply removals, increment depth, repeat until no removals or depth
       exceeds the cap.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        ci: LaggedCITest,
        *,
        alpha: float,
        max_cond_set: int | None = None,
        background: LaggedBackgroundKnowledge | None = None,
        n_jobs: int = 1,
    ) -> LaggedSkeleton:
        if n_jobs != 1:
            raise CBCDInputError("n_jobs != 1 not yet implemented in this slice; pass n_jobs=1")
        n = ci.n_vars
        max_lag = ci.max_lag
        parents: dict[LaggedVar, frozenset[LaggedVar]] = {}
        sepsets: dict[frozenset[LaggedVar], tuple[LaggedVar, ...]] = {}

        for target_var in range(n):
            target = LaggedVar(target_var, 0)
            # Initial candidate parents: all (X, -τ) for τ ∈ [1, max_lag],
            # filtered by background knowledge.
            survivors: list[LaggedVar] = []
            for x_var in range(n):
                for tau in range(1, max_lag + 1):
                    cand = LaggedVar(x_var, -tau)
                    if background is not None and background.is_forbidden_lagged(cand, target):
                        continue
                    survivors.append(cand)

            pval_max: dict[LaggedVar, float] = {z: -float("inf") for z in survivors}

            depth = 0
            while True:
                if max_cond_set is not None and depth > max_cond_set:
                    break
                if not survivors:
                    break
                # Sort by strength: smallest pval_max first (strongest evidence).
                # Ties broken deterministically by (var, -lag) so the result
                # is independent of Python's stable-sort + insertion order.
                survivors.sort(key=lambda z: (pval_max[z], z.var, -z.lag))
                if len(survivors) - 1 < depth:
                    break
                removed: set[LaggedVar] = set()
                for z in list(survivors):
                    if z in removed:
                        continue
                    # Build conditioning set: top-`depth` from survivors
                    # excluding z and any already-removed.
                    pool = [w for w in survivors if w != z and w not in removed]
                    if len(pool) < depth:
                        # Cannot form a size-`depth` conditioning set without z.
                        continue
                    S = tuple(pool[:depth])
                    p = ci(z, target, S)
                    if p > pval_max[z]:
                        pval_max[z] = p
                    if p > alpha:
                        removed.add(z)
                        sepsets[frozenset({z, target})] = S
                if not removed:
                    break
                survivors = [z for z in survivors if z not in removed]
                depth += 1

            parents[target] = frozenset(survivors)

        return LaggedSkeleton(
            n_vars=n,
            max_lag=max_lag,
            parents=parents,
            sepsets=sepsets,
        )
