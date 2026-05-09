"""Lagged primitives: LaggedVar, LaggedDataset, LaggedBackgroundKnowledge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError


@dataclass(frozen=True, slots=True)
class LaggedVar:
    """A variable at a specific lag.

    ``lag`` is non-positive: ``0`` means "current time t", ``-1`` is t-1, etc.
    Algorithms enforce ``-max_lag ≤ lag ≤ 0`` against the dataset; the class
    only enforces ``lag ≤ 0`` at construction.
    """

    var: int
    lag: int

    def __post_init__(self) -> None:
        if self.lag > 0:
            raise CBCDInputError(f"LaggedVar.lag must be ≤ 0 (past or current), got {self.lag}")
        if self.var < 0:
            raise CBCDInputError(f"LaggedVar.var must be ≥ 0, got {self.var}")

    @property
    def is_contemporaneous(self) -> bool:
        return self.lag == 0


@dataclass
class LaggedDataset:
    """Multivariate stationary time series with a max-lag horizon.

    Stationarity is assumed: algorithm parameters apply across all ``t``
    (decision D9). Use regime-switching variants for non-stationary data
    (out of scope for v0).
    """

    data: NDArray[np.float64]
    max_lag: int
    var_names: tuple[str, ...] | None = None
    time_index: NDArray[np.int_] | NDArray[np.datetime64] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise CBCDInputError(
                f"LaggedDataset.data must be 2-D (T, n_vars); got shape {self.data.shape}"
            )
        if self.max_lag < 0:
            raise CBCDInputError(f"max_lag must be ≥ 0, got {self.max_lag}")
        T, _ = self.data.shape
        if self.max_lag >= T - 1:
            raise CBCDInputError(
                f"max_lag={self.max_lag} too large for series length T={T}; "
                f"need max_lag < T - 1 for a usable training window"
            )
        if self.var_names is not None and len(self.var_names) != self.data.shape[1]:
            raise CBCDInputError(
                f"var_names length {len(self.var_names)} does not match "
                f"data columns {self.data.shape[1]}"
            )

    @property
    def n_vars(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])


def _has_lagged_cycle(n_vars: int, edges: frozenset[tuple[LaggedVar, LaggedVar]]) -> bool:
    """Detect a directed cycle in the *contemporaneous* projection of the edge
    set. Lagged edges (src.lag < dst.lag) cannot form cycles because of the
    time direction; only the lag=0 sub-edges are at risk.
    """
    contemporaneous: list[tuple[int, int]] = [
        (s.var, d.var) for s, d in edges if s.lag == 0 and d.lag == 0 and s.var != d.var
    ]
    children: dict[int, list[int]] = {i: [] for i in range(n_vars)}
    indeg = [0] * n_vars
    for u, v in contemporaneous:
        children[u].append(v)
        indeg[v] += 1
    stack = [i for i in range(n_vars) if indeg[i] == 0]
    visited = 0
    while stack:
        u = stack.pop()
        visited += 1
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    referenced = {u for u, _ in contemporaneous} | {v for _, v in contemporaneous}
    return visited != len(referenced | set(range(n_vars)))


@dataclass(frozen=True)
class LaggedBackgroundKnowledge:
    """Time-series-specific background knowledge.

    Constraints are over ``LaggedVar`` tuples, not bare variable indices —
    keeping types honest separates this from the i.i.d.
    ``BackgroundKnowledge`` (decision D5: fail-fast at construction).
    """

    forbidden_lagged: frozenset[tuple[LaggedVar, LaggedVar]] = field(default_factory=frozenset)
    """Directed lagged edges that must not appear: (src, dst).
    ``src.lag <= dst.lag`` is required for stationary edges."""

    required_lagged: frozenset[tuple[LaggedVar, LaggedVar]] = field(default_factory=frozenset)
    """Directed lagged edges that must appear."""

    no_autoregressive: frozenset[int] = field(default_factory=frozenset)
    """Variables for which no ``X_{t-τ} → X_t`` edge may appear at any τ > 0."""

    no_contemporaneous: frozenset[frozenset[int]] = field(default_factory=frozenset)
    """Pairs ``{i, j}`` with no contemporaneous (lag = 0) edge."""

    contemporaneous_tiers: tuple[frozenset[int], ...] = ()
    """Tiers for lag = 0 edges: tier-i variables cannot be contemporaneous
    ancestors of tier-j vars when j < i. Lagged edges always respect time."""

    def __post_init__(self) -> None:
        forbidden = frozenset(self.forbidden_lagged)
        required = frozenset(self.required_lagged)
        no_auto = frozenset(self.no_autoregressive)
        no_contemp = frozenset(frozenset(p) for p in self.no_contemporaneous)
        tiers = tuple(frozenset(t) for t in self.contemporaneous_tiers)
        object.__setattr__(self, "forbidden_lagged", forbidden)
        object.__setattr__(self, "required_lagged", required)
        object.__setattr__(self, "no_autoregressive", no_auto)
        object.__setattr__(self, "no_contemporaneous", no_contemp)
        object.__setattr__(self, "contemporaneous_tiers", tiers)

        for s, d in required | forbidden:
            if s.lag > d.lag:
                raise CBCDInputError(
                    f"lagged edge ({s}, {d}) violates time direction: src.lag must be ≤ dst.lag"
                )
            if s == d:
                raise CBCDInputError(f"self-loop in lagged edge ({s}, {d})")

        if required & forbidden:
            offending = required & forbidden
            raise CBCDInputError(
                f"required_lagged and forbidden_lagged overlap: {sorted(offending)}"
            )

        for s, d in required:
            if s.lag == 0 and d.lag == 0 and frozenset({s.var, d.var}) in no_contemp:
                raise CBCDInputError(
                    f"required contemporaneous edge {(s, d)} contradicts "
                    f"no_contemporaneous {{{s.var}, {d.var}}}"
                )
            if s.var == d.var and s.lag < d.lag and s.var in no_auto:
                raise CBCDInputError(
                    f"required autoregressive edge {(s, d)} contradicts "
                    f"no_autoregressive on var {s.var}"
                )

        if required:
            n_implied = max(max(s.var, d.var) for s, d in required) + 1
            if _has_lagged_cycle(n_implied, required):
                raise CBCDInputError("required_lagged contains a contemporaneous directed cycle")

        if tiers:
            tier_of: dict[int, int] = {}
            for level, members in enumerate(tiers):
                for v in members:
                    if v in tier_of:
                        raise CBCDInputError(
                            f"variable {v} appears in multiple contemporaneous "
                            f"tiers ({tier_of[v]}, {level})"
                        )
                    tier_of[v] = level
            for s, d in required:
                if (
                    s.lag == 0
                    and d.lag == 0
                    and s.var in tier_of
                    and d.var in tier_of
                    and tier_of[s.var] > tier_of[d.var]
                ):
                    raise CBCDInputError(
                        f"required contemporaneous edge {(s, d)} violates "
                        f"tier ordering ({tier_of[s.var]} > {tier_of[d.var]})"
                    )

    def is_forbidden_lagged(self, src: LaggedVar, dst: LaggedVar) -> bool:
        if (src, dst) in self.forbidden_lagged:
            return True
        if src.var == dst.var and src.lag < dst.lag and src.var in self.no_autoregressive:
            return True
        if src.lag == 0 and dst.lag == 0:
            if frozenset({src.var, dst.var}) in self.no_contemporaneous:
                return True
            for level, members in enumerate(self.contemporaneous_tiers):
                if src.var in members:
                    src_tier = level
                    break
            else:
                src_tier = -1
            for level, members in enumerate(self.contemporaneous_tiers):
                if dst.var in members:
                    dst_tier = level
                    break
            else:
                dst_tier = -1
            if src_tier >= 0 and dst_tier >= 0 and src_tier > dst_tier:
                return True
        return False

    def is_required_lagged(self, src: LaggedVar, dst: LaggedVar) -> bool:
        return (src, dst) in self.required_lagged
