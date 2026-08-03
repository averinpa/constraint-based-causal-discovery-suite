"""Time-series graph types: TimeSeriesDAG, TimeSeriesCPDAG, PartialTimeSeriesCPDAG.

Storage convention: ``endpoints`` is an ``(max_lag + 1, n_vars, n_vars)`` int8
array.

* ``endpoints[0, i, j]`` (lag-0 slice): mark at vertex ``j`` of the
  contemporaneous edge ``{i, j}_t``. The slice is *symmetric* in the sense
  that ``endpoints[0, i, j] == NO_EDGE`` iff ``endpoints[0, j, i] == NO_EDGE``.
* ``endpoints[τ, i, j]`` for τ ≥ 1: mark at vertex ``j`` of the lagged edge
  ``i_{t-τ} → j_t``. Slices for τ ≥ 1 are NOT mirror-symmetric — the cell
  ``endpoints[τ, j, i]`` represents a different edge ``j_{t-τ} → i_t``. The
  past-time mark on a lagged edge is implicitly TAIL (vanilla PCMCI does
  not produce bidirected lagged edges; LPCMCI and friends will extend
  storage when they land).

Validation differs by subclass: ``TimeSeriesDAG`` requires every edge
directed; ``TimeSeriesCPDAG`` permits undirected lag-0 edges.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cbcd.exceptions import CBCDInputError
from cbcd.graph.cpdag import CPDAG
from cbcd.graph.dag import DAG
from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.timeseries.lagged import LaggedVar, lagged_node_id


@dataclass(frozen=True, slots=True)
class LaggedEdge:
    """Read-only view of one time-series edge."""

    src: LaggedVar
    dst: LaggedVar
    mark_at_src: EndpointMark
    mark_at_dst: EndpointMark

    @property
    def lag(self) -> int:
        return self.dst.lag - self.src.lag


class _LaggedGraphBase(ABC):
    """Endpoint-mark int8 array storage for time-series graphs."""

    n_vars: int
    max_lag: int
    endpoints: NDArray[np.int8]
    var_names: tuple[str, ...] | None

    def __init__(
        self,
        n_vars: int,
        max_lag: int,
        endpoints: NDArray[np.int8] | None = None,
        var_names: tuple[str, ...] | None = None,
    ) -> None:
        if n_vars < 0:
            raise CBCDInputError(f"n_vars must be ≥ 0, got {n_vars}")
        if max_lag < 0:
            raise CBCDInputError(f"max_lag must be ≥ 0, got {max_lag}")
        shape = (max_lag + 1, n_vars, n_vars)
        if endpoints is None:
            endpoints = np.zeros(shape, dtype=np.int8)
        else:
            endpoints = np.ascontiguousarray(endpoints, dtype=np.int8)
            if endpoints.shape != shape:
                raise CBCDInputError(
                    f"endpoints shape {endpoints.shape} does not match expected {shape}"
                )
        if var_names is not None and len(var_names) != n_vars:
            raise CBCDInputError(
                f"var_names length {len(var_names)} does not match n_vars={n_vars}"
            )
        # Lag-0 slice: no self-loops, NO_EDGE symmetric.
        if n_vars > 0:
            if np.any(np.diag(endpoints[0]) != EndpointMark.NO_EDGE):
                raise CBCDInputError("self-loops at lag=0 are not allowed")
            no_edge_one_side = (endpoints[0] == EndpointMark.NO_EDGE) ^ (
                endpoints[0].T == EndpointMark.NO_EDGE
            )
            if np.any(no_edge_one_side):
                raise CBCDInputError(
                    "lag-0 endpoint slice is asymmetric in NO_EDGE: "
                    "edge presence must agree on both sides"
                )

        self.n_vars = n_vars
        self.max_lag = max_lag
        self.endpoints = endpoints
        self.var_names = var_names
        self._validate_endpoints()

    @abstractmethod
    def _validate_endpoints(self) -> None: ...

    def has_edge(self, src: LaggedVar, dst: LaggedVar) -> bool:
        tau = dst.lag - src.lag
        if tau < 0 or tau > self.max_lag:
            return False
        return bool(self.endpoints[tau, src.var, dst.var] != EndpointMark.NO_EDGE)

    def lagged_edges(self) -> tuple[LaggedEdge, ...]:
        """All edges with lag (= dst.lag - src.lag) > 0, anchored at dst.lag = 0."""
        out: list[LaggedEdge] = []
        for tau in range(1, self.max_lag + 1):
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if self.endpoints[tau, i, j] != EndpointMark.NO_EDGE:
                        out.append(
                            LaggedEdge(
                                src=LaggedVar(i, -tau),
                                dst=LaggedVar(j, 0),
                                mark_at_src=EndpointMark.TAIL,
                                mark_at_dst=EndpointMark(int(self.endpoints[tau, i, j])),
                            )
                        )
        return tuple(out)

    def contemporaneous_edges(self) -> tuple[LaggedEdge, ...]:
        """All edges at lag = 0, reported once with src.var < dst.var."""
        out: list[LaggedEdge] = []
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                m_at_j = int(self.endpoints[0, i, j])
                m_at_i = int(self.endpoints[0, j, i])
                if m_at_j == EndpointMark.NO_EDGE:
                    continue
                out.append(
                    LaggedEdge(
                        src=LaggedVar(i, 0),
                        dst=LaggedVar(j, 0),
                        mark_at_src=EndpointMark(m_at_i),
                        mark_at_dst=EndpointMark(m_at_j),
                    )
                )
        return tuple(out)

    def parents(self, v: LaggedVar) -> tuple[LaggedVar, ...]:
        """Parents of vertex ``v`` (lag must be 0; stationarity gives the rest).

        Parents are ``LaggedVar`` whose edge to ``v`` carries an arrowhead
        at ``v`` and a tail (or implicit tail for τ ≥ 1) at the source.
        """
        if v.lag != 0:
            raise CBCDInputError(
                f"parents() supports lag=0 targets only (use stationarity to shift); got {v}"
            )
        out: list[LaggedVar] = []
        # Lag-0 contemporaneous parents (DIRECTED i → v).
        for i in range(self.n_vars):
            if i == v.var:
                continue
            if (
                self.endpoints[0, i, v.var] == EndpointMark.ARROW
                and self.endpoints[0, v.var, i] == EndpointMark.TAIL
            ):
                out.append(LaggedVar(i, 0))
        # Lagged parents (τ ≥ 1).
        for tau in range(1, self.max_lag + 1):
            for i in range(self.n_vars):
                if self.endpoints[tau, i, v.var] == EndpointMark.ARROW:
                    out.append(LaggedVar(i, -tau))
        return tuple(out)

    @abstractmethod
    def to_summary_graph(self) -> object: ...

    @abstractmethod
    def to_contemporaneous_graph(self) -> object: ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LaggedGraphBase):
            return NotImplemented
        if type(self) is not type(other):
            return False
        if (self.n_vars, self.max_lag) != (other.n_vars, other.max_lag):
            return False
        return bool(np.array_equal(self.endpoints, other.endpoints))

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.n_vars, self.max_lag, self.endpoints.tobytes()))


def _validate_lagged_dag_marks(endpoints: NDArray[np.int8]) -> None:
    permitted = {
        int(EndpointMark.NO_EDGE),
        int(EndpointMark.TAIL),
        int(EndpointMark.ARROW),
    }
    seen = {int(m) for m in np.unique(endpoints)}
    unsupported = seen - permitted
    if unsupported:
        raise CBCDInputError(
            f"TimeSeriesDAG only supports NO_EDGE/TAIL/ARROW marks; got {sorted(unsupported)}"
        )
    max_lag_plus_one, n, _ = endpoints.shape
    # Lag-0 slice: every present edge must be directed (one TAIL, one ARROW).
    for i in range(n):
        for j in range(i + 1, n):
            mij = int(endpoints[0, i, j])
            mji = int(endpoints[0, j, i])
            if mij == EndpointMark.NO_EDGE:
                continue
            if mij == EndpointMark.ARROW and mji == EndpointMark.TAIL:
                continue
            if mij == EndpointMark.TAIL and mji == EndpointMark.ARROW:
                continue
            raise CBCDInputError(
                f"TimeSeriesDAG lag-0 edge ({i}, {j}) must be directed; got marks=({mji}, {mij})"
            )
    # Lagged slices: every present cell must be ARROW (mark at present).
    for tau in range(1, max_lag_plus_one):
        for i in range(n):
            for j in range(n):
                m = int(endpoints[tau, i, j])
                if m == EndpointMark.NO_EDGE:
                    continue
                if m != EndpointMark.ARROW:
                    raise CBCDInputError(
                        f"TimeSeriesDAG lagged cell (τ={tau}, {i}, {j}) "
                        f"must be ARROW or NO_EDGE; got mark={m}"
                    )


def _validate_lagged_cpdag_marks(endpoints: NDArray[np.int8]) -> None:
    permitted = {
        int(EndpointMark.NO_EDGE),
        int(EndpointMark.TAIL),
        int(EndpointMark.ARROW),
    }
    seen = {int(m) for m in np.unique(endpoints)}
    unsupported = seen - permitted
    if unsupported:
        raise CBCDInputError(
            f"TimeSeriesCPDAG only supports NO_EDGE/TAIL/ARROW marks; got {sorted(unsupported)}"
        )
    max_lag_plus_one, n, _ = endpoints.shape
    # Lag-0 slice: directed or undirected.
    for i in range(n):
        for j in range(i + 1, n):
            mij = int(endpoints[0, i, j])
            mji = int(endpoints[0, j, i])
            if mij == EndpointMark.NO_EDGE:
                continue
            if mij == EndpointMark.ARROW and mji == EndpointMark.TAIL:
                continue
            if mij == EndpointMark.TAIL and mji == EndpointMark.ARROW:
                continue
            if mij == EndpointMark.TAIL and mji == EndpointMark.TAIL:
                continue
            raise CBCDInputError(
                f"TimeSeriesCPDAG lag-0 edge ({i}, {j}) is not directed or undirected; "
                f"got marks=({mji}, {mij})"
            )
    # Lagged slices: ARROW or NO_EDGE.
    for tau in range(1, max_lag_plus_one):
        for i in range(n):
            for j in range(n):
                m = int(endpoints[tau, i, j])
                if m == EndpointMark.NO_EDGE:
                    continue
                if m != EndpointMark.ARROW:
                    raise CBCDInputError(
                        f"TimeSeriesCPDAG lagged cell (τ={tau}, {i}, {j}) "
                        f"must be ARROW or NO_EDGE; got mark={m}"
                    )


class TimeSeriesDAG(_LaggedGraphBase):
    """Directed time-series graph. Contemporaneous (lag-0) edges are directed;
    lagged edges (τ ≥ 1) are always TAIL→ARROW (past→present)."""

    def _validate_endpoints(self) -> None:
        _validate_lagged_dag_marks(self.endpoints)

    def to_summary_graph(self) -> DAG:
        """Project to a DAG over variable indices: edge i → j exists iff there
        is a directed edge between them at any lag."""
        n = self.n_vars
        ep = np.zeros((n, n), dtype=np.int8)
        # Lag-0: copy directed edges directly.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (
                    self.endpoints[0, i, j] == EndpointMark.ARROW
                    and self.endpoints[0, j, i] == EndpointMark.TAIL
                ):
                    ep[i, j] = EndpointMark.ARROW
                    ep[j, i] = EndpointMark.TAIL
        # Lagged: any τ-edge implies i → j in the summary graph.
        for tau in range(1, self.max_lag + 1):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if (
                        self.endpoints[tau, i, j] == EndpointMark.ARROW
                        and ep[i, j] == EndpointMark.NO_EDGE
                    ):
                        ep[i, j] = EndpointMark.ARROW
                        ep[j, i] = EndpointMark.TAIL
        return DAG(n, ep, self.var_names)

    def to_contemporaneous_graph(self) -> DAG:
        return DAG(self.n_vars, self.endpoints[0].copy(), self.var_names)


class TimeSeriesCPDAG(_LaggedGraphBase):
    """Output of PCMCI / PCMCI+. Lagged edges all directed past→present;
    contemporaneous edges may be directed or undirected."""

    def _validate_endpoints(self) -> None:
        _validate_lagged_cpdag_marks(self.endpoints)

    def to_summary_graph(self) -> CPDAG:
        n = self.n_vars
        ep = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (
                    self.endpoints[0, i, j] != EndpointMark.NO_EDGE
                    and self.endpoints[0, j, i] != EndpointMark.NO_EDGE
                ):
                    ep[i, j] = self.endpoints[0, i, j]
                    ep[j, i] = self.endpoints[0, j, i]
        for tau in range(1, self.max_lag + 1):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if (
                        self.endpoints[tau, i, j] == EndpointMark.ARROW
                        and ep[i, j] == EndpointMark.NO_EDGE
                    ):
                        ep[i, j] = EndpointMark.ARROW
                        ep[j, i] = EndpointMark.TAIL
        return CPDAG(n, ep, self.var_names)

    def to_contemporaneous_graph(self) -> CPDAG:
        return CPDAG(self.n_vars, self.endpoints[0].copy(), self.var_names)


@dataclass(frozen=True)
class TimeSeriesPAG:
    """Time-series PAG in *windowed* form — the output type for the latent case (LPCMCI / tsFCI).

    Unlike the compact ``(max_lag+1, n, n)`` TS types, a latent TS-PAG needs BOTH endpoint marks of a
    lagged edge (``o->``, ``<->``, ``o-o`` across time), which a single mark-at-destination cell
    cannot store. So it is held as a full ``PAG`` over the *lagged grid* of ``n_series`` variables at
    lags ``0 .. max_lag``: grid node ``var*(max_lag+1) + (-lag)`` (see ``lagged_node_id``). Reusing
    ``PAG`` gives mark validation for free and lets the orientation phase run ``FCIRules`` on the grid.
    """

    n_series: int
    max_lag: int
    window: PAG
    var_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_series < 0:
            raise CBCDInputError(f"n_series must be >= 0, got {self.n_series}")
        if self.max_lag < 0:
            raise CBCDInputError(f"max_lag must be >= 0, got {self.max_lag}")
        expected = self.n_series * (self.max_lag + 1)
        if self.window.n_vars != expected:
            raise CBCDInputError(
                f"window PAG has {self.window.n_vars} nodes, expected n_series*(max_lag+1)="
                f"{expected}"
            )
        if self.var_names is not None and len(self.var_names) != self.n_series:
            raise CBCDInputError(
                f"var_names length {len(self.var_names)} does not match n_series={self.n_series}"
            )

    @property
    def grid_size(self) -> int:
        return self.n_series * (self.max_lag + 1)

    def node_id(self, var: int, lag: int) -> int:
        """Grid index of the lagged variable ``(var, lag)`` (lag in ``[-max_lag, 0]``)."""
        if not (0 <= var < self.n_series):
            raise CBCDInputError(f"var must be in [0, {self.n_series}); got {var}")
        if not (-self.max_lag <= lag <= 0):
            raise CBCDInputError(f"lag must be in [-{self.max_lag}, 0]; got {lag}")
        return lagged_node_id(LaggedVar(var, lag), self.max_lag)

    def decode(self, node: int) -> LaggedVar:
        """Inverse of ``node_id``: grid index -> ``LaggedVar(var, lag)``."""
        if not (0 <= node < self.grid_size):
            raise CBCDInputError(f"node must be in [0, {self.grid_size}); got {node}")
        var, rem = divmod(int(node), self.max_lag + 1)
        return LaggedVar(var, -rem)

    def edges(self) -> tuple[LaggedEdge, ...]:
        """All edges in the window, each reported once as a ``LaggedEdge`` with both marks, anchored
        so ``src`` is the temporally earlier endpoint (past -> present; tie-broken by variable)."""
        ep = self.window.endpoints
        out: list[LaggedEdge] = []
        for a in range(self.grid_size):
            for b in range(a + 1, self.grid_size):
                if ep[a, b] == EndpointMark.NO_EDGE:
                    continue
                va, vb = self.decode(a), self.decode(b)
                if (va.lag, va.var) <= (vb.lag, vb.var):
                    s_id, d_id, s_v, d_v = a, b, va, vb
                else:
                    s_id, d_id, s_v, d_v = b, a, vb, va
                out.append(
                    LaggedEdge(
                        src=s_v,
                        dst=d_v,
                        mark_at_src=EndpointMark(int(ep[d_id, s_id])),
                        mark_at_dst=EndpointMark(int(ep[s_id, d_id])),
                    )
                )
        return tuple(out)

    def present_edges(self) -> tuple[LaggedEdge, ...]:
        """Present-anchored edges only — the later endpoint at lag 0. This is the canonical,
        non-redundant windowed view (the standard ``graph[i, j, tau]`` present-anchored convention):
        it drops the lag-shifted mirror copies that stationarity makes redundant."""
        return tuple(e for e in self.edges() if e.dst.lag == 0)


class PartialTimeSeriesCPDAG(_LaggedGraphBase):
    """Intermediate state used by PCMCI+ between contemporaneous-collider
    orientation and edge-rule closure. Vanilla PCMCI does not construct one;
    the class exists so PCMCI+ can land cleanly later.
    """

    def _validate_endpoints(self) -> None:
        _validate_lagged_cpdag_marks(self.endpoints)

    def to_summary_graph(self) -> CPDAG:
        return TimeSeriesCPDAG(
            self.n_vars, self.max_lag, self.endpoints, self.var_names
        ).to_summary_graph()

    def to_contemporaneous_graph(self) -> CPDAG:
        return CPDAG(self.n_vars, self.endpoints[0].copy(), self.var_names)
