"""LPCMCI-PAG: the working graph type for faithful LPCMCI (Gerhardus & Runge, NeurIPS 2020).

Clean-room from the authors' published math (SM §3.2, §S4). An LPCMCI-PAG is a windowed PAG over the
stationary lagged grid whose edges additionally carry a **middle mark** ``{empty, ?, L, R, !}`` that
records intermediate (non-)ancestor knowledge so orientations can be applied early and iteratively.

**Endpoint marks** reuse cbcd's :class:`EndpointMark` values — ``TAIL`` (1, ancestor), ``HEAD`` (2,
non-ancestor arrowhead; ``EndpointMark.ARROW``), ``CIRCLE`` (3, unknown) — plus a native
``CONFLICT`` (4) mark for an endpoint an orientation rule tried to set both ways (SM §S4).

**Homology / stationarity.** Every edge is a time-shift equivalence class. We store the graph keyed by
canonical edges ``(i, j, τ)`` meaning ``X^i_{t-τ} — X^j_t`` (``τ ∈ 0..τ_max``; for ``τ = 0`` we keep
``i < j``). ``_canonical`` maps any grid-node pair to its canonical key by shifting the later endpoint
to lag 0, so every read/write automatically hits all homologous copies — this single mechanism is what
enforces stationarity and order-independence (Thm 3).

Grid encoding (shared with the rest of ``cbcd.timeseries``): grid node ``v*(τ_max+1) + (-lag)`` for
``lag ∈ [-τ_max, 0]``; present slice = lag 0.
"""

from __future__ import annotations

import numpy as np

from cbcd.graph.marks import EndpointMark
from cbcd.graph.pag import PAG
from cbcd.timeseries.graph import TimeSeriesPAG

# --- endpoint marks (cbcd EndpointMark values + native CONFLICT) --------------------------------
NO_EDGE = int(EndpointMark.NO_EDGE)  # 0
TAIL = int(EndpointMark.TAIL)  # 1
HEAD = int(EndpointMark.ARROW)  # 2
CIRCLE = int(EndpointMark.CIRCLE)  # 3
CONFLICT = 4  # native LPCMCI extension: a rule proposed both tail and head at this endpoint

# --- middle marks -------------------------------------------------------------------------------
MM_EMPTY = 0
MM_Q = 1  # "?"  — promises nothing
MM_L = 2  # "L"
MM_R = 3  # "R"
MM_BANG = 4  # "!" — both L and R statements hold

_MM_SYMBOL = {MM_EMPTY: "", MM_Q: "?", MM_L: "L", MM_R: "R", MM_BANG: "!"}
_MARK_SYMBOL = {NO_EDGE: "", TAIL: "-", HEAD: ">", CIRCLE: "o", CONFLICT: "x"}


def grid_id(var: int, lag: int, max_lag: int) -> int:
    """Grid-node index of ``X^var_{t+lag}`` (``lag ∈ [-max_lag, 0]``)."""
    return var * (max_lag + 1) + (-lag)


def decode_grid(node: int, max_lag: int) -> tuple[int, int]:
    """Inverse of :func:`grid_id`: grid node -> ``(var, lag)`` with ``lag <= 0``."""
    var, rem = divmod(int(node), max_lag + 1)
    return var, -rem


def mm_combine(current: int, add: int) -> int:
    """Lemma S8 symbolic middle-mark algebra: ``? + * = *``; ``* + empty = empty``; ``L + R = !``.

    (``?`` is consistent with anything; ``empty`` implies every other mark; ``L`` and ``R`` together
    give ``!``.) Used by MMR and the Alg-S2/S3 middle-mark updates.
    """
    if current == MM_EMPTY or add == MM_EMPTY:
        return MM_EMPTY
    if current == MM_Q:
        return add
    if add == MM_Q:
        return current
    if current == add:
        return current
    if {current, add} == {MM_L, MM_R}:
        return MM_BANG
    # combining with ! (or L+! / R+!) stays ! (! already means both hold)
    return MM_BANG


class LPCMCIPAG:
    """Windowed LPCMCI-PAG over the lagged grid with endpoint + middle marks and homology.

    All accessors take **grid-node ids**; storage is canonical so writes propagate to every
    time-shifted copy. ``mark(a, b)`` is the endpoint mark *at a* on edge ``{a, b}``.
    """

    # Precomputed per (n_series, max_lag): flat canonical index + a_is_i + decode + order tables.
    _CACHE: dict[tuple[int, int], tuple] = {}

    def __init__(self, n_series: int, max_lag: int) -> None:
        self.n = n_series
        self.max_lag = max_lag
        self.grid_n = gn = n_series * (max_lag + 1)
        self._size = (max_lag + 1) * n_series * n_series
        # Flat canonical storage: index = τ*n*n + i*n + j.
        self._mi = np.full(self._size, NO_EDGE, dtype=np.int8)
        self._mj = np.full(self._size, NO_EDGE, dtype=np.int8)
        self._mm = np.full(self._size, MM_EMPTY, dtype=np.int8)
        # Precompute (and share) the canonical lookup tables — grid is small (≤ ~n·(τ+1)).
        cache = LPCMCIPAG._CACHE.get((n_series, max_lag))
        if cache is None:
            lin = [[0] * gn for _ in range(gn)]
            isi = [[False] * gn for _ in range(gn)]
            before = [[False] * gn for _ in range(gn)]
            dv = [0] * gn
            dl = [0] * gn
            nn = n_series
            for a in range(gn):
                va, la = decode_grid(a, max_lag)
                dv[a], dl[a] = va, la
                for b in range(gn):
                    vb, lb = decode_grid(b, max_lag)
                    if la == lb:
                        (tau, i, j, ai) = (0, va, vb, True) if va < vb else (0, vb, va, False)
                    elif la > lb:
                        tau, i, j, ai = la - lb, vb, va, False
                    else:
                        tau, i, j, ai = lb - la, va, vb, True
                    lin[a][b] = tau * nn * nn + i * nn + j
                    isi[a][b] = ai
                    before[a][b] = (la, va) < (lb, vb)
            cache = (lin, isi, before, dv, dl)
            LPCMCIPAG._CACHE[(n_series, max_lag)] = cache
        self._lin, self._isi, self._before_t, self._dv, self._dl = cache

    def _canonical(self, a: int, b: int) -> tuple[int, int, int, bool]:
        """Canonical ``(τ, i, j, a_is_i)`` for grid pair ``(a, b)`` (from the precomputed table)."""
        k = self._lin[a][b]
        nn = self.n
        tau, rem = divmod(k, nn * nn)
        i, j = divmod(rem, nn)
        return tau, i, j, self._isi[a][b]

    # --- edge / mark accessors ------------------------------------------------------------------
    def edge_exists(self, a: int, b: int) -> bool:
        if a == b:
            return False
        return bool(self._mi[self._lin[a][b]] != NO_EDGE)

    def mark(self, a: int, b: int) -> int:
        """Endpoint mark at ``a`` on edge ``{a, b}`` (``NO_EDGE`` if no edge)."""
        k = self._lin[a][b]
        return int(self._mi[k] if self._isi[a][b] else self._mj[k])

    def set_mark(self, a: int, b: int, value: int) -> None:
        """Set the endpoint mark at ``a`` on edge ``{a, b}`` (propagates to all homologous copies)."""
        k = self._lin[a][b]
        if self._isi[a][b]:
            self._mi[k] = value
        else:
            self._mj[k] = value

    def middle(self, a: int, b: int) -> int:
        return int(self._mm[self._lin[a][b]])

    def set_middle(self, a: int, b: int, value: int) -> None:
        self._mm[self._lin[a][b]] = value

    def combine_middle(self, a: int, b: int, add: int) -> None:
        """Update the middle mark of ``{a, b}`` by the Lemma S8 algebra (``current ⊕ add``)."""
        self.set_middle(a, b, mm_combine(self.middle(a, b), add))

    def add_edge(self, a: int, b: int, mark_a: int, mark_b: int, middle: int) -> None:
        """Create/overwrite edge ``{a, b}`` with the given endpoint marks and middle mark."""
        k = self._lin[a][b]
        if self._isi[a][b]:
            self._mi[k], self._mj[k] = mark_a, mark_b
        else:
            self._mi[k], self._mj[k] = mark_b, mark_a
        self._mm[k] = middle

    def remove_edge(self, a: int, b: int) -> None:
        k = self._lin[a][b]
        self._mi[k] = self._mj[k] = NO_EDGE
        self._mm[k] = MM_EMPTY

    # --- endpoint-mark predicates ---------------------------------------------------------------
    def is_head(self, a: int, b: int) -> bool:
        return self.mark(a, b) == HEAD

    def is_tail(self, a: int, b: int) -> bool:
        return self.mark(a, b) == TAIL

    def is_circle(self, a: int, b: int) -> bool:
        return self.mark(a, b) == CIRCLE

    # --- adjacency / order ----------------------------------------------------------------------
    def neighbors(self, a: int) -> list[int]:
        """All grid nodes adjacent to ``a`` (homology-respecting)."""
        return [b for b in range(self.grid_n) if b != a and self.edge_exists(a, b)]

    def parents(self, a: int) -> list[int]:
        """Definite parents of ``a``: neighbours ``b`` with ``b --> a`` (tail at ``b``, head at ``a``)."""
        return [
            b
            for b in self.neighbors(a)
            if self.mark(b, a) == TAIL and self.mark(a, b) == HEAD
        ]

    def before(self, a: int, b: int) -> bool:
        """Total (time) order: ``X^a < X^b`` iff ``a`` is temporally earlier, ties broken by index."""
        return self._before_t[a][b]

    def present_nodes(self) -> list[int]:
        return [grid_id(v, 0, self.max_lag) for v in range(self.n)]

    def has_nonempty_middle(self) -> bool:
        """True if any edge carries a middle mark other than ``!`` or empty (Alg-S2 stop test)."""
        return bool(np.any((self._mm != MM_EMPTY) & (self._mm != MM_BANG) & (self._mi != NO_EDGE)))

    def has_nonempty_middle_incl_bang(self) -> bool:
        """True if any edge carries a non-empty middle mark (Alg-S3 stop test)."""
        return bool(np.any((self._mm != MM_EMPTY) & (self._mi != NO_EDGE)))

    def copy(self) -> LPCMCIPAG:
        g = LPCMCIPAG(self.n, self.max_lag)
        g._mi = self._mi.copy()
        g._mj = self._mj.copy()
        g._mm = self._mm.copy()
        return g

    # --- conversion -----------------------------------------------------------------------------
    def to_timeseries_pag(self, var_names: tuple[str, ...] | None = None) -> TimeSeriesPAG:
        """Strip middle marks and export an ordinary windowed :class:`TimeSeriesPAG`.

        Every grid-node pair is filled from its canonical edge so all homologous copies appear; any
        residual ``CONFLICT`` endpoint (should not occur at convergence) is reported as ``CIRCLE``.
        """
        ep = np.full((self.grid_n, self.grid_n), NO_EDGE, dtype=np.int8)
        for a in range(self.grid_n):
            for b in range(a + 1, self.grid_n):
                if not self.edge_exists(a, b):
                    continue
                ma, mb = self.mark(a, b), self.mark(b, a)
                ep[b, a] = CIRCLE if ma == CONFLICT else ma  # mark at a
                ep[a, b] = CIRCLE if mb == CONFLICT else mb  # mark at b
        return TimeSeriesPAG(
            n_series=self.n,
            max_lag=self.max_lag,
            window=PAG(self.grid_n, ep),
            var_names=var_names,
        )

    def edge_str(self, a: int, b: int) -> str:
        """Human-readable edge, e.g. ``o-?->`` — for debugging / test messages."""
        if not self.edge_exists(a, b):
            return "(no edge)"
        return (
            f"{_MARK_SYMBOL[self.mark(a, b)]}-{_MM_SYMBOL[self.middle(a, b)]}-"
            f"{_MARK_SYMBOL[self.mark(b, a)]}"
        )


def complete_lpcmci_pag(n_series: int, max_lag: int) -> LPCMCIPAG:
    """Algorithm-1 line-1 initialisation: the complete LPCMCI-PAG.

    Lagged links (``0 < τ ≤ τ_max``) init as ``X^i_{t-τ} -L-> X^j_t`` (tail at past, head at present,
    middle ``L``); contemporaneous links (``τ = 0``) init as ``X^i_t o-?-o X^j_t`` (circles both ends,
    middle ``?``).
    """
    g = LPCMCIPAG(n_series, max_lag)
    for j in range(n_series):
        pj = grid_id(j, 0, max_lag)
        for i in range(n_series):
            # contemporaneous
            if i < j:
                g.add_edge(grid_id(i, 0, max_lag), pj, CIRCLE, CIRCLE, MM_Q)
            # lagged X^i_{t-τ} -> X^j_t
            for tau in range(1, max_lag + 1):
                past = grid_id(i, -tau, max_lag)
                g.add_edge(past, pj, TAIL, HEAD, MM_L)
    return g


def _homologous_pairs(a: int, b: int, max_lag: int, n_series: int) -> list[tuple[int, int]]:
    """All in-window grid pairs that are time-shifted copies of edge ``{a, b}`` (test helper)."""
    va, la = decode_grid(a, max_lag)
    vb, lb = decode_grid(b, max_lag)
    out: list[tuple[int, int]] = []
    for shift in range(-max_lag, max_lag + 1):
        la2, lb2 = la + shift, lb + shift
        if -max_lag <= la2 <= 0 and -max_lag <= lb2 <= 0:
            out.append((grid_id(va, la2, max_lag), grid_id(vb, lb2, max_lag)))
    return out
