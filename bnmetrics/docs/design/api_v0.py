"""
bnmetrics v0.2 API sketch — pressure-test before implementing.

This is a *design document expressed as Python stubs*. Nothing here runs.
The goal is to fix the contracts (Protocols, dataclasses, function
signatures) before any metric or viz code is written, so that adding a
new comparison metric or supporting a new graph type doesn't require
rewriting the foundation.

DESIGN PRINCIPLES (consequences of the bnmetrics 0.1.x audit):

1. **Pure functions over a god-class.** 0.1.x routes everything through
   `BNMetrics(g1, g2).compile_*` methods. v0.2 exposes top-level
   functions: `bnmetrics.shd(g1, g2)`, `bnmetrics.sid(true, est)`,
   `bnmetrics.markov_blanket(g, var)`. A single `compare()` function returns
   a `Comparison` dataclass for the multi-metric workflow.

2. **One graph contract: `GraphLike`.** A structural Protocol over
   (`n_vars: int`, `endpoints: NDArray[np.int8]`, `var_names: tuple |
   None`). cbcd's `DAG`/`CPDAG`/`PAG` instances conform with zero
   adaptation; bnmetrics itself imports nothing from cbcd. The
   `_to_endpoints` adapter accepts everything else (nx.DiGraph,
   ndarray, list-of-lists) and produces the canonical int8 matrix.

3. **No third edge dialect.** 0.1.x silently collapses `A↔B` (both
   `A→B` and `B→A` present in nx.DiGraph) to `type='undirected'`,
   conflating bidirected and undirected. v0.2's nx adapter REJECTS
   bidirected input. Bidirected edges live in the int8 matrix as
   ARROW/ARROW, and PAG callers route through cbcd's `PAG` (which
   conforms to GraphLike) or a raw int8 matrix.

4. **Soft dep on pandas; soft dep on viz libs.** The metric layer
   imports only numpy. `to_dataframe()` is a free function that
   lazy-imports pandas. Viz lives under `bnmetrics/viz/` and is gated on
   the `viz` extra (graphviz, plotly).

5. **All or nothing on input validation.** Functions raise
   `BNMInputError` on structurally invalid input (caller bug) and
   `BNMDataError` on data-shape mismatches. They never silently
   `print()` and `return None`.

OPEN QUESTIONS are marked OPEN: throughout. Resolve before implementing
the affected slice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# A. GRAPH-LIKE PROTOCOL
# =============================================================================
#
# The minimal structural contract every graph passed into bnmetrics must satisfy.
# Conformance is duck-typed: any object that exposes
#     - n_vars: int  (attribute)
#     - endpoints: NDArray[np.int8]  (attribute, NOT a property method)
#     - var_names: tuple[str, ...] | None  (attribute, may be None)
# satisfies the Protocol. No inheritance, no import of bnmetrics required.
#
# This is the load-bearing design choice. cbcd's `DAG`, `CPDAG`, `PAG`
# (and similarly any user-defined wrapper) plug into bnmetrics metrics with
# zero conversion overhead; cbcd has no idea bnmetrics exists, and vice
# versa. Same holds for any future cbcd graph types (`MAG`, MAG-PAG,
# time-series CPDAG via per-pair contraction).
#
# WHY ATTRIBUTE, NOT PROPERTY: a lazy-computed `endpoints` property
# would still pass `isinstance(obj, GraphLike)`, but downstream metric
# code accesses `g.endpoints` repeatedly inside hot loops; a property
# that recomputes (or returns a fresh ndarray) would silently kill
# performance and could trip array-contiguity checks in numpy
# operations. Lock the contract to "attribute, not method."


@runtime_checkable
class GraphLike(Protocol):
    """A graph backed by a square int8 endpoint-mark matrix.

    `endpoints[i, j]` is the mark at node `j` of the edge between `i`
    and `j`. The pair `(endpoints[j, i], endpoints[i, j])` together
    specify the edge type. NO_EDGE on either end means no edge exists;
    both ends must be NO_EDGE or both non-NO_EDGE. Invariants are
    enforced by the producer of the graph (cbcd, the user, or
    `_to_endpoints`), not by bnmetrics metric functions.
    """

    n_vars: int
    """Number of variables (rows/columns of `endpoints`)."""

    endpoints: NDArray[np.int8]
    """Square (n_vars, n_vars) endpoint-mark matrix."""

    var_names: tuple[str, ...] | None
    """Optional human-readable variable names; None means index-only."""


class EndpointMark(IntEnum):
    """Mark on one end of an edge. Numeric values match cbcd's
    `EndpointMark` so cbcd graphs and bnmetrics graphs are interchangeable
    via the int8 endpoint matrix without import-coupling.

    NO_EDGE on either end means no edge exists.
    """

    NO_EDGE = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3


# OPEN: should we expose a small `Edge` accessor dataclass like cbcd's
# `cbcd.graph.Edge` for callers that want a typed edge view? Lean NO —
# bnmetrics operates on whole graphs, not on individual edges. Decide at
# Slice 4 (viz) when per-edge highlighting needs a typed handle.


# =============================================================================
# B. EXCEPTIONS
# =============================================================================


class BNMError(Exception):
    """Base class for all bnmetrics errors."""


class BNMInputError(BNMError, ValueError):
    """Caller-side bug: invalid argument shape, type, or combination.

    Examples:
      - passing a non-square endpoint matrix
      - passing a `var_name` that isn't in `var_names`
      - asking for SID with a g1 that has undirected edges
      - passing nx.DiGraph with `type='bidirected'` to `_to_endpoints`
    """


class BNMDataError(BNMError, ValueError):
    """Structurally valid input that violates a metric's preconditions.

    Examples:
      - SID g1 contains a directed cycle
      - comparative metric called on graphs with different `n_vars`
    """


# =============================================================================
# C. ADAPTER (`_to_endpoints`)
# =============================================================================
#
# Every public bnmetrics function accepts `Graph | nx.DiGraph | NDArray |
# list[list]`; `_to_endpoints` is the single adapter that normalises to
# the canonical (n_vars, endpoints, var_names) triple.
#
# Mutation policy: the adapter is PURE. It never modifies caller-owned
# inputs, even when the input is mutable (nx.DiGraph). This fixes
# 0.1.x's silent in-place mutation behaviour where
# `mark_and_collapse_bidirected_edges` added `type` attrs to caller
# graphs.


# Type alias for "anything bnmetrics accepts as a graph" — used in the
# signatures below.
GraphLikeInput = Union[
    "GraphLike",
    "NDArray[Any]",
    list[list[int]],
    "Any",  # nx.DiGraph (avoid hard import of networkx in the design doc)
]


def _to_endpoints(
    obj: GraphLikeInput,
    *,
    var_names: tuple[str, ...] | None = None,
) -> tuple[int, NDArray[np.int8], tuple[str, ...] | None]:
    """Normalise any accepted input form to `(n_vars, endpoints, var_names)`.

    Accepted input forms:

    1. **GraphLike** (cbcd `DAG`/`CPDAG`/`PAG`, any user wrapper) —
       returns `(obj.n_vars, np.ascontiguousarray(obj.endpoints,
       dtype=np.int8), obj.var_names)` — pass-through with a defensive
       contiguity copy so downstream code can rely on `np.int8` and
       row-major layout.
    2. **`np.ndarray` of shape `(n, n)`** — interpreted as the int8
       endpoint matrix directly. `var_names` MAY be passed by the
       caller; otherwise None.
    3. **`list[list[int]]` of shape `(n, n)`** — same as ndarray after
       np.asarray.
    4. **`networkx.DiGraph`** — converted via the rules below.

    nx.DiGraph conversion rules (applied to a fresh copy; caller graph
    is never mutated):
      - Node ordering: `list(g.nodes())` insertion order. Documented
        and tested. `var_names` defaults to that order; an explicit
        `var_names` argument re-orders accordingly (raises
        `BNMInputError` if names don't match the node set).
      - Edge `type` attribute is consulted:
          - `type='directed'` or absent → `endpoints[i,j]=ARROW`,
            `endpoints[j,i]=TAIL`.
          - `type='undirected'` → `endpoints[i,j]=TAIL`,
            `endpoints[j,i]=TAIL` (both ends TAIL).
          - any other value, INCLUDING `'bidirected'` → raise
            `BNMInputError`. Bidirected edges must come through cbcd's
            PAG or a raw int8 matrix; nx.DiGraph is too lossy a
            container for the full PAG mark set.
      - Both `A→B` and `B→A` present with no `type` (or both
        `type='directed'`) → raise `BNMInputError` (caller almost
        certainly meant bidirected; force them to the typed channel).

    Raises:
        BNMInputError on shape mismatch, ambiguous edges, or
        unsupported `type` values.
    """
    ...


def _resolve_var(
    var: int | str,
    var_names: tuple[str, ...] | None,
    n_vars: int,
) -> int:
    """Resolve a variable handle to an integer index.

    int → returned as-is after bounds check.
    str → looked up in `var_names`; raise `BNMInputError` if `var_names
    is None` or the name isn't present.
    """
    ...


# OPEN: should the adapter accept pandas DataFrames (treating them as
# adjacency matrices with named columns/rows)? 0.1.x doesn't.
# Lean NO — pandas is a soft dep, and ndarray+var_names covers the
# same ground without coupling. Revisit if a real user asks.


# =============================================================================
# D. DESCRIPTIVE METRICS  (Slice 1)
# =============================================================================
#
# Every descriptive metric is a free function over `GraphLikeInput`.
# Per-node degree functions take a `var: int | str` (no 'All' sentinel
# from 0.1.x). Aggregates over multiple nodes use the `compare()`
# function or are computed by the caller from per-node returns.


def count_edges(g: GraphLikeInput) -> int:
    """Total edges (directed + undirected + bidirected + circle-ended).

    Counts each undirected/bidirected edge exactly once.
    """
    ...


def count_nodes(g: GraphLikeInput) -> int:
    """Number of variables — equivalent to `g.n_vars` after normalisation."""
    ...


def count_directed_arcs(g: GraphLikeInput) -> int:
    """Count edges with one ARROW and one TAIL endpoint."""
    ...


def count_undirected_arcs(g: GraphLikeInput) -> int:
    """Count edges with two TAIL endpoints."""
    ...


def count_bidirected_arcs(g: GraphLikeInput) -> int:
    """Count edges with two ARROW endpoints. (New in v0.2; 0.1.x had no concept.)"""
    ...


def count_circle_edges(g: GraphLikeInput) -> int:
    """Count edges where at least one endpoint is CIRCLE. (PAG-specific.)"""
    ...


def count_colliders(g: GraphLikeInput) -> int:
    """Number of unshielded colliders: triples X → Z ← Y with X, Y not adjacent.

    ARROWs into Z count as parents (so bidirected and directed both
    contribute); two such parents not adjacent in `g` form a collider.
    Matches 0.1.x semantics on DAG/CPDAG inputs; generalises naturally
    to PAG.
    """
    ...


def count_root_nodes(g: GraphLikeInput) -> int:
    """Nodes with no directed in-edges and no undirected/bidirected/circle
    edges. Strict: any non-NO_EDGE column entry disqualifies."""
    ...


def count_leaf_nodes(g: GraphLikeInput) -> int:
    """Nodes with no directed out-edges and no undirected/bidirected/circle
    edges. Strict (mirrors `count_root_nodes`)."""
    ...


def count_isolated_nodes(g: GraphLikeInput) -> int:
    """Nodes with no edges of any kind."""
    ...


def count_reversible_arcs(g: GraphLikeInput) -> int:
    """Directed arcs not part of any unshielded collider (`X → Z ← Y`)."""
    ...


def in_degree(g: GraphLikeInput, var: int | str) -> int:
    """Number of directed in-edges to `var`. CIRCLE/TAIL/bidirected ends
    are NOT counted (only TAIL→ARROW into var)."""
    ...


def out_degree(g: GraphLikeInput, var: int | str) -> int:
    """Number of directed out-edges from `var`."""
    ...


# OPEN: should `count_colliders` be parametrized by "shielded vs
# unshielded"? 0.1.x counts only unshielded; PAG/MAG literature
# sometimes wants the shielded count. Lean: keep `count_colliders` as
# unshielded (default), add `count_shielded_colliders` later if a
# user asks. Decision deferred.


# =============================================================================
# E. COMPARATIVE METRICS  (Slice 2)
# =============================================================================
#
# Every comparative metric takes (g1, g2) where g1 is the reference
# (truth) and g2 is the comparison (estimate). Both must have equal
# `n_vars` after normalisation; mismatch raises `BNMDataError`.
# Variable-name alignment: if both have `var_names`, they must match
# as ordered tuples (raise `BNMDataError` otherwise). If only one
# has var_names, indices align positionally (no name reconciliation).


def count_additions(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Edges in g2 that have no corresponding adjacency in g1
    (presence-only, ignoring marks)."""
    ...


def count_deletions(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Edges in g1 that have no corresponding adjacency in g2."""
    ...


def count_reversals(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Edges where the adjacency exists in both but the orientation
    differs. Includes:
      - directed in g1 reversed-directed in g2
      - directed in g1 → undirected/bidirected/circle in g2
      - undirected in g1 → directed in g2 (either direction)
    Bidirected vs undirected is treated as a reversal (semantic
    upgrade over 0.1.x, which collapsed both to undirected pre-comp).
    """
    ...


def shd(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Structural Hamming Distance = additions + deletions + reversals."""
    ...


def hd(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Hamming Distance = additions + deletions (skeleton only, ignores
    orientation and edge type)."""
    ...


def true_positives(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Edges that match between g1 and g2 with the same marks
    (directed→directed in same direction, undirected→undirected,
    bidirected→bidirected). 0.1.x's "hybrid" matching."""
    ...


def false_positives(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Edges in g2 that are not exact-mark matches in g1."""
    ...


def false_negatives(g1: GraphLikeInput, g2: GraphLikeInput) -> int:
    """Edges in g1 that are not exact-mark matches in g2."""
    ...


def precision(g1: GraphLikeInput, g2: GraphLikeInput) -> float:
    """TP / (TP + FP). Returns 0.0 when TP+FP == 0."""
    ...


def recall(g1: GraphLikeInput, g2: GraphLikeInput) -> float:
    """TP / (TP + FN). Returns 0.0 when TP+FN == 0."""
    ...


def f1(g1: GraphLikeInput, g2: GraphLikeInput) -> float:
    """Harmonic mean of precision and recall. Returns 0.0 when both 0."""
    ...


# OPEN: should the comparative metrics expose an
# `orientation: Literal['exact', 'skeleton']` knob to pick between
# 0.1.x's "hybrid" matching and a strict-direction match? Lean NO —
# users wanting skeleton-only use `hd`; users wanting strict-mark
# matching use `shd`/`f1`. Don't conflate semantics into one knob.


# =============================================================================
# F. SID  (Slice 3)
# =============================================================================
#
# Structural Intervention Distance (Peters & Bühlmann 2015). v0.2
# rebinds the algorithm to the int8 endpoint matrix natively (no
# nx.DiGraph round-trips), with integer indices throughout.
#
# Semantic requirements:
#   - g1 (true graph) must be a pure DAG: every edge directed
#     (TAIL→ARROW), no CIRCLEs, no undirected, no bidirected. Acyclicity
#     is the caller's responsibility (matches 0.1.x; checking is O(V+E)
#     and adds noise to fixture-heavy test loops).
#   - g2 (estimated graph) may be a DAG or CPDAG (TAIL/TAIL allowed
#     for the undirected components). CIRCLE marks are rejected.
#
# Bug fix vs 0.1.x: Slice 3 special-cases `possible_pa_gp == ∅` to a
# single trivial parent-set assignment, fixing the
# `np.meshgrid(*[[0,1]] * 0)` crash (`bnmetrics/sid.py:371` in 0.1.x).


@dataclass(frozen=True, slots=True)
class SIDResult:
    """Output of `bnmetrics.sid`. Includes the full incorrect-intervention
    matrix for downstream visualisation (Slice 4 heatmap).
    """

    sid: int
    """SID value: total mis-classified intervention pairs."""

    sid_lower_bound: int
    """Lower bound on SID under CPDAG ambiguity (0.1.x calls this
    sid_lower_bound)."""

    sid_upper_bound: int
    """Upper bound under CPDAG ambiguity."""

    incorrect_mat: NDArray[np.int8]
    """(n, n) binary matrix marking (i, j) pairs whose intervention
    prediction differs between g1 and g2."""

    @property
    def is_tight(self) -> bool:
        """`True` if lower == upper (g2 was either a DAG or its CPDAG
        had no Markov-equivalence ambiguity)."""
        return self.sid_lower_bound == self.sid_upper_bound


def sid(g1: GraphLikeInput, g2: GraphLikeInput) -> SIDResult:
    """Compute SID between true DAG `g1` and estimated DAG/CPDAG `g2`.

    Raises:
        BNMInputError if g1 has any non-directed edges, if g2 has any
        CIRCLE marks, or if g1.n_vars != g2.n_vars.
    """
    ...


# OPEN: SID generalisation to PAGs is research-open. Not in scope
# for v0.2.


# =============================================================================
# G. MARKOV BLANKET  (Slice 1, returns a sub-GraphLike)
# =============================================================================
#
# `markov_blanket(g, var)` returns a sub-`GraphLike` over the int8
# endpoint matrix containing var, its directed parents, its directed
# children, the directed co-parents of those children, and any
# undirected neighbours. Same definition as 0.1.x (`utils.py:71-126`)
# but on the int8 matrix.
#
# Sub-graph contract: the returned object satisfies `GraphLike` and
# can be passed to any other bnmetrics function. Variable indices are
# RE-NUMBERED in the sub-graph (0..k-1 over the blanket); `var_names`
# of the sub-graph carries the original names for the included
# variables. Callers who want the original index space use
# `markov_blanket_indices(g, var)` instead.


@dataclass(frozen=True, slots=True)
class _SubGraph:
    """Concrete `GraphLike` produced by `markov_blanket` and other
    sub-graph queries. Internal: callers see only the `GraphLike`
    Protocol.
    """

    n_vars: int
    endpoints: NDArray[np.int8]
    var_names: tuple[str, ...] | None


def markov_blanket(g: GraphLikeInput, var: int | str) -> _SubGraph:
    """Return the Markov-blanket subgraph of `var` in `g`.

    The blanket is { var, parents(var), children(var),
    co_parents_of(children(var)), undirected_neighbours(var) }. Edges
    in the subgraph are exactly the edges of `g` whose both endpoints
    are in the blanket.
    """
    ...


def markov_blanket_indices(g: GraphLikeInput, var: int | str) -> tuple[int, ...]:
    """The index set of the Markov blanket of `var` in `g`'s ORIGINAL
    index space. Useful when callers want to slice their own arrays
    without re-numbering."""
    ...


# =============================================================================
# H. `compare()` FAÇADE + `Comparison` DATACLASS  (Slice 2 onward)
# =============================================================================
#
# The single multi-metric entry point. Replaces 0.1.x's
# `BNMetrics(...).compare_df(...)`. Returns a frozen dataclass; the
# DataFrame view is a free function (§I) so the metric layer doesn't
# import pandas.


# Metric-name registry. Used both by `compare()` for the `metrics`
# argument and by callers iterating over available names.
DESCRIPTIVE_METRIC_NAMES: tuple[str, ...] = (
    "n_edges",
    "n_nodes",
    "n_directed_arcs",
    "n_undirected_arcs",
    "n_bidirected_arcs",
    "n_circle_edges",
    "n_colliders",
    "n_root_nodes",
    "n_leaf_nodes",
    "n_isolated_nodes",
    "n_reversible_arcs",
)

COMPARATIVE_METRIC_NAMES: tuple[str, ...] = (
    "additions",
    "deletions",
    "reversals",
    "shd",
    "hd",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
)

# SID is opt-in (expensive on large CPDAGs) and produces a structured
# result, so it is not included in COMPARATIVE_METRIC_NAMES; callers
# pass `include_sid=True` to `compare()` instead.


@dataclass(frozen=True, slots=True)
class Comparison:
    """Output of `bnmetrics.compare`. Pure-data; no methods that touch I/O
    or pandas. Use `bnmetrics.to_dataframe(c)` for tabular views.
    """

    g1_descriptive: dict[str, float]
    """Descriptive metrics on g1, keyed by name from DESCRIPTIVE_METRIC_NAMES."""

    g2_descriptive: dict[str, float] | None
    """Descriptive metrics on g2; None when `compare()` is called with g2 omitted."""

    comparative: dict[str, float] | None
    """Comparative metrics; None when g2 omitted."""

    sid: "SIDResult | None"
    """SID result if requested; None otherwise."""

    per_node: dict[str | int, dict[str, float]] | None
    """Per-Markov-blanket sub-results when `per_node=True`. Keys are
    variable names (preferred) or indices; each value is a flat
    dict of metric name → value mixing descriptive+comparative for
    that blanket."""

    var_names: tuple[str, ...] | None
    """Variable names from the inputs (g1 takes precedence)."""


def compare(
    g1: GraphLikeInput,
    g2: GraphLikeInput | None = None,
    *,
    descriptive: Iterable[str] | Literal["all"] | None = "all",
    comparative: Iterable[str] | Literal["all"] | None = "all",
    include_sid: bool = False,
    per_node: bool | Iterable[int | str] = False,
) -> Comparison:
    """Compute a multi-metric comparison.

    Args:
        g1: reference graph.
        g2: estimated graph; if None, only descriptive metrics are
            computed.
        descriptive: which descriptive metrics to compute, "all" for
            every name in DESCRIPTIVE_METRIC_NAMES, or None to skip.
        comparative: which comparative metrics, same shape as
            descriptive.
        include_sid: also compute SID (g1 must be a pure DAG).
        per_node: True → compute per-Markov-blanket results for every
            variable in g1; iterable of var handles → only those.
            False (default) → only top-level (whole-graph) results.

    Raises:
        BNMInputError on inconsistent inputs (e.g. comparative metrics
        requested but g2 omitted; include_sid=True with non-DAG g1).
    """
    ...


# =============================================================================
# I. DataFrame VIEW  (free function, lazy pandas import)
# =============================================================================


def to_dataframe(comparison: Comparison) -> "Any":
    """Render a `Comparison` as a pandas DataFrame. Imports pandas
    lazily; raises `ImportError` with a helpful message if pandas is
    not installed.

    Returns a frame with one row per node (`node_name`) plus an `All`
    row for whole-graph metrics, columns for each descriptive (with
    `_base` suffix when both g1 and g2 are descriptive-summarised) and
    comparative metric. Mirrors 0.1.x's `compare_df` shape but is
    decoupled from the computation.
    """
    ...


# =============================================================================
# J. VISUALIZATION  (Slice 4 — gated on `viz` extra)
# =============================================================================
#
# All viz functions are stubs in v0.2 design; implementations land in
# Slice 4. Same input contract (`GraphLikeInput`); same `var | int |
# str` handle for nodes.


def plot_graph(g: GraphLikeInput, *, title: str = "DAG", highlight: Iterable[int | str] = ()) -> "Any":
    """Render a single graph via graphviz. Returns the rendered SVG
    string in non-notebook contexts; displays in IPython otherwise."""
    ...


def plot_side_by_side(
    g1: GraphLikeInput,
    g2: GraphLikeInput,
    *,
    name1: str = "G1",
    name2: str = "G2",
    mode: 'Literal["matches", "diff", "none"]' = "matches",
    highlight_nodes: Iterable[int | str] = (),
    highlight_node_color: str = "#c8e6c9",  # pastel green default
    highlight_edge_color: str = "#f08080",  # pastel red default
) -> "Any":
    """Side-by-side render of g1 and g2 with edge highlighting.

    ``mode`` selects which edges receive the highlight stroke:
      * ``"matches"`` (default) — true-positive edges (same kind, same
        direction; CIRCLE-bearing edges must agree on the full mark
        pair). Mirrors the 0.1.x behaviour.
      * ``"diff"`` — additions, deletions, reversals, and kind changes.
        Each side's edge is highlighted in whichever panel contains it.
      * ``"none"`` — no edge highlighting (only ``highlight_nodes``).
    """
    ...


def plot_sid_matrix(result: "SIDResult", *, var_names: Sequence[str] | None = None) -> "Any":
    """Plotly heatmap of `result.incorrect_mat`."""
    ...


# OPEN: visualization for cbcd PAG outputs (CIRCLE marks). Slice 4
# decision: does the graphviz renderer use a small open circle for
# CIRCLE? Most PAG papers use exactly that convention. Confirm at
# Slice 4 implementation time.


# =============================================================================
# K. PUBLIC RE-EXPORTS  (the v0.x API contract)
# =============================================================================
#
# Mirrors cbcd's D15 stance: the names in this list are committed
# backwards-compatible across all v0.x minor and patch bumps. Stub-only
# items (currently `count_circle_edges` for non-PAG callers,
# `_SubGraph` / `markov_blanket_indices`) are NOT yet frozen until
# the relevant slice ships.

PUBLIC_API_v0: tuple[str, ...] = (
    # graph contract + marks
    "GraphLike",
    "EndpointMark",
    # exceptions
    "BNMError",
    "BNMInputError",
    "BNMDataError",
    # descriptive
    "count_edges",
    "count_nodes",
    "count_directed_arcs",
    "count_undirected_arcs",
    "count_bidirected_arcs",
    "count_circle_edges",
    "count_colliders",
    "count_root_nodes",
    "count_leaf_nodes",
    "count_isolated_nodes",
    "count_reversible_arcs",
    "in_degree",
    "out_degree",
    # comparative
    "count_additions",
    "count_deletions",
    "count_reversals",
    "shd",
    "hd",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    # sid
    "sid",
    "SIDResult",
    # markov blanket
    "markov_blanket",
    "markov_blanket_indices",
    # multi-metric comparison + view
    "compare",
    "Comparison",
    "to_dataframe",
    "DESCRIPTIVE_METRIC_NAMES",
    "COMPARATIVE_METRIC_NAMES",
    # viz (Slice 4 — stubs until then)
    "plot_graph",
    "plot_side_by_side",
    "plot_sid_matrix",
)


# =============================================================================
# L. DECISIONS LOG + OPEN QUESTIONS
# =============================================================================
#
# D1 (2026-05-07). API shape: pure functions + a thin compare() wrapper. `BNMetrics`
#   god-class dropped entirely. Migration cost paid once on the v0.1
#   → v0.2 break; CHANGELOG documents the mapping (`BNMetrics(g1,g2)
#   .shd` → `bnmetrics.shd(g1, g2)`).
#
# D2 (2026-05-07). Internal storage is the int8 endpoint matrix
#   matching cbcd's numeric convention. bnmetrics DOES NOT import cbcd; the
#   GraphLike Protocol is the only interop contract.
#
# D3 (2026-05-07). nx.DiGraph adapter ERRORS on bidirected. No third
#   edge dialect. Bidirected callers route through cbcd PAG or a raw
#   int8 matrix.
#
# D4 (2026-05-07). Test strategy: hand-computed canonical fixtures
#   (Y, fork, chain, M, diamond, ASIA) in every slice, plus a frozen
#   golden snapshot of ~30 random fixtures against bnmetrics 0.1.x output
#   at `tests/fixtures_legacy.json`. Five fixtures with known 0.1.x
#   SID crashes carry `"sid": {"skipped": ...}` in the snapshot and
#   are reclaimed via hand-computed values in Slice 3 tests.
#
# D5 (2026-05-07). pandas is a soft dep. Metric layer imports only
#   numpy. `to_dataframe` lazy-imports pandas at call time.
#
# D6 (2026-05-07). graphviz / plotly / IPython are gated under the
#   `viz` extra; the metric layer never imports them.
#
# D7 (2026-05-07). networkx is a soft dep too — only the
#   `_to_endpoints` adapter imports it, lazily, when an nx.DiGraph
#   input is detected. Callers who pass cbcd graphs or raw matrices
#   never trigger the import.
#
# D8 (2026-05-07). `Comparison` is `frozen=True, slots=True`. No
#   methods that perform I/O; `to_dataframe` is a free function.
#
# D9 (2026-05-07). SID `g1` is required to be a pure DAG. Acyclicity
#   is NOT checked (matches 0.1.x; runtime cost is meaningful for
#   fixture-heavy test loops). Callers passing cyclic g1 get garbage,
#   same as in 0.1.x.
#
# D10 (2026-05-07). The `'All'` aggregator sentinel from 0.1.x is
#   dropped. `in_degree(g, 'All')` is no longer a valid call.
#   Aggregates use `compare()`'s `per_node=False` (default) for the
#   whole-graph view.
#
# D11 (2026-05-07). Public API contract: the names listed in §K
#   `PUBLIC_API_v0` are committed backwards-compatible across v0.x.
#   Stubs (e.g. PAG-specific viz) are not yet frozen until the
#   implementing slice lands.
#
# OPEN QUESTIONS (resolved or deferred):
#
#   O1 (resolved 2026-05-09, v0.2.2): CIRCLE marks render as graphviz
#       `odot` arrowheads (the "small open circle on the relevant
#       endpoint" question — answer: yes). Edge-matching for CIRCLE-
#       bearing edges compares the full ``(mij, mji)`` mark pair so
#       different PAG topologies don't false-match.
#   O2 (resolved 2026-05-09, v0.2.2): `plot_side_by_side` gained
#       ``mode: Literal["matches", "diff", "none"] = "matches"``.
#       ``"diff"`` highlights additions, deletions, reversals, and
#       kind changes. ``"none"`` disables edge highlighting (covers
#       the previous `highlight_true_positives=False` use case).
#   O3 (deferred): SID generalisation to PAGs is research-open. Not
#       in scope for v0.2; revisit when a referenced extension is
#       published.
#   O4 (deferred): support for cbcd's TimeSeriesCPDAG (3D endpoint
#       matrix). Out of scope for v0.2; the GraphLike Protocol speaks
#       only in 2D matrices today.
#   O5 (deferred): a `pandas.DataFrame` adjacency input form for
#       `_to_endpoints`. NO for now; revisit if a user asks.
