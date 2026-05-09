# The `GraphLike` Protocol

`bnm.GraphLike` is a structural Protocol — i.e. a duck-typed
interface enforced at the input boundary by `bnm.to_graphlike`,
not by inheritance. Any object exposing the three required members
is accepted by every `bnm` function.

## Members

A conforming object provides:

- `n_vars: int` — the number of vertices.
- `endpoints: numpy.ndarray` — an `int8` matrix of shape
  `(n_vars, n_vars)` encoding edge marks (decision **D5**:
  `0` no edge / `1` TAIL / `2` ARROW / `3` CIRCLE).
- `var_names: tuple[str, ...]` — variable names of length
  `n_vars`. May be empty if the graph has no named vertices.

## Why a Protocol rather than a base class

The four-package suite (`cbcd`, `bnm`, `dagsampler`, `citk`) is
designed so that no package imports another at runtime. Cross-
package interoperability is mediated by structural Protocols
defined at the *boundary* of each package: `cbcd.CITest` for
conditional-independence tests, `bnm.GraphLike` for graph types.
Because these are runtime-checkable Protocols (PEP 544), a
producer (e.g. `cbcd.CPDAG`) and a consumer (e.g. `bnm.shd`)
agree on the interface without either one importing the other.
This is the load-bearing claim that lets the suite ship as four
independently-versioned packages rather than a single monolith.

## Validation

`bnm.to_graphlike(obj)` performs the following normalisation and
validation:

1. If `obj` already satisfies `GraphLike`, the endpoint matrix is
   copied to ensure caller mutations do not affect downstream
   metric computations.
2. If `obj` is a `numpy.ndarray` or list-of-lists, it is taken
   directly as the endpoint matrix.
3. If `obj` is a `networkx.DiGraph`, the directed-edge convention
   is applied (with `type="undirected"` edges encoded as
   TAIL-on-both-ends).
4. The resulting matrix is checked for: square shape; symmetry of
   adjacency (an edge present at one end must be present at the
   other); valid mark values; no self-loops.

A `BNMInputError` is raised for any failure; the error message
includes the offending row/column where applicable.

## What about MAGs and PAGs?

PAGs and MAGs use the same endpoint-mark vocabulary as DAGs and
CPDAGs (with the addition of CIRCLE marks for PAGs). They
satisfy `GraphLike` directly, and every `bnm` metric — except
SID, which is currently DAG-vs-{DAG, CPDAG} only — accepts them
as input.

## Time-series graphs

The current `GraphLike` Protocol does not cover lagged time-series
graphs (e.g. `cbcd.TimeSeriesCPDAG`). Open question O4 in the bnm
design document tracks the time-series extension; it is deferred
until the i.i.d. surface stabilises through one or more public
releases.

## References

- Van Rossum, G., Lehtosalo, J., & Langa, Ł. (2014). PEP 544 –
  Protocols: Structural subtyping (static duck typing).
