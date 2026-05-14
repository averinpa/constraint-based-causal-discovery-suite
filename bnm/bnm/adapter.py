"""Input-graph adapter: normalise everything to (n_vars, endpoints, var_names)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bnm._graph import _Graph
from bnm.exceptions import BNMInputError
from bnm.marks import EndpointMark
from bnm.protocol import GraphLike

_VALID_MARKS = frozenset(int(m) for m in EndpointMark)


def _is_graphlike(obj: object) -> bool:
    """Duck-typed GraphLike check.

    Avoids `isinstance(obj, GraphLike)` (which scans for the attrs
    *and* triggers descriptor lookups) and avoids accidentally
    matching plain ndarrays (which have `n_vars`-like shape access
    but no `.endpoints`).
    """
    return (
        hasattr(obj, "n_vars")
        and hasattr(obj, "endpoints")
        and hasattr(obj, "var_names")
        and not isinstance(obj, np.ndarray)
    )


def _validate_endpoints(endpoints: NDArray[np.int8], *, source: str) -> None:
    """Common shape/mark validation. Raises BNMInputError on failure.

    Vectorised over the full (n, n) matrix — O(n²) numpy ops, no Python
    loop over indices. Hot-pathed by every external GraphLike call.
    """
    if endpoints.ndim != 2 or endpoints.shape[0] != endpoints.shape[1]:
        raise BNMInputError(
            f"{source}: endpoints must be a square 2D matrix, got shape {endpoints.shape}"
        )
    unique = np.unique(endpoints)
    bad_marks = [int(v) for v in unique if int(v) not in _VALID_MARKS]
    if bad_marks:
        raise BNMInputError(
            f"{source}: endpoint marks must be in {sorted(_VALID_MARKS)}; "
            f"got unexpected marks {bad_marks}"
        )
    no_edge = int(EndpointMark.NO_EDGE)
    diag = np.diagonal(endpoints)
    bad_diag = np.flatnonzero(diag != no_edge)
    if bad_diag.size:
        i = int(bad_diag[0])
        raise BNMInputError(
            f"{source}: diagonal must be NO_EDGE; got {int(endpoints[i, i])} at ({i}, {i})"
        )
    no_edge_mask = endpoints == no_edge
    asymmetric = no_edge_mask != no_edge_mask.T
    if asymmetric.any():
        upper = np.triu(asymmetric, k=1)
        ii, jj = np.where(upper)
        i, j = int(ii[0]), int(jj[0])
        raise BNMInputError(
            f"{source}: edge ({i}, {j}) has one NO_EDGE end and one non-NO_EDGE end; "
            f"marks=({int(endpoints[j, i])}, {int(endpoints[i, j])})"
        )


def _resolve_var_names(
    var_names: tuple[str, ...] | list[str] | None,
    n_vars: int,
    *,
    source: str,
) -> tuple[str, ...] | None:
    if var_names is None:
        return None
    names = tuple(var_names)
    if len(names) != n_vars:
        raise BNMInputError(
            f"{source}: var_names has {len(names)} entries but graph has {n_vars} variables"
        )
    if len(set(names)) != len(names):
        raise BNMInputError(f"{source}: var_names contains duplicates")
    return names


def _from_nx_digraph(
    g: Any,
    var_names: tuple[str, ...] | list[str] | None,
) -> tuple[int, NDArray[np.int8], tuple[str, ...] | None]:
    """Convert nx.DiGraph (with optional 'type' edge attr) to int8 matrix.

    Rules:
      - Node ordering = ``list(g.nodes())`` insertion order, unless
        ``var_names`` is provided (in which case the explicit order
        wins; raises if the name set differs from g.nodes()).
      - Edge ``type='directed'`` or absent → endpoints[src, dst]=ARROW,
        endpoints[dst, src]=TAIL.
      - Edge ``type='undirected'`` → endpoints[i, j]=TAIL on both ends.
      - Any other ``type`` value (including 'bidirected') → raise.
      - Both ``A→B`` and ``B→A`` present (no/'directed' type) → raise
        (caller should be using the typed channel).
    """
    nodes = list(g.nodes())
    n_vars = len(nodes)
    if var_names is None:
        names = tuple(str(n) for n in nodes)
    else:
        names = tuple(var_names)
        if set(names) != {str(n) for n in nodes}:
            raise BNMInputError(
                "_to_endpoints: var_names must match the node set of the nx.DiGraph"
            )
    name_to_idx = {n: i for i, n in enumerate(names)}
    # Allow lookup by either string or original node identity.
    lookup = {n: name_to_idx[str(n)] for n in nodes}

    endpoints = np.zeros((n_vars, n_vars), dtype=np.int8)
    arrow = int(EndpointMark.ARROW)
    tail = int(EndpointMark.TAIL)

    seen_directed: set[tuple[int, int]] = set()
    for u, v, data in g.edges(data=True):
        i = lookup[u]
        j = lookup[v]
        if i == j:
            raise BNMInputError(f"_to_endpoints: self-loop on node {u!r} is not supported")
        edge_type = data.get("type", "directed")
        if edge_type == "directed":
            if (j, i) in seen_directed:
                raise BNMInputError(
                    f"_to_endpoints: nx.DiGraph contains both {u!r}→{v!r} and "
                    f"{v!r}→{u!r} without type='undirected'. Use type='undirected' "
                    f"explicitly, or pass an int8 matrix for bidirected/PAG semantics."
                )
            endpoints[i, j] = arrow
            endpoints[j, i] = tail
            seen_directed.add((i, j))
        elif edge_type == "undirected":
            # Idempotent: both ends already TAIL after first emission.
            endpoints[i, j] = tail
            endpoints[j, i] = tail
        else:
            raise BNMInputError(
                f"_to_endpoints: nx.DiGraph edge {u!r}→{v!r} has unsupported "
                f"type={edge_type!r}. Allowed: 'directed', 'undirected'. "
                f"Bidirected and CIRCLE-mark graphs must be passed as an int8 "
                f"endpoint matrix or as a cbcd PAG instance."
            )

    return n_vars, endpoints, names


def _to_endpoints(
    obj: object,
    *,
    var_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[int, NDArray[np.int8], tuple[str, ...] | None]:
    """Normalise any accepted input form to ``(n_vars, endpoints, var_names)``.

    Accepted input forms:
      1. ``GraphLike`` (cbcd DAG/CPDAG/PAG, any user wrapper) —
         pass-through with a contiguous int8 copy.
      2. ``np.ndarray`` of shape ``(n, n)`` — interpreted as the
         endpoint matrix directly.
      3. ``list[list[int]]`` — same as ndarray after `np.asarray`.
      4. ``networkx.DiGraph`` — converted via the rules in
         :func:`_from_nx_digraph`. Networkx is lazy-imported only
         when this branch is taken.

    Raises
    ------
    BNMInputError
        On any shape/type/value violation.
    """
    if _is_graphlike(obj):
        gl: GraphLike = obj  # type: ignore[assignment]
        endpoints = np.ascontiguousarray(gl.endpoints, dtype=np.int8)
        # Fast path: trust internal `_Graph` instances (we built them
        # via this same adapter, so they're already validated). Skip
        # the O(n²) re-validation. External GraphLike implementations
        # still get validated.
        if not isinstance(obj, _Graph):
            _validate_endpoints(endpoints, source="GraphLike input")
        names = (
            _resolve_var_names(var_names, gl.n_vars, source="_to_endpoints")
            if var_names is not None
            else gl.var_names
        )
        return gl.n_vars, endpoints, names

    if isinstance(obj, np.ndarray):
        endpoints = np.ascontiguousarray(obj, dtype=np.int8)
        _validate_endpoints(endpoints, source="ndarray input")
        n_vars = endpoints.shape[0]
        names = _resolve_var_names(var_names, n_vars, source="_to_endpoints")
        return n_vars, endpoints, names

    if isinstance(obj, list):
        endpoints = np.asarray(obj, dtype=np.int8)
        _validate_endpoints(endpoints, source="list input")
        n_vars = endpoints.shape[0]
        names = _resolve_var_names(var_names, n_vars, source="_to_endpoints")
        return n_vars, endpoints, names

    # Networkx is a soft dep — only import when an nx.DiGraph is detected.
    try:
        import networkx as nx  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if isinstance(obj, nx.DiGraph):
            return _from_nx_digraph(obj, var_names)

    raise BNMInputError(
        f"_to_endpoints: unsupported input type {type(obj).__name__}. "
        f"Expected GraphLike (e.g. cbcd DAG/CPDAG/PAG), np.ndarray, "
        f"list of lists, or networkx.DiGraph."
    )


def to_graphlike(
    obj: object,
    *,
    var_names: tuple[str, ...] | list[str] | None = None,
) -> _Graph:
    """Public-facing wrapper returning a concrete `_Graph` (a `GraphLike`).

    Useful when callers want a normalised handle once and then pass it
    repeatedly to bnm functions without re-validating each time.
    """
    n_vars, endpoints, names = _to_endpoints(obj, var_names=var_names)
    return _Graph(n_vars=n_vars, endpoints=endpoints, var_names=names)


def _resolve_var(
    var: int | str,
    var_names: tuple[str, ...] | None,
    n_vars: int,
) -> int:
    """Resolve a variable handle to an integer index.

    int → bounds-checked and returned.
    str → looked up in ``var_names``; raise BNMInputError if
    ``var_names is None`` or the name isn't present.
    """
    if isinstance(var, (int, np.integer)):
        i = int(var)
        if not 0 <= i < n_vars:
            raise BNMInputError(f"variable index {var} out of range [0, {n_vars})")
        return i
    if isinstance(var, str):
        if var_names is None:
            raise BNMInputError(
                f"variable {var!r} given as a string, but the graph has no var_names"
            )
        try:
            return var_names.index(var)
        except ValueError:
            raise BNMInputError(f"variable {var!r} not found in var_names {var_names}") from None
    raise BNMInputError(f"variable handle must be int or str, got {type(var).__name__}")
