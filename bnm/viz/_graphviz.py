"""Graphviz-backed graph rendering."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from bnm.adapter import _resolve_var, _to_endpoints
from bnm.exceptions import BNMError, BNMInputError
from bnm.marks import EndpointMark

PathLike = str | os.PathLike[str]

# Supported `save=` extensions for graphviz output. `dot` and `gv` write
# the raw DOT source; everything else routes through `Digraph.pipe(format=...)`.
_GRAPHVIZ_FORMATS = frozenset(
    {"svg", "png", "pdf", "jpg", "jpeg", "ps", "ps2", "dot", "gv", "json"}
)


def _save_dot(dot: Any, path: PathLike) -> None:
    """Persist a `graphviz.Digraph` to disk; format inferred from extension."""
    p = Path(path)
    fmt = p.suffix.lstrip(".").lower()
    if not fmt:
        raise BNMInputError(
            f"save path {p!s} has no extension; cannot infer format. "
            f"Use one of {sorted(_GRAPHVIZ_FORMATS)}."
        )
    if fmt not in _GRAPHVIZ_FORMATS:
        raise BNMInputError(
            f"unsupported save format {fmt!r} for graphviz output; "
            f"supported: {sorted(_GRAPHVIZ_FORMATS)}"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("dot", "gv"):
        p.write_text(dot.source)
    else:
        p.write_bytes(dot.pipe(format=fmt))


NO_EDGE = int(EndpointMark.NO_EDGE)
TAIL = int(EndpointMark.TAIL)
ARROW = int(EndpointMark.ARROW)
CIRCLE = int(EndpointMark.CIRCLE)


# Aesthetic presets. Each entry is ``(graph_attr, node_attr, edge_attr)``
# applied as Digraph-wide defaults. Per-node / per-edge overrides
# (highlighting) layer on top. The ``rankdir`` graph-attr in each
# preset is the style's natural default but is overridden by the
# ``direction`` parameter on every plot function.
_STYLES: dict[str, tuple[dict[str, str], dict[str, str], dict[str, str]]] = {
    # Subtle: ellipse with very-light grey fill, thin border. Modern-
    # but-conservative academic look. Default in v0.2.
    "subtle": (
        {
            "bgcolor": "white",
            "fontname": "Helvetica",
            "rankdir": "TB",
            "splines": "true",
            "nodesep": "0.3",
        },
        {
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#f8f8f8",
            "color": "#888888",
            "fontcolor": "#222222",
            "fontname": "Helvetica",
            "fontsize": "11",
            "width": "0.5",
            "height": "0.32",
            "penwidth": "0.8",
        },
        {
            "color": "#555555",
            "arrowsize": "0.7",
            "penwidth": "0.9",
        },
    ),
}

StyleName = Literal["subtle"]
Direction = Literal["TB", "LR", "auto"]


def _resolve_style(
    style: StyleName,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if style not in _STYLES:
        raise BNMInputError(f"unknown viz style {style!r}; available: {sorted(_STYLES)}")
    return _STYLES[style]


def _auto_direction(endpoints) -> str:
    """Pick ``"TB"`` or ``"LR"`` based on graph shape.

    Counts directed-edge layers (Kahn topological levels) and average
    breadth (``n_vars / layers``). Returns ``"LR"`` for "thin and tall"
    graphs (avg breadth < 1.5), else ``"TB"``. Undirected and other
    non-directed edges are ignored when computing layers — they don't
    impose a ranking. Falls back to ``"TB"`` on cycles or empty graphs.
    """
    from collections import deque  # local: keep module top-level lean

    n = endpoints.shape[0]
    if n == 0:
        return "TB"
    children: list[list[int]] = [[] for _ in range(n)]
    in_deg = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mij = int(endpoints[i, j])
            mji = int(endpoints[j, i])
            if mij == ARROW and mji == TAIL:
                children[i].append(j)
                in_deg[j] += 1

    queue: deque[int] = deque(v for v in range(n) if in_deg[v] == 0)
    layer = [0] * n
    visited = 0
    while queue:
        v = queue.popleft()
        visited += 1
        for c in children[v]:
            if layer[v] + 1 > layer[c]:
                layer[c] = layer[v] + 1
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)

    if visited < n:
        # Cycle (shouldn't happen on valid DAG/CPDAG inputs).
        return "TB"
    layers = max(layer) + 1
    avg_breadth = n / layers
    return "LR" if avg_breadth < 1.5 else "TB"


def _resolve_direction(direction: Direction, endpoints) -> str:
    if direction == "auto":
        return _auto_direction(endpoints)
    if direction in ("TB", "LR"):
        return direction
    raise BNMInputError(f"unknown viz direction {direction!r}; allowed: 'TB', 'LR', 'auto'")


def _require_graphviz() -> Any:
    try:
        import graphviz  # noqa: PLC0415
    except ImportError as exc:
        raise BNMError(
            "bnm.viz requires the `viz` extra. Install with `pip install bnm[viz]` "
            "(adds graphviz, plotly, ipython)."
        ) from exc
    return graphviz


def _label_of(idx: int, var_names: tuple[str, ...] | None) -> str:
    return var_names[idx] if var_names is not None else str(idx)


def _build_dot(
    n_vars: int,
    endpoints,
    var_names: tuple[str, ...] | None,
    *,
    name: str,
    highlighted_nodes: set[int],
    highlighted_edges: set[tuple[int, int, str]],
    style: StyleName,
    direction: Direction,
) -> Any:
    """Build a graphviz.Digraph from an int8 endpoint matrix.

    Edge marks are rendered using arrowhead / arrowtail / dir options:
      - i→j directed: arrowhead at j only.
      - i—j undirected: dir='none'.
      - i↔j bidirected: arrowhead at both.
      - CIRCLE marks: rendered as `odot` arrowheads (PAG convention).

    `highlighted_edges` contains entries `(i, j, kind)` where kind is
    'directed', 'undirected', or 'bidirected'; matching edges get a
    crimson stroke. Highlighted nodes get a lightgreen fill (overrides
    the style's default node fill).
    """
    graphviz = _require_graphviz()
    graph_attr, node_attr, edge_attr = _resolve_style(style)
    dot = graphviz.Digraph(name=name)
    dot.graph_attr.update(graph_attr)
    dot.graph_attr["rankdir"] = _resolve_direction(direction, endpoints)
    dot.node_attr.update(node_attr)
    dot.edge_attr.update(edge_attr)

    for v in range(n_vars):
        attrs: dict[str, str] = {}
        if v in highlighted_nodes:
            attrs["style"] = "filled"
            attrs["fillcolor"] = "lightgreen"
        dot.node(_label_of(v, var_names), **attrs)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mij = int(endpoints[i, j])
            mji = int(endpoints[j, i])
            if mij == NO_EDGE:
                continue
            li, lj = _label_of(i, var_names), _label_of(j, var_names)

            arrowhead = _mark_to_graphviz_arrow(mij)
            arrowtail = _mark_to_graphviz_arrow(mji)

            edge_kind = _classify_edge(mij, mji)
            colour = "crimson" if (i, j, edge_kind) in highlighted_edges else None

            edge_attrs: dict[str, str] = {}
            if edge_kind == "directed-fwd":
                # i → j: just the arrow at j.
                if colour:
                    edge_attrs["color"] = colour
                dot.edge(li, lj, **edge_attrs)
            elif edge_kind == "directed-bwd":
                # j → i: render in reverse direction.
                if colour:
                    edge_attrs["color"] = colour
                dot.edge(lj, li, **edge_attrs)
            elif edge_kind == "undirected":
                edge_attrs["dir"] = "none"
                if colour:
                    edge_attrs["color"] = colour
                dot.edge(li, lj, **edge_attrs)
            elif edge_kind == "bidirected":
                edge_attrs["dir"] = "both"
                if colour:
                    edge_attrs["color"] = colour
                dot.edge(li, lj, **edge_attrs)
            else:
                # PAG with CIRCLE marks — use arrowhead/arrowtail directly.
                edge_attrs["dir"] = "both"
                edge_attrs["arrowhead"] = arrowhead
                edge_attrs["arrowtail"] = arrowtail
                if colour:
                    edge_attrs["color"] = colour
                dot.edge(li, lj, **edge_attrs)

    return dot


def _mark_to_graphviz_arrow(mark: int) -> str:
    """Map an EndpointMark to a graphviz arrowhead/arrowtail style."""
    if mark == ARROW:
        return "normal"
    if mark == TAIL:
        return "none"
    if mark == CIRCLE:
        return "odot"
    return "none"


def _classify_edge(mij: int, mji: int) -> str:
    if mij == ARROW and mji == TAIL:
        return "directed-fwd"
    if mij == TAIL and mji == ARROW:
        return "directed-bwd"
    if mij == TAIL and mji == TAIL:
        return "undirected"
    if mij == ARROW and mji == ARROW:
        return "bidirected"
    return "circle"


def _normalise_node_set(
    nodes: Iterable[int | str],
    var_names: tuple[str, ...] | None,
    n_vars: int,
) -> set[int]:
    return {_resolve_var(v, var_names, n_vars) for v in nodes}


def plot_graph(
    g: object,
    *,
    title: str = "DAG",
    highlight: Iterable[int | str] = (),
    style: StyleName = "subtle",
    direction: Direction = "TB",
    save: PathLike | None = None,
) -> Any:
    """Render a single graph as a graphviz.Digraph.

    Args:
        g: any GraphLikeInput (cbcd graph, ndarray, list, nx.DiGraph,
            internal `_Graph`).
        title: graph name (visible if you call `.render(...)` to a file).
        highlight: variables to fill lightgreen.
        style: aesthetic preset. Currently ``"subtle"`` (the v0.2
            default — ellipse with light grey fill, thin border,
            Helvetica).
        direction: layout direction. ``"TB"`` (default) for top-down,
            ``"LR"`` for left-to-right (better for chain-shaped or
            long-named graphs), or ``"auto"`` to switch based on
            graph shape (LR for tall thin graphs, TB otherwise).
        save: optional path. Format is inferred from the file extension
            (``svg``, ``png``, ``pdf``, ``jpg``, ``jpeg``, ``ps``,
            ``json``, or raw DOT source via ``dot``/``gv``). The
            function still returns the Digraph regardless of whether
            ``save`` is set, so you can also display it inline.

    Returns:
        graphviz.Digraph. In a Jupyter context, it auto-renders inline.
        For programmatic use, call ``.pipe(format='svg')`` or
        ``.render(...)`` on the returned object.
    """
    n_vars, endpoints, var_names = _to_endpoints(g)
    highlighted = _normalise_node_set(highlight, var_names, n_vars)
    dot = _build_dot(
        n_vars,
        endpoints,
        var_names,
        name=title,
        highlighted_nodes=highlighted,
        highlighted_edges=set(),
        style=style,
        direction=direction,
    )
    if save is not None:
        _save_dot(dot, save)
    return dot


def _matching_edges(
    n_vars: int,
    endpoints1,
    endpoints2,
) -> set[tuple[int, int, str]]:
    """Edges that match exactly between g1 and g2 (same kind, same
    direction). Used for true-positive highlighting."""
    out: set[tuple[int, int, str]] = set()
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mij1, mji1 = int(endpoints1[i, j]), int(endpoints1[j, i])
            mij2, mji2 = int(endpoints2[i, j]), int(endpoints2[j, i])
            if mij1 == NO_EDGE or mij2 == NO_EDGE:
                continue
            kind1 = _classify_edge(mij1, mji1)
            kind2 = _classify_edge(mij2, mji2)
            if kind1 == kind2:
                out.add((i, j, kind1))
    return out


class _SideBySideDot:
    """Pair of graphviz Digraphs returned from :func:`plot_side_by_side`.

    Renders as two SVGs in an HTML flex container in Jupyter; iterates
    as a 2-tuple ``(dot_g1, dot_g2)`` for backwards-compatible
    unpacking (``dot1, dot2 = bnm.plot_side_by_side(...)``); supports
    indexing ``[0]`` / ``[1]`` for direct access.
    """

    __slots__ = ("dot1", "dot2", "name1", "name2")

    def __init__(self, dot1: Any, dot2: Any, name1: str, name2: str) -> None:
        self.dot1 = dot1
        self.dot2 = dot2
        self.name1 = name1
        self.name2 = name2

    def __iter__(self) -> Iterable[Any]:
        yield self.dot1
        yield self.dot2

    def __getitem__(self, idx: int) -> Any:
        return (self.dot1, self.dot2)[idx]

    def __len__(self) -> int:
        return 2

    def _repr_html_(self) -> str:
        svg1 = self.dot1.pipe(format="svg").decode("utf-8")
        svg2 = self.dot2.pipe(format="svg").decode("utf-8")
        h_style = "margin:4px 0;font-weight:500;color:#333;"
        outer = "display: flex; gap: 36px; align-items: flex-start; flex-wrap: wrap;"
        return (
            f'<div style="{outer}">'
            f'<div style="text-align: center;">'
            f'<h4 style="{h_style}">{self.name1}</h4>{svg1}</div>'
            f'<div style="text-align: center;">'
            f'<h4 style="{h_style}">{self.name2}</h4>{svg2}</div>'
            "</div>"
        )


def plot_side_by_side(
    g1: object,
    g2: object,
    *,
    name1: str = "G1",
    name2: str = "G2",
    highlight_true_positives: bool = True,
    highlight_nodes: Iterable[int | str] = (),
    style: StyleName = "subtle",
    direction: Direction = "TB",
    save: PathLike | tuple[PathLike, PathLike] | None = None,
) -> _SideBySideDot:
    """Render g1 and g2 as two side-by-side graphviz.Digraphs.

    Variables in `highlight_nodes` are filled lightgreen in both. If
    `highlight_true_positives`, edges that match exactly (same kind,
    same direction) get a crimson stroke in both.

    ``style`` selects the aesthetic preset (default ``"subtle"``).
    ``direction`` is ``"TB"`` (default), ``"LR"``, or ``"auto"``;
    when ``"auto"``, each graph's direction is chosen independently
    based on its own shape.

    ``save`` accepts:
      * a single path like ``"comparison.svg"`` — written as two files
        ``comparison_<name1>.svg`` and ``comparison_<name2>.svg``;
      * a tuple of two paths ``(path_g1, path_g2)`` — written one each;
      * ``None`` (default) — no save.

    Variable-name alignment between g1 and g2 follows the same rule as
    comparative metrics: equal `var_names` tuples (when both have
    them) or positional alignment otherwise.
    """
    from bnm.exceptions import BNMDataError  # local to avoid surfacing in __init__

    n1, ep1, names1 = _to_endpoints(g1)
    n2, ep2, names2 = _to_endpoints(g2)
    if n1 != n2:
        raise BNMDataError(f"plot_side_by_side: g1 has {n1} variables, g2 has {n2}")
    if names1 is not None and names2 is not None and names1 != names2:
        raise BNMDataError("plot_side_by_side: g1.var_names and g2.var_names differ")
    var_names = names1 if names1 is not None else names2
    highlighted_nodes = _normalise_node_set(highlight_nodes, var_names, n1)
    highlighted_edges = _matching_edges(n1, ep1, ep2) if highlight_true_positives else set()

    dot1 = _build_dot(
        n1,
        ep1,
        var_names,
        name=name1,
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
        style=style,
        direction=direction,
    )
    dot2 = _build_dot(
        n2,
        ep2,
        var_names,
        name=name2,
        highlighted_nodes=highlighted_nodes,
        highlighted_edges=highlighted_edges,
        style=style,
        direction=direction,
    )
    if save is not None:
        _save_side_by_side(dot1, dot2, name1=name1, name2=name2, save=save)
    return _SideBySideDot(dot1, dot2, name1=name1, name2=name2)


def _save_side_by_side(
    dot1: Any,
    dot2: Any,
    *,
    name1: str,
    name2: str,
    save: PathLike | tuple[PathLike, PathLike],
) -> None:
    if isinstance(save, tuple):
        if len(save) != 2:
            raise BNMInputError(
                f"plot_side_by_side: save tuple must have exactly two entries; got {len(save)}."
            )
        _save_dot(dot1, save[0])
        _save_dot(dot2, save[1])
        return

    p = Path(save)
    suffix = p.suffix
    if not suffix:
        raise BNMInputError(
            f"plot_side_by_side: save path {p!s} has no extension; can't infer format."
        )
    stem = p.with_suffix("")
    p1 = stem.with_name(f"{stem.name}_{name1}").with_suffix(suffix)
    p2 = stem.with_name(f"{stem.name}_{name2}").with_suffix(suffix)
    _save_dot(dot1, p1)
    _save_dot(dot2, p2)
